from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator
from app.services.recommender import rank_journals
from app.models.base import SessionLocal
from app.models.entities import Journal, QueryRun, Recommendation
from sqlalchemy import func
from typing import List, Optional
import time
import re

router = APIRouter()

# Request Models
class RecommendationRequest(BaseModel):
    abstract: str = Field(..., min_length=50, max_length=5000, description="Research abstract text")
    top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    
    @validator('abstract')
    def validate_abstract(cls, v):
        # Remove excessive whitespace and check for meaningful content
        cleaned = re.sub(r'\s+', ' ', v.strip())
        if len(cleaned.split()) < 10:
            raise ValueError('Abstract must contain at least 10 words')
        return cleaned

class BatchRecommendationRequest(BaseModel):
    abstracts: List[str] = Field(..., description="List of research abstracts")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of recommendations per abstract")
    
    @validator('abstracts')
    def validate_abstracts(cls, v):
        if not (1 <= len(v) <= 10):
            raise ValueError('Must provide between 1 and 10 abstracts')
        return v

# Response Models
class JournalRecommendation(BaseModel):
    journal_name: str = Field(..., description="Name of the recommended journal")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., description="Rank in the recommendation list")
    
    class Config:
        json_schema_extra = {
            "example": {
                "journal_name": "Nature Machine Intelligence",
                "similarity_score": 0.85,
                "rank": 1
            }
        }

class RecommendationResponse(BaseModel):
    query_id: int = Field(..., description="Unique identifier for this query")
    recommendations: List[JournalRecommendation] = Field(..., description="List of journal recommendations")
    total_journals: int = Field(..., description="Total number of journals in database")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: float = Field(..., description="Unix timestamp of the request")

class BatchRecommendationResponse(BaseModel):
    results: List[RecommendationResponse] = Field(..., description="Recommendations for each abstract")
    total_processing_time_ms: float = Field(..., description="Total processing time for all abstracts")
    timestamp: float = Field(..., description="Unix timestamp of the request")

class DatabaseStats(BaseModel):
    total_journals: int
    total_queries: int
    total_recommendations: int
    journals_with_profiles: int
    avg_similarity_score: Optional[float]

# Main recommendation endpoint
@router.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """
    Get journal recommendations based on research abstract.
    
    This endpoint uses a hybrid ML approach combining TF-IDF and BERT embeddings
    to find the most relevant journals for your research.
    """
    start_time = time.time()
    
    try:
        # Get recommendations
        results = rank_journals(request.abstract, top_k=request.top_k or 10)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No recommendations found. The database might be empty or your query is too specific."
            )
        
        # Get query ID from the most recent query  
        db = SessionLocal()
        try:
            latest_query = db.query(QueryRun).order_by(QueryRun.id.desc()).first()
            if latest_query:
                query_id_value = getattr(latest_query, 'id', 0)
            else:
                query_id_value = 0
            
            # Get total journal count
            total_journals = db.query(Journal).count()
        finally:
            db.close()
        
        # Format response
        recommendations = [
            JournalRecommendation(
                journal_name=result["journal"],
                similarity_score=result["similarity"], 
                rank=idx + 1
            )
            for idx, result in enumerate(results)
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            query_id=query_id_value,
            recommendations=recommendations,
            total_journals=total_journals,
            processing_time_ms=round(processing_time, 2),
            timestamp=time.time()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# Batch recommendation endpoint
@router.post("/batch-recommend", response_model=BatchRecommendationResponse)
def get_batch_recommendations(request: BatchRecommendationRequest):
    """
    Get journal recommendations for multiple abstracts at once.
    
    Useful for processing multiple papers or comparing recommendations.
    """
    start_time = time.time()
    
    try:
        results = []
        
        for abstract in request.abstracts:
            # Create individual request
            individual_request = RecommendationRequest(
                abstract=abstract,
                top_k=request.top_k
            )
            
            # Get recommendations (reuse the single recommendation logic)
            try:
                recommendation_response = get_recommendations(individual_request)
                results.append(recommendation_response)
            except HTTPException as e:
                # Handle individual failures gracefully
                results.append(RecommendationResponse(
                    query_id=0,
                    recommendations=[],
                    total_journals=0,
                    processing_time_ms=0,
                    timestamp=time.time()
                ))
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchRecommendationResponse(
            results=results,
            total_processing_time_ms=round(total_processing_time, 2),
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch recommendation failed: {str(e)}")

# Database statistics endpoint
@router.get("/stats", response_model=DatabaseStats)
def get_database_stats():
    """
    Get statistics about the journal database and recommendation history.
    """
    try:
        db = SessionLocal()
        
        total_journals = db.query(Journal).count()
        total_queries = db.query(QueryRun).count()
        total_recommendations = db.query(Recommendation).count()
        
        # Count journals with profiles (have ML vectors)
        from app.models.entities import JournalProfile
        journals_with_profiles = db.query(JournalProfile).filter(
            JournalProfile.tfidf_vector.isnot(None),
            JournalProfile.bert_vector.isnot(None)
        ).count()
        
        # Calculate average similarity score
        avg_similarity = db.query(func.avg(Recommendation.similarity)).scalar()
        
        db.close()
        
        return DatabaseStats(
            total_journals=total_journals,
            total_queries=total_queries,
            total_recommendations=total_recommendations,
            journals_with_profiles=journals_with_profiles,
            avg_similarity_score=round(avg_similarity, 3) if avg_similarity else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Legacy endpoint for backward compatibility
@router.post("/recommend-simple")
def recommend_simple(req: RecommendationRequest):
    """
    Simple recommendation endpoint (legacy format)
    """
    try:
        results = rank_journals(req.abstract, top_k=req.top_k or 10)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
