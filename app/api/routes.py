from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.recommender import rank_journals
from typing import List

router = APIRouter()

class RecoRequest(BaseModel):
    abstract: str = Field(..., min_length=50, max_length=5000)

class RecoResponse(BaseModel):
    journal: str
    similarity: float

@router.post("/recommend", response_model=List[RecoResponse])
def recommend(req: RecoRequest):
    try:
        return rank_journals(req.abstract)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
