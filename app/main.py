from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
import time
import traceback

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Journal Recommender API",
    description="A machine learning-powered API for recommending academic journals based on research abstracts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url),
            "timestamp": time.time()
        }
    )

# Include API routes
app.include_router(router, prefix="/api", tags=["recommendations"])

# Enhanced health endpoint
@app.get("/ping", tags=["health"])
def ping():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "database": settings.DB_PATH,
        "service": "Journal Recommender API",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/", tags=["info"])
def root():
    """API information endpoint"""
    return {
        "message": "Welcome to Journal Recommender API",
        "docs": "/docs",
        "health": "/ping",
        "api_version": "1.0.0",
        "endpoints": {
            "recommend": "/api/recommend",
            "batch_recommend": "/api/batch-recommend",
            "stats": "/api/stats"
        }
    }
