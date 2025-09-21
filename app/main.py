from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings

app = FastAPI(title="Journal Recommender")
app.include_router(router, prefix="/api")

# health endpoint
@app.get("/ping")
def ping():
    return {"status":"ok", "db": settings.DB_PATH}
