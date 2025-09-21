from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

class Settings(BaseSettings):
    ENV: str = "development"
    DB_PATH: str = str(DATA_DIR / "journal_rec.db")
    SQL_ECHO: bool = False           # set True for debugging
    OPENALEX_EMAIL: str = "you@org.com"
    TOP_K: int = 10                  # journals returned
    USE_GPU: bool = False            # torch device flag

    class Config:
        env_file = str(BASE_DIR / ".env")

settings = Settings()
