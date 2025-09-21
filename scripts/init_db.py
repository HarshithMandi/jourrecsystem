import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.base import Base, engine
from app.models.entities import Journal, JournalProfile, Work, QueryRun, Recommendation
from sqlalchemy import text

def init():
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_pub_year ON works(publication_year)"
        ))
    print("SQLite schema ready!")

if __name__ == "__main__":
    init()
