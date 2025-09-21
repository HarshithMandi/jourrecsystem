from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

# Create the SQLAlchemy engine
engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    echo=settings.SQL_ECHO,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create the declarative base class
Base = declarative_base()

# Create a sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
