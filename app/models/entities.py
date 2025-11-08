import uuid, json, datetime as dt
from sqlalchemy import (Column, Integer, String, Text, Boolean, Float,
                        DateTime, ForeignKey, Index)
from sqlalchemy.orm import relationship
from app.models.base import Base

class Journal(Base):
    __tablename__ = "journals"
    id = Column(Integer, primary_key=True)
    openalex_id = Column(String(64), unique=True, index=True, nullable=False)  # For OpenAlex or synthetic IDs (e.g., ojs:<setSpec>)
    source_type = Column(String(32), index=True, default="openalex")          # 'openalex' | 'ojs' | 'manual'
    external_id = Column(String(128), index=True)                              # Raw external identifier (OJS setSpec, etc.)
    name = Column(String(512), nullable=False)
    display_name = Column(String(512))
    issn = Column(String(32), index=True)
    eissn = Column(String(32), index=True)
    is_open_access = Column(Boolean, default=False)
    publisher = Column(String(255))
    impact_factor = Column(Float)
    subjects = Column(Text)              # JSON list as str
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow,
                        onupdate=dt.datetime.utcnow)

    profile = relationship("JournalProfile", back_populates="journal",
                           uselist=False, cascade="all,delete")
    works = relationship("Work", back_populates="journal",
                         cascade="all,delete")

class JournalProfile(Base):
    __tablename__ = "journal_profiles"
    id = Column(Integer, primary_key=True)
    journal_id = Column(Integer, ForeignKey("journals.id", ondelete="CASCADE"),
                        unique=True, nullable=False)
    scope_text = Column(Text)
    tfidf_vector = Column(Text)          # array→json str
    bert_vector = Column(Text)           # array→json str
    total_articles = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow,
                        onupdate=dt.datetime.utcnow)

    journal = relationship("Journal", back_populates="profile")

class Work(Base):
    __tablename__ = "works"
    id = Column(Integer, primary_key=True)
    openalex_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    publication_year = Column(Integer, index=True)
    journal_id = Column(Integer, ForeignKey("journals.id", ondelete="CASCADE"))
    journal = relationship("Journal", back_populates="works")

class QueryRun(Base):
    __tablename__ = "query_runs"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    model_used = Column(String(32))
    timestamp = Column(DateTime, default=dt.datetime.utcnow)

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("query_runs.id", ondelete="CASCADE"))
    journal_id = Column(Integer, ForeignKey("journals.id", ondelete="CASCADE"))
    similarity = Column(Float, nullable=False)
    rank = Column(Integer)

Index("idx_similarity_desc", Recommendation.similarity.desc())
