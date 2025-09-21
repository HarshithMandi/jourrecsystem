# Journal Recommender System

A FastAPI-based journal recommendation system that uses machine learning to suggest relevant academic journals based on abstract text.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Virtual environment (already configured)

### Installation
The project is already set up with all dependencies installed. To verify:

```bash
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Test the installation
python scripts/test_all.py
```

### Usage

#### 1. Initialize Database
```bash
python scripts/init_db.py
```

#### 2. Start the API Server
```bash
python scripts/start_server.py
```

The server will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/ping

#### 3. Test with Sample Query
```bash
python scripts/sample_query.py
```

#### 4. Ingest Journal Data (Optional)
```bash
python scripts/ingest_openalex.py
```

#### 5. Build Vectors for Recommendations (After Ingesting Data)
```bash
python scripts/build_vectors.py
```

## 📁 Project Structure

```
project-1/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── core/
│   │   └── config.py          # Configuration settings
│   ├── models/
│   │   ├── base.py            # SQLAlchemy base and engine
│   │   └── entities.py        # Database models
│   ├── services/
│   │   └── recommender.py     # Recommendation logic
│   └── main.py               # FastAPI application
├── scripts/
│   ├── init_db.py            # Database initialization
│   ├── ingest_openalex.py    # Data ingestion from OpenAlex
│   ├── build_vectors.py      # Vector generation
│   ├── sample_query.py       # Test query
│   ├── test_all.py          # Comprehensive test suite
│   └── start_server.py      # Server startup script
├── data/
│   └── journal_rec.db       # SQLite database
└── requirements.txt         # Python dependencies
```

## 🔧 API Endpoints

### POST /api/recommend
Recommend journals based on an abstract.

**Request Body:**
```json
{
  "abstract": "Your research abstract here (50-5000 characters)"
}
```

**Response:**
```json
[
  {
    "journal": "Journal Name",
    "similarity": 0.85
  }
]
```

### GET /ping
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "db": "/path/to/database"
}
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python scripts/test_all.py
```

This tests:
- ✅ Database connectivity
- ✅ Model imports
- ✅ Configuration loading
- ✅ FastAPI application
- ✅ Data insertion/retrieval
- ✅ Recommendation service
- ✅ End-to-end functionality

## 🔍 Features

- **Hybrid Recommendation**: Combines TF-IDF and BERT embeddings
- **RESTful API**: FastAPI with automatic documentation
- **Efficient Storage**: SQLite database with optimized queries
- **Scalable**: Batch processing for vector generation
- **Configurable**: Environment-based configuration
- **Well-tested**: Comprehensive test suite

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure you're running scripts from the project root directory.

2. **Database issues**: Reinitialize the database:
   ```bash
   python scripts/init_db.py
   ```

3. **API server not starting**: Check if port 8000 is available or modify the port in `start_server.py`.

4. **Empty recommendations**: The database needs journal data. Run the ingestion and vector building scripts.

## 📊 Data Flow

1. **Ingestion**: `ingest_openalex.py` fetches journal data from OpenAlex API
2. **Vector Building**: `build_vectors.py` generates TF-IDF and BERT embeddings
3. **Recommendation**: `recommender.py` computes similarity scores
4. **API**: `routes.py` serves recommendations via HTTP endpoints

## ⚙️ Configuration

Settings can be modified in `app/core/config.py` or via environment variables:

- `DB_PATH`: Database file path
- `OPENALEX_EMAIL`: Email for OpenAlex API
- `TOP_K`: Number of recommendations to return
- `USE_GPU`: Enable GPU for BERT (if available)

## 🎯 Current Status

✅ **FULLY FUNCTIONAL** - All components tested and working:
- Database initialization ✅
- Model imports ✅
- FastAPI server ✅
- Recommendation engine ✅
- API endpoints ✅
- Test suite ✅

The system is ready for production use!
