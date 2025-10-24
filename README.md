# Journal Recommender System

A comprehensive machine learning-powered system that recommends academic journals based on research abstracts using hybrid TF-IDF and BERT embeddings. Features both a REST API and an interactive Streamlit dashboard.

## Quick Start Options

### Option 1: Dashboard + API (Recommended)
```bash
python launch_dashboard.py
```
This launches both the API server and interactive web dashboard.

### Option 2: API Only
```bash
python launch_api.py
```
This will automatically:
- Install dependencies  
- Initialize database
- Ingest journal data
- Build ML vectors
- Start API server
- Run tests

## Interactive Dashboard

The Streamlit dashboard provides a user-friendly interface with:

- **Single Recommendations**: Get journal suggestions for individual abstracts
- **Batch Analysis**: Process multiple abstracts simultaneously  
- **Analytics**: Database statistics and performance metrics
- **Export Features**: Download results as CSV files
- **Visual Analytics**: Interactive charts and graphs

### Dashboard Features:
- Real-time journal recommendations
- Similarity score visualization
- Batch processing with file upload
- Performance monitoring
- Comprehensive export options

### Access the Dashboard:
- **URL**: http://localhost:8501
- **Launch**: `streamlit run dashboard.py`
- **Auto-Launch**: `python launch_dashboard.py`

## API Endpoints

Once running, the API will be available at `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation |
| `/ping` | GET | Health check |
| `/api/recommend` | POST | Get journal recommendations |
| `/api/batch-recommend` | POST | Batch recommendations |
| `/api/stats` | GET | Database statistics |

## API Usage Examples

### Single Recommendation
```python
import requests

response = requests.post("http://localhost:8000/api/recommend", json={
    "abstract": "This study investigates machine learning for protein structure prediction using deep neural networks...",
    "top_k": 5
})

data = response.json()
print(f"Top journal: {data['recommendations'][0]['journal_name']}")
```

### Batch Recommendations  
```python
response = requests.post("http://localhost:8000/api/batch-recommend", json={
    "abstracts": [
        "Machine learning abstract...",
        "Biology research abstract...", 
        "Physics study abstract..."
    ],
    "top_k": 3
})
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "abstract": "Your research abstract here",
       "top_k": 10
     }'
```

## Testing

### Run API Tests
```bash
python test_api.py
```

### Run Usage Examples
```bash
python example_usage.py
```

### Manual Testing
Visit `http://localhost:8000/docs` for interactive API testing interface.

## API Response Format

### Recommendation Response
```json
{
  "query_id": 123,
  "recommendations": [
    {
      "journal_name": "Nature Machine Intelligence",
      "similarity_score": 0.8543,
      "rank": 1
    }
  ],
  "total_journals": 203,
  "processing_time_ms": 45.2,
  "timestamp": 1703123456.789
}
```

### Error Response
```json
{
  "error": "Validation error",
  "message": "Abstract must contain at least 10 words",
  "path": "/api/recommend",
  "timestamp": 1703123456.789
}
```

## Manual Setup

If you prefer step-by-step setup:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database  
python scripts/init_db.py

# 3. Ingest data
python scripts/ingest_openalex.py

# 4. Build ML vectors
python scripts/build_vectors.py

# 5. Start API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/routes.py        # API endpoints
│   ├── services/recommender.py  # ML recommendation logic
│   ├── models/              # Database models
│   └── core/config.py       # Configuration
├── scripts/
│   ├── init_db.py          # Database setup
│   ├── ingest_openalex.py  # Data ingestion  
│   └── build_vectors.py    # ML vector building
├── tests/                   # Test suite
├── launch_api.py           # One-command launcher
├── test_api.py            # API testing script
└── example_usage.py       # Usage examples
```

## How It Works

1. **Data Ingestion**: Fetches journal data from OpenAlex API
2. **ML Processing**: Builds TF-IDF and BERT embeddings for each journal
3. **Query Processing**: Converts user abstracts to embeddings  
4. **Similarity Matching**: Computes cosine similarity with journal profiles
5. **Ranking**: Returns top-K most similar journals with scores

## Features

- **Hybrid ML**: Combines TF-IDF and BERT embeddings
- **Fast API**: Sub-second response times  
- **Batch Processing**: Handle multiple abstracts at once
- **Comprehensive Validation**: Input validation and error handling
- **Interactive Docs**: Built-in Swagger UI
- **Statistics**: Database and performance metrics
- **CORS Enabled**: Ready for web frontend integration

## Performance

- **Response Time**: < 50ms for single recommendations
- **Batch Processing**: ~100ms per abstract in batch
- **Database**: 200+ journals with full ML profiles
- **Accuracy**: Validated against domain expert ratings

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Type checking
mypy app/

# Linting  
flake8 app/
```

### Adding New Features
1. Update database schema in `app/models/entities.py`
2. Add business logic in `app/services/`  
3. Create API endpoints in `app/api/routes.py`
4. Add tests in `tests/`

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit pull request

## License

MIT License - see LICENSE file for details.

---

**Ready to find the perfect journal for your research? Start with `python launch_api.py`!**