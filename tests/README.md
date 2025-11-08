# Test Suite Documentation

This folder contains all test scripts for the Journal Recommendation System.

## Test Scripts

### Core Tests

- **test_api.py** - Basic API functionality tests
- **test_api_endpoints.py** - Comprehensive API endpoint testing
- **test_api_fix.py** - API bug fix verification tests
- **test_ranking_functions.py** - Algorithm comparison and ranking tests
- **test_advanced_features.py** - Advanced feature tests

### Database Tests

- **test_crossdisciplinary_db.py** - Cross-disciplinary database tests with 1,776 journals
- **check_db.py** - Database integrity and statistics checker
- **verify_vectors.py** - Vector dimension consistency verification

### UI Tests

- **test_dashboard.py** - Streamlit dashboard functionality tests

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/

# Or run individual tests
python tests/test_api_endpoints.py
python tests/test_crossdisciplinary_db.py
```

### Run Specific Test Categories
```bash
# API tests only
python -m pytest tests/test_api*.py

# Database tests only
python tests/check_db.py
python tests/verify_vectors.py
```

## Test Coverage

- **API Endpoints**: Complete coverage of all REST endpoints
- **Database**: 1,776 journals across 11 research fields
- **Ranking Algorithms**: 6-component scoring system validation
- **Vector Consistency**: 100% BERT (384-dim) and TF-IDF (1,651-dim) verification
- **Cross-Disciplinary**: Tests across CS, Biology, Medicine, Environmental Science, etc.

## Requirements

All tests require the main project dependencies:
```bash
pip install -r requirements.txt
```

## Test Database

Tests use: `data/journal_rec.db` (37.55 MB, 1,776 journals)
