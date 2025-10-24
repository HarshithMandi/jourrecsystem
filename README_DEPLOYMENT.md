# 📚 Journal Recommendation System - Streamlit Deployment

A machine learning-powered journal recommendation system using TF-IDF and BERT embeddings.

## 🚀 Live Demo

**Deployed on Streamlit Community Cloud:** [Your App URL will be here]

## 🌟 Features

- **Smart Recommendations**: Combines TF-IDF and BERT embeddings for accurate journal matching
- **Advanced Analysis**: Detailed similarity breakdowns and ranking comparisons  
- **Interactive Visualizations**: Scatter plots, bar charts, and heatmaps
- **Real-time Processing**: Fast recommendations using pre-computed vectors
- **Clean Database**: 200+ curated journals from OpenAlex

## 📊 How It Works

1. **Input**: Enter your research abstract
2. **Processing**: System analyzes text using TF-IDF + BERT embeddings
3. **Matching**: Compares against 200+ journal profiles using cosine similarity
4. **Results**: Get ranked recommendations with detailed scoring

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/HarshithMandi/jourrecsystem.git
cd jourrecsystem
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Initialize database (if needed):**
```bash
cd scripts
python init_db.py
python ingest_openalex.py
python build_vectors.py
```

4. **Run locally:**
```bash
streamlit run app_standalone.py
```

## 🌐 Deployment on Streamlit Community Cloud

### Step 1: Prepare Your Repository

Ensure your GitHub repository has:
- ✅ `requirements.txt` 
- ✅ `app_standalone.py` (main Streamlit app)
- ✅ All source code and database files

### Step 2: Deploy

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Configure deployment:**
   - Repository: `HarshithMandi/jourrecsystem`
   - Branch: `main`
   - Main file path: `app_standalone.py`
   - App URL: Choose your custom URL

5. **Click "Deploy!"**

### Step 3: Monitor Deployment

- Deployment typically takes 2-5 minutes
- Check logs for any issues
- App will be available at your chosen URL

## 📁 Project Structure

```
jourrecsystem/
├── app/
│   ├── api/              # FastAPI routes
│   ├── core/             # Configuration
│   ├── models/           # Database models
│   └── services/         # ML recommendation logic
├── data/                 # Database and data files
├── scripts/              # Data ingestion and processing
├── app_standalone.py     # Main Streamlit app (deployment)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
OPENALEX_EMAIL=your-email@domain.com
DATABASE_URL=sqlite:///data/journal_rec.db
```

### Streamlit Configuration
Configuration is handled in `.streamlit/config.toml`:
```toml
[general]
email = ""

[server]
headless = true
port = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
```

## 🤖 Technology Stack

- **Frontend**: Streamlit
- **ML/NLP**: scikit-learn, sentence-transformers, BERT
- **Database**: SQLite with SQLAlchemy ORM  
- **Visualization**: Plotly
- **Data**: OpenAlex API
- **Deployment**: Streamlit Community Cloud

## 🎯 API Endpoints (Local Development)

When running with FastAPI backend:

- `GET /ping` - Health check
- `POST /api/recommend` - Get recommendations
- `POST /api/recommend-detailed` - Detailed analysis
- `POST /api/compare-rankings` - Compare ranking methods

## 📈 Performance

- **Response Time**: < 2 seconds for recommendations
- **Database Size**: ~50MB (200 journals + vectors)
- **Memory Usage**: ~500MB (includes BERT model)

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies in `requirements.txt` are installed
2. **Database Issues**: Run database initialization scripts
3. **Memory Issues**: TF-IDF/BERT models require ~500MB RAM
4. **Slow Performance**: Pre-computed vectors should load quickly

### Deployment Issues

1. **App Won't Start**: Check Streamlit logs for dependency issues
2. **Database Not Found**: Ensure `data/journal_rec.db` is in repository
3. **Import Errors**: Verify all Python modules are properly structured

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Harshith Mandi**
- GitHub: [@HarshithMandi](https://github.com/HarshithMandi)
- Repository: [jourrecsystem](https://github.com/HarshithMandi/jourrecsystem)

---

**Built with ❤️ using Streamlit and modern ML techniques**