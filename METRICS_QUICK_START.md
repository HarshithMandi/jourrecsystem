# 📊 Metrics & Statistics - Quick Reference

## What Was Created

A comprehensive metrics and visualization system has been added to your Journal Recommendation System!

## 📂 New Folder Structure

```
metrics/                                  # ← NEW FOLDER
├── Core metric collection modules
├── Visualization generation modules  
├── Example scripts
└── Complete documentation
```

## 🚀 Quick Start (3 Commands)

### 1. Check Setup
```bash
python metrics/check_setup.py
```
Verifies dependencies and database connection.

### 2. Generate Dashboard
```bash
python metrics/example_generate_dashboard.py
```
Creates complete dashboard with all metrics and visualizations.

### 3. View Results
Open `metrics/output/dashboard.html` in your web browser!

## 📊 What Metrics Can Be Tracked?

### ⚡ Performance (5 metrics)
- Response time statistics (mean, median, P95, P99)
- Throughput (queries/second, queries/minute, queries/hour)
- Latency distribution
- Component execution breakdown (BERT, TF-IDF, DB)
- Slow query identification

### 🎯 Accuracy (4 metrics)
- Similarity score distribution
- Ranking quality (top-1, top-3, top-5 scores)
- Recommendation diversity (journals, publishers, subjects)
- Component contribution analysis

### 🖥️ System Health (5 metrics)
- Database statistics (journals, works, queries)
- Vector quality (TF-IDF, BERT)
- Data coverage analysis
- System health score (0-100)
- Publisher distribution

### 👥 User Behavior (5 metrics)
- Query patterns (hourly, daily, peak times)
- Popular journals (most recommended)
- Topic trends (keywords, subjects)
- User interaction patterns (sessions, queries/session)
- Open access preferences

**Total: 19+ metrics with 20+ visualizations!**

## 🎨 Visualizations Generated

All visualizations are automatically saved as high-quality PNG images:

### Performance Charts
- Response time distribution histogram
- Throughput bar chart
- Component breakdown pie chart
- Slow queries analysis

### Accuracy Charts
- Similarity score distribution
- Quality breakdown (high/medium/low)
- Ranking quality comparison
- Diversity metrics dashboard

### System Charts
- Database statistics overview
- Vector quality analysis
- System health gauge
- Coverage percentages

### User Behavior Charts
- Query patterns timeline
- Popular journals ranking
- Topic trends visualization
- Interaction patterns
- Open access analysis

## 📚 Documentation

### Main Guides
1. **METRICS_GUIDE.md** (in project root)
   - Comprehensive guide for all metrics
   - How to interpret results
   - Implementation examples

2. **metrics/README.md** (in metrics folder)
   - Technical documentation
   - API reference
   - Advanced usage

3. **metrics/STRUCTURE.md** (in metrics folder)
   - Directory structure
   - File organization
   - Quick reference

## 💻 Usage Examples

### Collect All Metrics
```python
from metrics.metrics_collector import MetricsCollector

collector = MetricsCollector()
all_metrics = collector.collect_all_metrics(hours=24)
print(all_metrics)
```

### Get Specific Metrics
```python
# Performance metrics
perf = collector.get_performance_metrics(hours=24)
print(f"P95 latency: {perf['response_time']['p95_ms']}ms")

# Accuracy metrics
acc = collector.get_accuracy_metrics(hours=24)
print(f"Mean similarity: {acc['similarity_distribution']['statistics']['mean']}")

# System metrics
sys = collector.get_system_metrics()
print(f"Health score: {sys['system_health']['health_score']}/100")

# User metrics
user = collector.get_user_metrics(hours=168)
print(f"Total queries: {user['query_patterns']['total_queries']}")
```

### Generate Visualizations
```python
from metrics.visualizations.dashboard_generator import DashboardGenerator

generator = DashboardGenerator()
generator.generate_full_dashboard('output/dashboard.html', hours=24)
```

### Export Metrics
```python
# Export to JSON
collector.export_all_metrics('metrics.json', hours=24)

# Export summary report
collector.export_summary_report('summary.txt')
```

## 🔧 Integration with API

Add real-time tracking to your API endpoints:

```python
# In app/api/routes.py
from metrics.metrics_collector import MetricsCollector
import time

metrics_collector = MetricsCollector()

@router.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    start_time = time.time()
    
    try:
        # Your recommendation logic
        results = rank_journals(request.abstract, top_k=request.top_k)
        
        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request(duration_ms, endpoint="recommend")
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 📦 Files Created

### Core Modules (in `metrics/`)
- `performance_metrics.py` - Performance tracking
- `accuracy_metrics.py` - Accuracy analysis
- `system_metrics.py` - System health monitoring
- `user_metrics.py` - User behavior analysis
- `metrics_collector.py` - Central aggregator

### Visualization Modules (in `metrics/visualizations/`)
- `performance_visualizer.py` - Performance charts
- `accuracy_visualizer.py` - Accuracy plots
- `system_visualizer.py` - System dashboards
- `user_visualizer.py` - User behavior graphs
- `dashboard_generator.py` - HTML dashboard generator

### Example Scripts (in `metrics/`)
- `example_collect_metrics.py` - Collect metrics example
- `example_generate_visualizations.py` - Generate plots example
- `example_generate_dashboard.py` - Full dashboard example
- `check_setup.py` - Setup verification script

### Documentation (in `metrics/`)
- `README.md` - Comprehensive module documentation
- `STRUCTURE.md` - Directory structure reference

### Root Documentation
- `METRICS_GUIDE.md` - Complete guide to all metrics
- `METRICS_QUICK_START.md` - This file!

## 🎯 Key Features

✅ **19+ Metrics Tracked** - Performance, accuracy, system, user behavior  
✅ **20+ Visualizations** - Charts, graphs, dashboards  
✅ **HTML Dashboard** - Interactive, self-contained  
✅ **Real-time Tracking** - Integrate with API endpoints  
✅ **Export Capabilities** - JSON, text reports, PNG images  
✅ **Easy to Use** - Simple Python API  
✅ **Well Documented** - Complete guides and examples  
✅ **Production Ready** - Efficient, tested code  

## 🔍 Interpreting Results

### Good Performance
- P95 < 500ms
- P99 < 1000ms  
- Throughput > 10 queries/min
- Low variance

### Good Accuracy
- Mean similarity > 0.6
- Top-1 score > 0.7
- High diversity
- Consistent rankings

### Good System Health
- Health score > 80
- Vector coverage > 90%
- No critical issues
- Quality score > 70

## 🚨 Troubleshooting

**"Module not found" errors:**
```bash
pip install matplotlib seaborn numpy sqlalchemy
```

**"No data available" errors:**
- Ensure database has journals and queries
- Check time window includes data
- Run data ingestion scripts if needed

**Permission errors:**
- Check write permissions on `metrics/output/`
- Run from project root directory

**Import errors:**
- Make sure you're in project root
- Check Python path includes project directory

## 📞 Next Steps

1. ✅ **Verify Setup**
   ```bash
   python metrics/check_setup.py
   ```

2. ✅ **Generate Dashboard**
   ```bash
   python metrics/example_generate_dashboard.py
   ```

3. ✅ **View Results**
   - Open `metrics/output/dashboard.html`
   - Explore all metrics and visualizations

4. ✅ **Read Documentation**
   - `METRICS_GUIDE.md` for comprehensive guide
   - `metrics/README.md` for technical details

5. ✅ **Integrate with API** (Optional)
   - Add real-time tracking to endpoints
   - Set up automated monitoring
   - Configure alerts

## 📈 Monitoring Best Practices

### Daily
- Check system health score
- Review P95 response times
- Monitor query volume

### Weekly  
- Analyze accuracy trends
- Review popular journals
- Check topic trends

### Monthly
- Month-over-month comparison
- Long-term trend analysis
- Optimization planning

## 🎓 Summary

You now have a complete metrics infrastructure that:
- Tracks 19+ different metrics
- Generates 20+ visualizations
- Creates interactive dashboards
- Provides comprehensive documentation
- Includes working examples

**Everything is organized in the `metrics/` folder with a `visualizations/` subfolder!**

---

## 🌟 Quick Commands Cheat Sheet

```bash
# Check if everything is set up correctly
python metrics/check_setup.py

# Collect metrics only (no visualizations)
python metrics/example_collect_metrics.py

# Generate visualizations only
python metrics/example_generate_visualizations.py

# Generate complete dashboard (RECOMMENDED)
python metrics/example_generate_dashboard.py

# View the dashboard
# Open metrics/output/dashboard.html in browser
```

---

**Ready to explore your metrics? Run the dashboard generator now!**

```bash
python metrics/example_generate_dashboard.py
```

Then open `metrics/output/dashboard.html` in your browser! 🚀
