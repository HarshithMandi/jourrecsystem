# Metrics and Analytics Module

Comprehensive metrics tracking and visualization system for the Journal Recommendation System.

## 📊 Overview

This module provides detailed metrics, statistics, and visualizations to monitor and analyze:

- **Performance**: Response times, throughput, latency distributions
- **Accuracy**: Similarity scores, ranking quality, recommendation diversity
- **System Health**: Database statistics, vector quality, data coverage
- **User Behavior**: Query patterns, popular journals, topic trends

## 🚀 Quick Start

### 1. Generate Complete Dashboard

The easiest way to get all metrics and visualizations:

```python
python metrics/example_generate_dashboard.py
```

This creates an interactive HTML dashboard with all metrics and visualizations in `metrics/output/dashboard.html`.

### 2. Collect Metrics Only

To collect metrics without generating visualizations:

```python
python metrics/example_collect_metrics.py
```

Exports:
- `metrics/output/all_metrics.json` - Complete metrics in JSON format
- `metrics/output/summary_report.txt` - Human-readable summary

### 3. Generate Visualizations Only

To generate all visualization plots:

```python
python metrics/example_generate_visualizations.py
```

Saves PNG files to `metrics/output/visualizations/`

## 📁 Structure

```
metrics/
├── __init__.py                      # Module initialization
├── performance_metrics.py           # Performance tracking
├── accuracy_metrics.py              # Accuracy analysis
├── system_metrics.py                # System health monitoring
├── user_metrics.py                  # User behavior analysis
├── metrics_collector.py             # Central metrics aggregation
│
├── visualizations/                  # Visualization submodule
│   ├── __init__.py
│   ├── performance_visualizer.py   # Performance plots
│   ├── accuracy_visualizer.py      # Accuracy charts
│   ├── system_visualizer.py        # System health dashboards
│   ├── user_visualizer.py          # User behavior plots
│   └── dashboard_generator.py      # HTML dashboard generator
│
├── example_collect_metrics.py      # Example: Collect metrics
├── example_generate_visualizations.py  # Example: Generate plots
├── example_generate_dashboard.py   # Example: Full dashboard
└── README.md                        # This file
```

## 📈 Metrics Categories

### 1. Performance Metrics

**What it tracks:**
- Response time statistics (mean, median, p95, p99)
- Throughput (queries per second/minute/hour)
- Latency distribution
- Component execution breakdown (BERT, TF-IDF, DB operations)
- Slow query identification

**Usage:**
```python
from metrics.performance_metrics import PerformanceMetrics

perf = PerformanceMetrics()

# Record request timing
perf.record_request_time(duration_ms=245.6, endpoint="recommend")

# Get statistics
stats = perf.get_response_time_stats(hours=24)
print(f"P95 latency: {stats['p95_ms']}ms")

# Get throughput
throughput = perf.get_throughput_metrics(hours=24)
print(f"Queries per hour: {throughput['queries_per_hour']}")
```

**Visualizations:**
- Response time distribution histogram
- Throughput timeline
- Component breakdown pie chart
- Slow queries bar chart

### 2. Accuracy Metrics

**What it tracks:**
- Similarity score distributions
- Ranking quality (top-1, top-3, top-5 scores)
- Recommendation diversity (journals, publishers, subjects)
- Quality breakdown (high/medium/low quality recommendations)

**Usage:**
```python
from metrics.accuracy_metrics import AccuracyMetrics

acc = AccuracyMetrics()

# Get similarity distribution
sim_dist = acc.get_similarity_score_distribution(hours=24)
print(f"Mean similarity: {sim_dist['statistics']['mean']}")

# Get ranking quality
ranking = acc.get_ranking_quality_metrics(hours=24)
print(f"Top-1 avg score: {ranking['top1_avg_score']}")

# Get diversity
diversity = acc.get_recommendation_diversity(hours=24)
print(f"Unique journals: {diversity['unique_journals_recommended']}")
```

**Visualizations:**
- Similarity score distribution
- Quality breakdown pie chart
- Top-K ranking quality comparison
- Diversity metrics dashboard

### 3. System Metrics

**What it tracks:**
- Database statistics (journals, profiles, works, queries)
- Vector quality (dimensions, norms, anomalies)
- Data coverage (completeness percentages)
- System health indicators
- Publisher statistics

**Usage:**
```python
from metrics.system_metrics import SystemMetrics

sys_met = SystemMetrics()

# Get database stats
db_stats = sys_met.get_database_statistics()
print(f"Total journals: {db_stats['journals']['total']}")
print(f"Vector coverage: {db_stats['coverage']['vector_coverage']}%")

# Check system health
health = sys_met.get_system_health_indicators()
print(f"Status: {health['status']}")
print(f"Health score: {health['health_score']}/100")

# Get vector quality
vectors = sys_met.get_vector_quality_metrics()
print(f"TF-IDF sparsity: {vectors['tfidf_vectors']['sparsity']}")
```

**Visualizations:**
- Database statistics dashboard
- Vector quality metrics
- System health gauge
- Coverage percentages

### 4. User Metrics

**What it tracks:**
- Query patterns (hourly/daily distributions, peak times)
- Popular journals (most recommended, highest similarity)
- Topic trends (keywords, subject areas)
- User interaction patterns (sessions, queries per session)
- Open access preferences

**Usage:**
```python
from metrics.user_metrics import UserMetrics

user_met = UserMetrics()

# Get query patterns
patterns = user_met.get_query_patterns(hours=24)
print(f"Peak hour: {patterns['peak_hour']['hour']}:00")
print(f"Total queries: {patterns['total_queries']}")

# Get popular journals
popular = user_met.get_popular_journals(limit=10, hours=168)
for journal in popular['top_journals']:
    print(f"{journal['journal_name']}: {journal['recommendation_count']} recs")

# Get topic trends
trends = user_met.get_topic_trends(hours=168)
print("Top keywords:", list(trends['top_keywords'].keys())[:10])
```

**Visualizations:**
- Query patterns over time
- Popular journals ranking
- Topic trends word cloud
- Interaction patterns dashboard
- Open access preference analysis

## 🎨 Visualizations

### Performance Visualizations

1. **Response Time Stats** - Bar chart showing mean, median, p95, p99, max
2. **Response Time Distribution** - Histogram with percentile breakdown
3. **Throughput Timeline** - Horizontal bar chart of query rates
4. **Component Breakdown** - Pie chart showing time distribution
5. **Slow Queries** - Bar chart of slowest requests

### Accuracy Visualizations

1. **Similarity Distribution** - Histogram with statistics summary
2. **Quality Breakdown** - Pie chart of high/medium/low quality
3. **Ranking Quality** - Bar chart comparing top-K performance
4. **Diversity Metrics** - Multi-panel dashboard

### System Visualizations

1. **Database Stats** - Multi-panel overview
2. **Vector Quality** - TF-IDF and BERT metrics comparison
3. **System Health** - Gauge with health checks

### User Visualizations

1. **Query Patterns** - Time series with peak hours
2. **Popular Journals** - Horizontal bar charts
3. **Topic Trends** - Top keywords and subjects
4. **Interaction Patterns** - Session statistics
5. **Open Access Preference** - Comparison charts

## 🔧 Advanced Usage

### Using MetricsCollector

The `MetricsCollector` provides a unified interface to all metrics:

```python
from metrics.metrics_collector import MetricsCollector

collector = MetricsCollector()

# Collect all metrics
all_metrics = collector.collect_all_metrics(hours=24)

# Get specific categories
performance = collector.get_performance_metrics(hours=24)
accuracy = collector.get_accuracy_metrics(hours=24)
system = collector.get_system_metrics()
user = collector.get_user_metrics(hours=168)

# Get dashboard summary
summary = collector.get_summary_dashboard()
print(summary)

# Export to files
collector.export_all_metrics('output/metrics.json', hours=24)
collector.export_summary_report('output/summary.txt')

# Record metrics in real-time
collector.record_request(duration_ms=123.4, endpoint="recommend")
collector.record_component_time("bert", duration_ms=45.2)
```

### Integrating with API

To track metrics in your API, modify `app/api/routes.py`:

```python
from metrics.metrics_collector import MetricsCollector
import time

# Initialize collector (once, at module level)
metrics_collector = MetricsCollector()

@router.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    start_time = time.time()
    
    try:
        # Your recommendation logic here
        results = rank_journals(request.abstract, top_k=request.top_k)
        
        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request(duration_ms, endpoint="recommend")
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Custom Visualizations

Create your own visualizations:

```python
from metrics.visualizations.performance_visualizer import PerformanceVisualizer
import matplotlib.pyplot as plt

viz = PerformanceVisualizer()

# Custom plot
metrics = collector.get_performance_metrics(hours=24)
viz.plot_response_time_stats(metrics, 'custom_plot.png')

# Or create completely custom visualizations
response_times = metrics['response_time']
plt.figure(figsize=(10, 6))
plt.hist(response_times, bins=50)
plt.xlabel('Response Time (ms)')
plt.ylabel('Frequency')
plt.title('Custom Response Time Distribution')
plt.savefig('my_custom_plot.png')
```

## 📊 Metrics You Can Track

### Performance
- ✅ Response time (mean, median, std, min, max, p95, p99)
- ✅ Throughput (queries/second, queries/minute, queries/hour)
- ✅ Latency distribution with histogram
- ✅ Component breakdown (BERT, TF-IDF, database, etc.)
- ✅ Slow query identification
- ⚠️  Query processing breakdown (requires instrumentation)

### Accuracy
- ✅ Similarity score distribution
- ✅ Quality breakdown (high/medium/low)
- ✅ Top-K ranking quality (top-1, top-3, top-5)
- ✅ Recommendation diversity (journals, publishers, subjects)
- ✅ Open access vs traditional ratio
- ⚠️  Precision@K (requires ground truth labels)
- ⚠️  NDCG (requires relevance scores)
- ⚠️  MRR (requires click data)

### System
- ✅ Database entity counts
- ✅ Vector coverage percentages
- ✅ Vector quality metrics
- ✅ Data completeness analysis
- ✅ System health indicators
- ✅ Publisher statistics
- ✅ Anomaly detection

### User Behavior
- ✅ Query patterns (hourly, daily, peak times)
- ✅ Popular journals (most recommended)
- ✅ Topic trends (keywords, subjects)
- ✅ Session statistics
- ✅ Queries per session
- ✅ Open access preference
- ⚠️  Recommendation acceptance rate (requires click tracking)
- ⚠️  Query refinement patterns (requires session analysis)

**Legend:**
- ✅ Fully implemented
- ⚠️  Requires additional instrumentation or data

## 🔮 Future Enhancements

### Metrics to Add

1. **A/B Testing Metrics**
   - Compare different recommendation algorithms
   - Statistical significance testing
   - Conversion rate optimization

2. **User Feedback Metrics**
   - Click-through rates
   - Recommendation acceptance rates
   - User satisfaction scores

3. **Cost Metrics**
   - API call costs (if using external services)
   - Compute resource usage
   - Storage costs

4. **Model Performance**
   - Embedding quality scores
   - Model drift detection
   - Feature importance analysis

### Visualization Enhancements

1. **Interactive Dashboards**
   - Real-time metrics updates
   - Drill-down capabilities
   - Custom date range selection

2. **Comparative Analysis**
   - Before/after comparisons
   - Multi-version analysis
   - Regression detection

3. **Alerts and Notifications**
   - Threshold-based alerts
   - Anomaly detection alerts
   - Email/Slack notifications

## 🛠️ Dependencies

Required packages (already in `requirements.txt`):
```
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.21.0
sqlalchemy>=1.4.0
```

## 📝 Notes

- Metrics are collected from the database, so make sure your application is logging queries and recommendations
- Visualizations require matplotlib and seaborn
- Dashboard generation creates a self-contained HTML file with embedded images
- For production use, consider:
  - Setting up automated metrics collection (cron jobs)
  - Implementing real-time monitoring
  - Adding alerting for critical metrics
  - Storing historical metrics for trend analysis

## 🤝 Contributing

To add new metrics:

1. Add metric calculation to appropriate module (performance_metrics.py, etc.)
2. Add visualization to corresponding visualizer
3. Update MetricsCollector to include new metric
4. Add to dashboard_generator.py
5. Update this README

## 📚 Examples

See the `example_*.py` files for complete working examples:
- `example_collect_metrics.py` - Collect and export metrics
- `example_generate_visualizations.py` - Generate all plots
- `example_generate_dashboard.py` - Create full HTML dashboard

## 🐛 Troubleshooting

**Problem**: "No data available" errors
- **Solution**: Make sure you have queries in the database and the time window includes them

**Problem**: Matplotlib/visualization errors
- **Solution**: Install required dependencies: `pip install matplotlib seaborn`

**Problem**: Empty visualizations
- **Solution**: Check that metrics are being collected properly and database has data

**Problem**: Path errors
- **Solution**: Run scripts from project root directory

## 📞 Support

For issues or questions about the metrics module, please check:
1. This README
2. Example scripts in the metrics folder
3. Main project documentation

---

**Version**: 1.0  
**Last Updated**: 2025  
**Author**: Journal Recommendation System Team
