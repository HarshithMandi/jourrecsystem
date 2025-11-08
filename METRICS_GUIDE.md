# 📊 Metrics and Statistics Guide

## Comprehensive Guide to Performance and Accuracy Metrics for Journal Recommendation System

---

## 🎯 Executive Summary

This document outlines all metrics and statistics that can be gathered from the Journal Recommendation System to evaluate:

1. **Performance** - How fast and efficient the system is
2. **Accuracy** - How good the recommendations are
3. **System Health** - Overall system status and data quality
4. **User Behavior** - How users interact with the system

---

## 📂 Organization

All metrics-related files are organized in the `metrics/` folder:

```
metrics/
├── Core Modules (Python files that collect metrics)
│   ├── performance_metrics.py      # Speed & throughput tracking
│   ├── accuracy_metrics.py         # Recommendation quality
│   ├── system_metrics.py           # System health monitoring
│   ├── user_metrics.py             # User behavior analysis
│   └── metrics_collector.py        # Central aggregation
│
└── visualizations/ (Subfolder with all visualization code)
    ├── performance_visualizer.py   # Performance charts
    ├── accuracy_visualizer.py      # Accuracy plots
    ├── system_visualizer.py        # System dashboards
    ├── user_visualizer.py          # User behavior graphs
    └── dashboard_generator.py      # HTML dashboard creator
```

---

## 🚀 Quick Start

### Generate Everything at Once

```bash
python metrics/example_generate_dashboard.py
```

This single command will:
- ✅ Collect all metrics
- ✅ Generate all visualizations
- ✅ Create an interactive HTML dashboard
- ✅ Save everything to `metrics/output/`

### View Individual Metrics

```bash
# Collect metrics only (JSON + text report)
python metrics/example_collect_metrics.py

# Generate visualizations only (PNG files)
python metrics/example_generate_visualizations.py
```

---

## 📊 1. PERFORMANCE METRICS

### What They Measure
How fast your system processes requests and serves recommendations.

### Key Metrics

#### A. Response Time Statistics
- **Mean Response Time**: Average time to process a request
- **Median Response Time**: Middle value (50th percentile)
- **P95 Response Time**: 95% of requests are faster than this
- **P99 Response Time**: 99% of requests are faster than this
- **Min/Max Response Time**: Fastest and slowest requests
- **Standard Deviation**: Consistency of response times

**Why it matters**: Users expect fast responses. High P95/P99 indicates some users get slow responses.

#### B. Throughput Metrics
- **Queries Per Second (QPS)**: How many queries handled per second
- **Queries Per Minute**: How many queries handled per minute
- **Queries Per Hour**: How many queries handled per hour
- **Peak Throughput**: Maximum queries handled in a time period

**Why it matters**: Indicates system capacity and scalability.

#### C. Latency Distribution
- **Histogram**: Distribution of response times across ranges (0-50ms, 50-100ms, etc.)
- **Percentile Breakdown**: P25, P50, P75, P90, P95, P99

**Why it matters**: Shows if most requests are fast or if there's high variance.

#### D. Component Breakdown
- **BERT Encoding Time**: Time spent generating BERT embeddings
- **TF-IDF Calculation Time**: Time spent on TF-IDF similarity
- **Database Operations Time**: Time spent querying database
- **Ranking Time**: Time spent sorting and ranking results
- **Response Formatting Time**: Time spent preparing API response

**Why it matters**: Identifies bottlenecks in the system.

#### E. Slow Query Identification
- **Threshold-based Detection**: Queries slower than X milliseconds
- **Top N Slowest Queries**: List of slowest requests
- **Slow Query Patterns**: Common characteristics of slow queries

**Why it matters**: Helps optimize problematic queries.

### Visualizations Generated
1. **Response Time Stats Bar Chart** - Compare mean, median, P95, P99, max
2. **Response Time Distribution Histogram** - See distribution across ranges
3. **Throughput Bar Chart** - Queries per second/minute/hour
4. **Component Breakdown Pie Chart** - Time spent in each component
5. **Slow Queries Bar Chart** - Identify slowest requests

---

## 🎯 2. ACCURACY METRICS

### What They Measure
How good your recommendations are and how well they match user needs.

### Key Metrics

#### A. Similarity Score Distribution
- **Mean Similarity**: Average relevance score
- **Median Similarity**: Middle relevance score
- **Score Distribution**: How scores spread across ranges (0-0.1, 0.1-0.2, etc.)
- **High Quality Count**: Recommendations with score > 0.7
- **Medium Quality Count**: Recommendations with score 0.4-0.7
- **Low Quality Count**: Recommendations with score < 0.4

**Why it matters**: Higher scores = more relevant recommendations.

#### B. Ranking Quality
- **Top-1 Average Score**: Average similarity of #1 ranked result
- **Top-3 Average Score**: Average similarity of top 3 results
- **Top-5 Average Score**: Average similarity of top 5 results
- **Score Drop Between Ranks**: How much scores decrease from rank to rank
- **Top-1 to Top-10 Gap**: Difference between best and 10th result
- **Ranking Consistency**: How consistent the score drops are

**Why it matters**: Shows if best results are truly better than others.

#### C. Recommendation Diversity
- **Unique Journals Recommended**: How many different journals appear
- **Unique Publishers**: Number of different publishers
- **Subject Diversity**: Number of different subject areas
- **Open Access Ratio**: Percentage of OA vs traditional journals
- **Publisher Distribution**: Spread across publishers (avoid monopoly)

**Why it matters**: Diverse recommendations give users more options.

#### D. Component Contribution Analysis
- **BERT Similarity Contribution**: How much BERT impacts final score
- **TF-IDF Contribution**: How much TF-IDF impacts final score
- **Title Match Contribution**: Impact of title similarity
- **Keyword Match Contribution**: Impact of keyword overlap
- **Impact Factor Boost**: Effect of journal impact factor
- **Field Matching Boost**: Effect of subject area matching

**Why it matters**: Understand which components drive recommendations.

### Advanced Metrics (Require Additional Data)

#### E. Precision@K
- **Precision@5**: % of top 5 results that are relevant
- **Precision@10**: % of top 10 results that are relevant

*Requires: User feedback or expert labels on relevance*

#### F. NDCG (Normalized Discounted Cumulative Gain)
- Measures ranking quality with position-based discounting
- Higher NDCG = better ranking

*Requires: Relevance scores for each result*

#### G. Mean Reciprocal Rank (MRR)
- Average of 1/(rank of first relevant result)
- Higher MRR = relevant results appear earlier

*Requires: Click or selection data*

### Visualizations Generated
1. **Similarity Distribution Histogram** - See score distribution with statistics
2. **Quality Breakdown Pie Chart** - High/medium/low quality split
3. **Ranking Quality Bar Chart** - Compare top-1, top-3, top-5 scores
4. **Diversity Dashboard** - Multi-panel view of diversity metrics

---

## 🖥️ 3. SYSTEM METRICS

### What They Measure
Overall health, data quality, and system status.

### Key Metrics

#### A. Database Statistics
- **Total Journals**: Number of journals in database
- **Journals with Profiles**: Journals that have ML vectors
- **Open Access Journals**: Number of OA journals
- **Journals with Impact Factors**: Coverage of impact factor data
- **Journals with ISSN**: Coverage of ISSN data
- **Total Works**: Research papers indexed
- **Works with Abstracts**: Papers with abstract text
- **Total Queries**: Historical queries processed
- **Total Recommendations**: Historical recommendations made

**Why it matters**: More data = better recommendations.

#### B. Vector Quality Metrics
- **TF-IDF Dimensions**: Size of TF-IDF vectors
- **BERT Dimensions**: Size of BERT vectors (384 for MiniLM)
- **TF-IDF Sparsity**: % of zeros in TF-IDF vectors
- **TF-IDF Non-Zero Features**: Average non-zero features per journal
- **Vector Norms**: Average magnitude of vectors
- **Anomaly Detection**: Zero norms, unexpected dimensions

**Why it matters**: Quality vectors = accurate similarity calculations.

#### C. Data Coverage Analysis
- **Profile Coverage**: % of journals with ML profiles
- **Impact Factor Coverage**: % with impact factor data
- **Subject Coverage**: % with subject classifications
- **Vector Coverage**: % with both TF-IDF and BERT vectors
- **Overall Quality Score**: Weighted average of all coverage

**Why it matters**: Incomplete data reduces recommendation quality.

#### D. System Health Indicators
- **Health Score**: 0-100 overall system health
- **Status**: HEALTHY / WARNING / CRITICAL
- **Health Checks**: Database accessible, journals present, vectors present, recent activity, data quality
- **Issues**: Critical problems requiring attention
- **Warnings**: Non-critical concerns

**Why it matters**: Early warning system for problems.

#### E. Publisher Statistics
- **Total Publishers**: Number of unique publishers
- **Top Publishers**: Most represented publishers
- **Average Journals per Publisher**: Distribution metric
- **Single-Journal Publishers**: Publishers with only one journal

**Why it matters**: Understand data sources and potential biases.

### Visualizations Generated
1. **Database Stats Dashboard** - Multi-panel overview
2. **Vector Quality Charts** - TF-IDF and BERT metrics
3. **System Health Gauge** - Visual health score indicator
4. **Coverage Percentages** - Horizontal bar chart

---

## 👥 4. USER BEHAVIOR METRICS

### What They Measure
How users interact with the system and what they search for.

### Key Metrics

#### A. Query Patterns
- **Total Queries**: Number of queries in time window
- **Hourly Distribution**: Queries by hour of day
- **Daily Distribution**: Queries by day
- **Peak Hour**: Busiest hour
- **Query Length Statistics**: Avg, median, min, max query length
- **Queries Per Day**: Average daily query rate

**Why it matters**: Understand usage patterns and peak times.

#### B. Popular Journals
- **Top Recommended Journals**: Most frequently recommended
- **Recommendation Count**: How many times each appears
- **Average Similarity Score**: Quality of recommendations
- **Publisher Distribution**: Which publishers dominate
- **Open Access Ratio**: OA vs traditional in recommendations

**Why it matters**: Shows which journals the system favors.

#### C. Topic Trends
- **Top Keywords**: Most common words in queries
- **Top Subject Areas**: Most queried research fields
- **Keyword Frequency**: How often each keyword appears
- **Emerging Topics**: New or trending keywords
- **Subject Distribution**: Spread across disciplines

**Why it matters**: Understand research trends and user interests.

#### D. User Interaction Patterns
- **Total Sessions**: Unique user sessions
- **Queries Per Session**: Average queries per session
- **Single-Query Sessions**: % of one-and-done users
- **Multi-Query Sessions**: % of engaged users
- **Session Duration**: Time spent per session
- **Recommendations Per Query**: Average results shown

**Why it matters**: Measure user engagement and satisfaction.

#### E. Open Access Preference
- **OA Recommendation Rate**: % of OA journals recommended
- **OA Availability Rate**: % of OA journals in database
- **OA vs Traditional Ratio**: Comparison
- **User Selection Pattern**: If OA journals are preferred

**Why it matters**: Understand OA trends and preferences.

### Advanced Metrics (Require Additional Instrumentation)

#### F. Recommendation Acceptance Rate
- **Click-Through Rate (CTR)**: % of recommendations clicked
- **Position-Based CTR**: CTR by ranking position
- **Time to First Click**: How quickly users find relevant result
- **Multiple Selections**: Average journals selected per query

*Requires: Click tracking in UI*

#### G. Query Refinement Patterns
- **Refinement Rate**: % of queries that are refined
- **Refinement Strategies**: How users modify queries
- **Query Similarity**: Similarity between consecutive queries
- **Success After Refinement**: If refinement improves results

*Requires: Session-based query analysis*

### Visualizations Generated
1. **Query Patterns Timeline** - Hourly and daily distributions
2. **Popular Journals Bar Chart** - Top recommended journals
3. **Topic Trends Word Cloud** - Keyword visualization
4. **Interaction Dashboard** - Session statistics
5. **Open Access Comparison** - OA vs traditional analysis

---

## 🎨 Visualization Examples

### Performance Visualizations
- 📊 **Bar Charts**: Response time stats, throughput metrics
- 📈 **Histograms**: Latency distribution
- 🥧 **Pie Charts**: Component time breakdown
- 📉 **Line Graphs**: Throughput over time

### Accuracy Visualizations
- 📊 **Histograms**: Similarity score distribution
- 🥧 **Pie Charts**: Quality breakdown (high/medium/low)
- 📊 **Bar Charts**: Top-K ranking comparison
- 📋 **Dashboards**: Multi-metric diversity view

### System Visualizations
- 📊 **Multi-Panel Dashboards**: Database statistics
- 📈 **Bar Charts**: Vector quality metrics
- 🎯 **Gauge Charts**: System health score
- 📊 **Horizontal Bars**: Coverage percentages

### User Visualizations
- 📈 **Time Series**: Query patterns over time
- 📊 **Horizontal Bars**: Popular journals ranking
- ☁️ **Word Clouds**: Topic trends (keywords)
- 🥧 **Pie Charts**: Session engagement
- 📊 **Comparison Charts**: OA vs traditional

---

## 🔍 How to Interpret Metrics

### Good Performance Indicators
- ✅ P95 response time < 500ms
- ✅ P99 response time < 1000ms
- ✅ Throughput > 10 queries/minute
- ✅ Low standard deviation in response times
- ✅ No slow queries > 2000ms

### Good Accuracy Indicators
- ✅ Mean similarity score > 0.6
- ✅ Top-1 average score > 0.7
- ✅ >50% high-quality recommendations (score > 0.7)
- ✅ Consistent score drops between ranks
- ✅ High diversity (many unique journals)

### Good System Health Indicators
- ✅ Health score > 80
- ✅ Vector coverage > 90%
- ✅ No critical issues
- ✅ Data quality score > 70
- ✅ No vector anomalies

### Good User Behavior Indicators
- ✅ Growing query volume
- ✅ High multi-query session rate (>30%)
- ✅ Diverse topic coverage
- ✅ Balanced publisher distribution

---

## 🛠️ Implementation Guide

### Step 1: Basic Metrics Collection

```python
from metrics.metrics_collector import MetricsCollector

collector = MetricsCollector()
metrics = collector.collect_all_metrics(hours=24)
print(metrics)
```

### Step 2: Add to API (Real-time Tracking)

```python
# In app/api/routes.py
from metrics.metrics_collector import MetricsCollector
import time

metrics = MetricsCollector()

@router.post("/recommend")
def get_recommendations(request):
    start = time.time()
    
    # Your logic here
    results = rank_journals(request.abstract)
    
    # Record metrics
    duration_ms = (time.time() - start) * 1000
    metrics.record_request(duration_ms, "recommend")
    
    return results
```

### Step 3: Generate Dashboard

```python
from metrics.visualizations.dashboard_generator import DashboardGenerator

generator = DashboardGenerator()
generator.generate_full_dashboard('dashboard.html', hours=24)
```

---

## 📈 Monitoring Strategy

### Daily Monitoring
- Check system health score
- Review P95 response times
- Monitor query volume
- Check for critical issues

### Weekly Analysis
- Analyze accuracy trends
- Review popular journals
- Check topic trends
- Evaluate diversity metrics

### Monthly Review
- Compare month-over-month performance
- Analyze long-term trends
- Identify optimization opportunities
- Plan improvements

---

## 🚨 Alert Thresholds (Recommended)

### Critical Alerts
- Health score < 50
- P99 response time > 5000ms
- Vector coverage < 50%
- No queries in 24 hours
- Database connection failure

### Warning Alerts
- Health score < 70
- P95 response time > 1000ms
- Vector coverage < 80%
- Data quality score < 60
- Anomalies detected in vectors

---

## 📚 Additional Resources

- **Full Documentation**: `metrics/README.md`
- **Example Scripts**: `metrics/example_*.py`
- **API Integration**: See routes.py for tracking examples
- **Custom Metrics**: Extend existing classes in metrics/

---

## 🎓 Summary

You now have a comprehensive metrics system that tracks:

✅ **Performance**: Speed, throughput, latency  
✅ **Accuracy**: Similarity scores, ranking quality, diversity  
✅ **System Health**: Database stats, vector quality, coverage  
✅ **User Behavior**: Query patterns, popular topics, engagement  

All organized in `metrics/` folder with `visualizations/` subfolder for charts.

**Next Steps**:
1. Run `python metrics/example_generate_dashboard.py`
2. Open `metrics/output/dashboard.html` in browser
3. Review all metrics and visualizations
4. Integrate real-time tracking in your API
5. Set up regular monitoring schedule

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintainer**: Journal Recommendation System Team
