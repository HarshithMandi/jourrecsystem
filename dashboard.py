#!/usr/bin/env python3
"""
Streamlit Dashboard for Journal Recommender API
A comprehensive web interface for journal recommendations.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# Configure page
st.set_page_config(
    page_title="Journal Recommender Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_recommendations(abstract, top_k=10):
    """Get journal recommendations from the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/recommend",
            json={"abstract": abstract, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            # Try to get detailed error message from response
            try:
                error_detail = response.json()
                if 'detail' in error_detail:
                    # Extract validation errors
                    if isinstance(error_detail['detail'], list) and len(error_detail['detail']) > 0:
                        error_msg = error_detail['detail'][0].get('msg', 'Validation error')
                        return {"error": error_msg}
                    else:
                        return {"error": str(error_detail['detail'])}
                else:
                    return {"error": f"API Error: {response.status_code}"}
            except:
                return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_batch_recommendations(abstracts, top_k=5):
    """Get batch recommendations from the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/batch-recommend",
            json={"abstracts": abstracts, "top_k": top_k},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_database_stats():
    """Get database statistics from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main dashboard application."""
    
    # Header
    st.title("📚 Journal Recommender Dashboard")
    st.markdown("*Find the perfect journal for your research using AI-powered recommendations*")
    
    # Check API status
    api_status = check_api_status()
    
    if not api_status:
        st.error("🚨 **API Server Not Running**")
        st.markdown("Please start the API server first:")
        st.code("uvicorn app.main:app --reload --port 8000", language="bash")
        st.markdown("Then refresh this page.")
        return
    
    st.success("**API Server Connected**")
    
    # Sidebar
    st.sidebar.title("Dashboard Navigation")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Single Recommendation", "Batch Analysis", "Database Statistics", "About"]
    )
    
    # Home Page
    if page == "Home":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Welcome to Journal Recommender")
            st.markdown("""
            This dashboard provides an intuitive interface for finding the most relevant 
            academic journals for your research. Our system uses advanced machine learning 
            techniques combining TF-IDF and BERT embeddings for accurate recommendations.
            
            ### ✨ Features:
            - **Single Recommendations**: Get journal suggestions for one abstract
            - **Batch Analysis**: Process multiple abstracts at once
            - **Smart Weighting**: 30% TF-IDF + 70% BERT for optimal semantic matching
            - **Real-time Statistics**: Database insights and performance metrics
            """)
            
            # Quick stats
            stats = get_database_stats()
            if "error" not in stats:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Journals", stats.get("total_journals", 0))
                with col_b:
                    st.metric("Total Queries", stats.get("total_queries", 0))
                with col_c:
                    st.metric("Recommendations Made", stats.get("total_recommendations", 0))
        
        with col2:
            st.image("https://via.placeholder.com/300x200/4CAF50/white?text=Journal+Recommender", 
                    caption="AI-Powered Journal Matching")
            
            # Quick start
            st.subheader("Quick Start")
            sample_abstract = st.text_area(
                "Try a sample abstract:",
                "Machine learning algorithms for protein structure prediction and drug discovery applications.",
                height=100
            )
            
            if st.button("Get Quick Recommendations", type="primary"):
                with st.spinner("Finding journals..."):
                    result = get_recommendations(sample_abstract, 3)
                    if "error" not in result:
                        st.success("Found recommendations!")
                        for i, rec in enumerate(result["recommendations"], 1):
                            st.write(f"**{i}. {rec['journal_name']}** (Score: {rec['similarity_score']:.3f})")
                    else:
                        st.error(f"Error: {result['error']}")
    
    # Single Recommendation Page
    elif page == "Single Recommendation":
        st.header("Single Abstract Recommendation")
        st.markdown("Get personalized journal recommendations for your research abstract.")
        
        # Input section
        with st.container():
            st.subheader("Your Research Abstract")
            abstract = st.text_area(
                "Enter your research abstract (minimum 50 characters):",
                height=150,
                placeholder="Describe your research methodology, findings, and implications..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Number of recommendations:", 1, 20, 10)
            with col2:
                # Check both character length and word count
                word_count = len(abstract.split()) if abstract else 0
                is_valid = len(abstract) >= 50 and word_count >= 10
                if st.button("Get Recommendations", type="primary", disabled=not is_valid):
                    # Get recommendations
                    with st.spinner("Analyzing your abstract and finding the best journals..."):
                        start_time = time.time()
                        result = get_recommendations(abstract, top_k)
                        processing_time = time.time() - start_time
                    
                    if "error" not in result:
                        st.success(f"Found {len(result['recommendations'])} recommendations in {processing_time:.2f}s")
                        
                        # Display results
                        st.subheader("Recommended Journals")
                        
                        # Create DataFrame for better display
                        recs_data = []
                        for i, rec in enumerate(result["recommendations"], 1):
                            recs_data.append({
                                "Rank": i,
                                "Journal Name": rec["journal_name"],
                                "Similarity Score": f"{rec['similarity_score']:.3f}",
                                "Match Percentage": f"{rec['similarity_score'] * 100:.1f}%"
                            })
                        
                        df = pd.DataFrame(recs_data)
                        
                        # Interactive table
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Visualization
                        fig = px.bar(
                            df,
                            x="Journal Name",
                            y=[float(x.replace('%', '')) for x in df["Match Percentage"]],
                            title="Journal Recommendation Scores",
                            labels={"y": "Match Percentage (%)"},
                            color=[float(x.replace('%', '')) for x in df["Match Percentage"]],
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Export options
                        st.subheader("📤 Export Results")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"journal_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Additional metrics
                        with st.expander("Detailed Metrics"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f} ms")
                            with col2:
                                st.metric("Total Journals", result.get("total_journals", 0))
                            with col3:
                                st.metric("Average Score", f"{sum(rec['similarity_score'] for rec in result['recommendations']) / len(result['recommendations']):.3f}")
                            with col4:
                                st.metric("Query ID", result.get("query_id", "N/A"))
                    
                    else:
                        st.error(f"Error: {result['error']}")
            
            # Validation feedback
            if len(abstract) > 0:
                word_count = len(abstract.split())
                if len(abstract) < 50:
                    st.warning(f"Abstract must be at least 50 characters long. Current: {len(abstract)} characters.")
                elif word_count < 10:
                    st.warning(f"Abstract must contain at least 10 words. Current: {word_count} words.")
                else:
                    st.success("Abstract meets all requirements. Ready to get recommendations!")
    
    # Batch Analysis Page
    elif page == "Batch Analysis":
        st.header("Batch Abstract Analysis")
        st.markdown("Process multiple research abstracts simultaneously for comparative analysis.")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "File Upload"]
        )
        
        abstracts = []
        
        if input_method == "Manual Entry":
            st.subheader("Enter Multiple Abstracts")
            
            # Dynamic abstract inputs
            if "num_abstracts" not in st.session_state:
                st.session_state.num_abstracts = 2
            
            col1, col2 = st.columns([1, 4])
            with col1:
                num_abstracts = st.number_input("Number of abstracts:", 1, 10, st.session_state.num_abstracts)
                st.session_state.num_abstracts = num_abstracts
            
            for i in range(num_abstracts):
                abstract = st.text_area(
                    f"Abstract {i+1}:",
                    height=100,
                    key=f"abstract_{i}"
                )
                if abstract.strip():
                    abstracts.append(abstract.strip())
        
        else:  # File Upload
            st.subheader("Upload Abstracts File")
            uploaded_file = st.file_uploader(
                "Choose a file (TXT, CSV, or JSON):",
                type=["txt", "csv", "json"]
            )
            
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1].lower()
                content = uploaded_file.read().decode('utf-8')
                
                if file_type == "txt":
                    abstracts = [line.strip() for line in content.split('\n') if line.strip()]
                elif file_type == "csv":
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'abstract' in df.columns:
                            abstracts = df['abstract'].dropna().tolist()
                        else:
                            st.error("CSV must have an 'abstract' column")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
                elif file_type == "json":
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            abstracts = [str(item) for item in data if str(item).strip()]
                        else:
                            st.error("JSON must contain a list of abstracts")
                    except Exception as e:
                        st.error(f"Error reading JSON: {e}")
        
        # Process batch
        if abstracts:
            st.success(f"Found {len(abstracts)} valid abstracts")
            
            col1, col2 = st.columns(2)
            with col1:
                batch_top_k = st.slider("Recommendations per abstract:", 1, 10, 5)
            with col2:
                if st.button("Process Batch", type="primary"):
                    with st.spinner(f"Processing {len(abstracts)} abstracts..."):
                        start_time = time.time()
                        batch_result = get_batch_recommendations(abstracts, batch_top_k)
                        processing_time = time.time() - start_time
                    
                    if "error" not in batch_result:
                        st.success(f"Processed {len(batch_result['results'])} abstracts in {processing_time:.2f}s")
                        
                        # Results analysis
                        st.subheader("Batch Analysis Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Abstracts", len(batch_result["results"]))
                        with col2:
                            st.metric("Total Processing Time", f"{batch_result.get('total_processing_time_ms', 0):.0f} ms")
                        with col3:
                            avg_time = batch_result.get('total_processing_time_ms', 0) / len(batch_result["results"])
                            st.metric("Avg Time per Abstract", f"{avg_time:.0f} ms")
                        with col4:
                            total_recs = sum(len(r["recommendations"]) for r in batch_result["results"])
                            st.metric("Total Recommendations", total_recs)
                        
                        # Detailed results
                        for i, result in enumerate(batch_result["results"]):
                            with st.expander(f"📄 Abstract {i+1} Results ({len(result['recommendations'])} recommendations)"):
                                st.text(f"Abstract: {abstracts[i][:200]}...")
                                
                                # Create DataFrame for this result
                                result_data = []
                                for j, rec in enumerate(result["recommendations"], 1):
                                    result_data.append({
                                        "Rank": j,
                                        "Journal": rec["journal_name"],
                                        "Score": f"{rec['similarity_score']:.3f}"
                                    })
                                
                                if result_data:
                                    df_result = pd.DataFrame(result_data)
                                    st.dataframe(df_result, use_container_width=True, hide_index=True)
                        
                        # Export batch results
                        st.subheader("📤 Export Batch Results")
                        
                        # Prepare comprehensive export data
                        export_data = []
                        for i, result in enumerate(batch_result["results"]):
                            for j, rec in enumerate(result["recommendations"], 1):
                                export_data.append({
                                    "Abstract_ID": i + 1,
                                    "Abstract_Text": abstracts[i],
                                    "Rank": j,
                                    "Journal_Name": rec["journal_name"],
                                    "Similarity_Score": rec["similarity_score"],
                                    "Processing_Time_MS": result.get("processing_time_ms", 0)
                                })
                        
                        export_df = pd.DataFrame(export_data)
                        csv_export = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📄 Download Complete Batch Results (CSV)",
                            data=csv_export,
                            file_name=f"batch_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error(f"Batch processing error: {batch_result['error']}")
        
        else:
            st.info("Please enter or upload abstracts to begin batch analysis.")
    
    # Database Statistics Page
    elif page == "Database Statistics":
        st.header("Database Statistics & Analytics")
        st.markdown("Comprehensive insights into the journal database and system performance.")
        
        # Get statistics
        stats = get_database_stats()
        
        if "error" not in stats:
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Journals",
                    stats.get("total_journals", 0),
                    help="Number of journals in the database"
                )
            with col2:
                st.metric(
                    "Total Queries",
                    stats.get("total_queries", 0),
                    help="Number of recommendation queries processed"
                )
            with col3:
                st.metric(
                    "Recommendations Made",
                    stats.get("total_recommendations", 0),
                    help="Total recommendations generated"
                )
            with col4:
                avg_recs = stats.get("total_recommendations", 0) / max(stats.get("total_queries", 1), 1)
                st.metric(
                    "Avg Recs per Query",
                    f"{avg_recs:.1f}",
                    help="Average recommendations per query"
                )
            
            # Advanced metrics
            if stats.get("journals_with_profiles"):
                st.subheader("🧬 Machine Learning Coverage")
                col1, col2 = st.columns(2)
                
                with col1:
                    coverage = (stats["journals_with_profiles"] / stats["total_journals"]) * 100
                    st.metric(
                        "ML Profile Coverage",
                        f"{coverage:.1f}%",
                        help="Percentage of journals with ML vectors"
                    )
                
                with col2:
                    if stats.get("avg_similarity_score"):
                        st.metric(
                            "Avg Similarity Score",
                            f"{stats['avg_similarity_score']:.3f}",
                            help="Average similarity score across recommendations"
                        )
                
                # Coverage visualization
                fig_coverage = go.Figure(data=[
                    go.Pie(
                        labels=['With ML Profiles', 'Without ML Profiles'],
                        values=[
                            stats["journals_with_profiles"],
                            stats["total_journals"] - stats["journals_with_profiles"]
                        ],
                        hole=0.4,
                        title="Journal ML Profile Coverage"
                    )
                ])
                st.plotly_chart(fig_coverage, use_container_width=True)
            
            # System health
            st.subheader("🏥 System Health")
            
            # API performance test
            if st.button("Run Performance Test"):
                test_abstract = "Machine learning for biomedical data analysis and drug discovery applications."
                
                with st.spinner("Testing API performance..."):
                    start_time = time.time()
                    test_result = get_recommendations(test_abstract, 5)
                    response_time = (time.time() - start_time) * 1000
                
                if "error" not in test_result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{response_time:.0f} ms")
                    with col2:
                        st.metric("API Processing", f"{test_result.get('processing_time_ms', 0):.0f} ms")
                    with col3:
                        status = "🟢 Excellent" if response_time < 1000 else "🟡 Good" if response_time < 3000 else "🔴 Slow"
                        st.metric("Performance", status)
                else:
                    st.error(f"Performance test failed: {test_result['error']}")
            
            # Refresh stats
            if st.button("Refresh Statistics"):
                st.rerun()
        
        else:
            st.error(f"Could not fetch statistics: {stats['error']}")
    
    # About Page
    elif page == "About":
        st.header("About Journal Recommender")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Purpose
            The Journal Recommender System helps researchers find the most suitable academic 
            journals for their research papers using advanced machine learning techniques.
            
            ## Technology Stack
            
            ### Machine Learning
            - **TF-IDF Vectorization**: Term frequency analysis for keyword matching
            - **BERT Embeddings**: Semantic understanding using transformer models
            - **Hybrid Approach**: 30% TF-IDF + 70% BERT for optimal results
            
            ### Backend
            - **FastAPI**: High-performance API framework
            - **SQLite**: Lightweight database for journal storage
            - **scikit-learn**: Machine learning utilities
            - **sentence-transformers**: BERT model implementation
            
            ### Frontend
            - **Streamlit**: Interactive web dashboard
            - **Plotly**: Dynamic visualizations
            - **Pandas**: Data manipulation and analysis
            
            ## Features
            
            ### Single Recommendations
            - Real-time journal suggestions
            - Similarity score analysis
            - Interactive visualizations
            - CSV export functionality
            
            ### Batch Processing
            - Multiple abstract analysis
            - File upload support (TXT, CSV, JSON)
            - Comparative results
            - Comprehensive reporting
            
            ### Analytics Dashboard
            - Database statistics
            - Performance monitoring
            - ML model coverage
            - System health checks
            
            ## Algorithm Details
            
            The recommendation system uses a two-stage approach:
            
            1. **Feature Extraction**:
               - TF-IDF vectors capture keyword importance
               - BERT embeddings understand semantic meaning
            
            2. **Similarity Calculation**:
               - Cosine similarity for both TF-IDF and BERT
               - Weighted combination: 30% TF-IDF + 70% BERT
               - Ranking by final similarity scores
            
            ## Getting Started
            
            1. **Start the API Server**:
               ```bash
               uvicorn app.main:app --reload --port 8000
               ```
            
            2. **Launch the Dashboard**:
               ```bash
               streamlit run dashboard.py
               ```
            
            3. **Begin Exploring**:
               - Try the single recommendation feature
               - Upload multiple abstracts for batch analysis
               - Monitor system performance
            """)
        
        with col2:
            st.markdown("""
            ## Quick Stats
            """)
            
            # Quick stats display
            stats = get_database_stats()
            if "error" not in stats:
                st.metric("Journals", stats.get("total_journals", 0))
                st.metric("Queries", stats.get("total_queries", 0))
                st.metric("Recommendations", stats.get("total_recommendations", 0))
            
            st.markdown("""
            ## Configuration
            
            **API Endpoint**: `http://localhost:8000`
            
            **Model Settings**:
            - TF-IDF: 20K features
            - BERT: all-MiniLM-L6-v2
            - Weighting: 30% / 70%
            
            ## 📞 Support
            
            For issues or questions:
            - Check API server status
            - Review error messages
            - Verify input formats
            
            ## 🔄 Version
            
            Dashboard Version: 1.0.0
            Last Updated: October 2025
            """)

if __name__ == "__main__":
    main()