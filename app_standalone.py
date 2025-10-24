#!/usr/bin/env python3
"""
Streamlit Standalone App for Journal Recommender
This version works without requiring a separate API server for deployment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import your services directly
from app.services.recommender import rank_journals, get_ranking_comparisons, analyze_text_distribution
from app.models.base import SessionLocal
from app.models.entities import Journal

# Configure page
st.set_page_config(
    page_title="Journal Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #2E86AB;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def get_database_stats():
    """Get basic database statistics"""
    try:
        db = SessionLocal()
        journal_count = db.query(Journal).count()
        open_access_count = db.query(Journal).filter(Journal.is_open_access == True).count()
        with_impact_factor = db.query(Journal).filter(Journal.impact_factor.isnot(None)).count()
        db.close()
        
        return {
            "total_journals": journal_count,
            "open_access": open_access_count,
            "with_impact_factor": with_impact_factor
        }
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return {"total_journals": 0, "open_access": 0, "with_impact_factor": 0}

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Journal Recommender System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Database Stats
        with st.expander("📊 Database Statistics", expanded=True):
            stats = get_database_stats()
            st.metric("Total Journals", stats["total_journals"])
            st.metric("Open Access", stats["open_access"])
            st.metric("With Impact Factor", stats["with_impact_factor"])
        
        # Recommendation Settings
        st.subheader("⚙️ Settings")
        top_k = st.slider("Number of recommendations", 1, 20, 5)
        method = st.selectbox(
            "Recommendation Method",
            ["combined", "tfidf", "bert"],
            help="Combined uses both TF-IDF and BERT embeddings"
        )
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔍 Recommendations", "📈 Advanced Analysis"])
    
    with tab1:
        recommendations_page(top_k, method)
    
    with tab2:
        advanced_analysis_page(top_k)

def recommendations_page(top_k, method):
    st.header("Get Journal Recommendations")
    
    # Abstract input
    abstract = st.text_area(
        "📝 Enter your research abstract:",
        height=150,
        placeholder="Enter your research abstract here to get personalized journal recommendations..."
    )
    
    if st.button("🚀 Get Recommendations", type="primary"):
        if abstract.strip():
            with st.spinner("🔍 Finding the best journals for your research..."):
                try:
                    # Get recommendations directly from service
                    results = rank_journals(abstract, top_k)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} recommendations!")
                        
                        # Display recommendations
                        for i, result in enumerate(results, 1):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="recommendation-card">
                                        <h4>#{i}. {result['journal_name']}</h4>
                                        <p><strong>Publisher:</strong> {result.get('publisher', 'Unknown')}</p>
                                        <p><strong>Open Access:</strong> {'Yes' if result.get('is_open_access') else 'No'}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.metric("Combined Score", f"{result['similarity_combined']:.3f}")
                                    st.metric("TF-IDF Score", f"{result['similarity_tfidf']:.3f}")
                                
                                with col3:
                                    st.metric("BERT Score", f"{result['similarity_bert']:.3f}")
                                    if result.get('impact_factor'):
                                        st.metric("Impact Factor", f"{result['impact_factor']:.2f}")
                    else:
                        st.warning("No recommendations found. Please try a different abstract.")
                        
                except Exception as e:
                    st.error(f"Error getting recommendations: {e}")
        else:
            st.warning("Please enter an abstract to get recommendations.")

def advanced_analysis_page(top_k):
    st.header("🔬 Advanced Analysis")
    
    abstract = st.text_area(
        "📝 Enter your research abstract for detailed analysis:",
        height=120,
        key="advanced_abstract"
    )
    
    if st.button("🔍 Analyze", type="primary", key="analyze_btn"):
        if abstract.strip():
            try:
                with st.spinner("🔄 Performing advanced analysis..."):
                    # Get comparison data
                    comparison_data = get_ranking_comparisons(abstract, top_k)
                    
                    # Create tabs for different analyses
                    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                        "📊 Similarity Breakdown", 
                        "🏆 Ranking Comparisons", 
                        "📈 Visualizations"
                    ])
                    
                    with analysis_tab1:
                        similarity_breakdown(comparison_data)
                    
                    with analysis_tab2:
                        ranking_comparisons(comparison_data)
                    
                    with analysis_tab3:
                        create_visualizations(comparison_data, abstract)
                        
            except Exception as e:
                st.error(f"Analysis error: {e}")
        else:
            st.warning("Please enter an abstract for analysis.")

def similarity_breakdown(comparison_data):
    st.subheader("🎯 Similarity Score Breakdown")
    
    # Combined results
    combined_results = comparison_data.get("similarity_ranking", [])
    if combined_results:
        df_combined = pd.DataFrame(combined_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # TF-IDF scores
            fig_tfidf = px.bar(
                df_combined, 
                x="similarity_tfidf", 
                y="journal_name",
                orientation="h",
                title="TF-IDF Similarity Scores",
                labels={"similarity_tfidf": "TF-IDF Score", "journal_name": "Journal"}
            )
            fig_tfidf.update_layout(height=400)
            st.plotly_chart(fig_tfidf, use_container_width=True)
        
        with col2:
            # BERT scores
            fig_bert = px.bar(
                df_combined, 
                x="similarity_bert", 
                y="journal_name",
                orientation="h",
                title="BERT Similarity Scores",
                labels={"similarity_bert": "BERT Score", "journal_name": "Journal"}
            )
            fig_bert.update_layout(height=400)
            st.plotly_chart(fig_bert, use_container_width=True)

def ranking_comparisons(comparison_data):
    st.subheader("🏆 Method Comparison")
    
    # Compare different ranking methods
    methods = ["similarity_ranking", "tfidf_only_ranking", "bert_only_ranking"]
    method_names = ["Combined", "TF-IDF Only", "BERT Only"]
    
    comparison_df = []
    for method, name in zip(methods, method_names):
        results = comparison_data.get(method, [])
        for i, result in enumerate(results[:5], 1):
            comparison_df.append({
                "Method": name,
                "Rank": i,
                "Journal": result["journal_name"],
                "Score": result.get("similarity_combined", result.get("similarity_tfidf", result.get("similarity_bert", 0)))
            })
    
    if comparison_df:
        df = pd.DataFrame(comparison_df)
        
        # Pivot for heatmap
        pivot_df = df.pivot(index="Journal", columns="Method", values="Rank")
        
        fig_heatmap = px.imshow(
            pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            title="Journal Ranking Comparison Across Methods",
            labels=dict(color="Rank"),
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

def create_visualizations(comparison_data, abstract):
    st.subheader("📈 Interactive Visualizations")
    
    # Scatter plot: TF-IDF vs BERT
    combined_results = comparison_data.get("similarity_ranking", [])
    if combined_results:
        df = pd.DataFrame(combined_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                df, 
                x="similarity_tfidf", 
                y="similarity_bert",
                hover_data=["journal_name"],
                title="TF-IDF vs BERT Similarity",
                labels={
                    "similarity_tfidf": "TF-IDF Score", 
                    "similarity_bert": "BERT Score"
                }
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Text analysis
            try:
                text_analysis = analyze_text_distribution(abstract)
                if text_analysis:
                    # Word frequency
                    words = list(text_analysis.keys())[:10]
                    frequencies = list(text_analysis.values())[:10]
                    
                    fig_words = px.bar(
                        x=frequencies,
                        y=words,
                        orientation='h',
                        title="Top 10 Words in Abstract",
                        labels={"x": "Frequency", "y": "Words"}
                    )
                    fig_words.update_layout(height=400)
                    st.plotly_chart(fig_words, use_container_width=True)
            except:
                st.info("Word frequency analysis not available")

if __name__ == "__main__":
    main()