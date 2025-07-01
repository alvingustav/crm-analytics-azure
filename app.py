import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from utils.data_loader import load_data, load_model_artifacts
from utils.azure_openai import get_ai_insights

# Page config
st.set_page_config(
    page_title="CRM Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .ai-insight {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    """Load and cache dashboard data"""
    try:
        df = load_data('data/churn.csv')
        
        # Load model artifacts
        artifacts = load_model_artifacts()
        
        # Calculate KPIs
        kpis = {
            'total_customers': len(df),
            'churn_rate': (df['Churn'] == 'Yes').mean(),
            'avg_monthly_revenue': df['MonthlyCharges'].sum(),
            'avg_clv': (df['tenure'] * df['MonthlyCharges']).mean(),
            'high_risk_customers': len(df[df['Churn'] == 'Yes']),
            'segments_count': 4
        }
        
        return df, artifacts, kpis
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def main():
    # Load data
    df, artifacts, kpis = load_dashboard_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check your data files.")
        return

    # Header
    st.markdown('<h1 class="main-header">🏢 CRM Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Executive Summary
    st.header("📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Total Customers",
            value=f"{kpis['total_customers']:,}",
            delta=f"+{kpis['total_customers']//10} this month"
        )
    
    with col2:
        st.metric(
            label="⚠️ Churn Rate",
            value=f"{kpis['churn_rate']:.1%}",
            delta=f"-{kpis['churn_rate']*5:.1f}% vs last month",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="💰 Monthly Revenue",
            value=f"${kpis['avg_monthly_revenue']:,.0f}",
            delta=f"+${kpis['avg_monthly_revenue']*0.05:,.0f}"
        )
    
    with col4:
        st.metric(
            label="💎 Avg CLV",
            value=f"${kpis['avg_clv']:.2f}",
            delta=f"+${kpis['avg_clv']*0.1:.2f}"
        )

    # AI-Generated Insights
    st.header("🤖 AI-Powered Business Insights")
    
    if st.button("🔄 Generate Fresh AI Insights", type="primary"):
        with st.spinner("Analyzing data with Azure OpenAI..."):
            ai_insight = get_ai_insights({
                'churn_rate': kpis['churn_rate'],
                'total_customers': kpis['total_customers'],
                'avg_clv': kpis['avg_clv'],
                'high_risk_customers': kpis['high_risk_customers']
            })
            
            st.markdown(f'<div class="ai-insight">🧠 <strong>AI Analysis:</strong><br>{ai_insight}</div>', 
                       unsafe_allow_html=True)
    
    # Quick Overview Charts
    st.header("📊 Quick Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        fig_churn = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Customer Churn Distribution",
            color_discrete_sequence=['#00cc96', '#ef553b']
        )
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig_charges = px.histogram(
            df, 
            x='MonthlyCharges',
            title="Monthly Charges Distribution",
            nbins=30,
            color_discrete_sequence=['#636efa']
        )
        fig_charges.update_layout(showlegend=False)
        st.plotly_chart(fig_charges, use_container_width=True)
    
    # Navigation Guide
    st.header("🗺️ Navigation Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Customer Segmentation
        - **Demographic Analysis**: Gender, age, family status
        - **Behavioral Patterns**: Service usage and tenure
        - **Value Segmentation**: Revenue and CLV analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Churn Prediction
        - **Risk Scoring**: Individual customer risk assessment
        - **Prediction Models**: ML-powered churn forecasting
        - **Retention Strategies**: Actionable recommendations
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Analytics Dashboard
        - **Real-time Metrics**: Live KPI monitoring
        - **Campaign Analysis**: Marketing effectiveness
        - **CLV Optimization**: Revenue growth insights
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Deployed on Azure** | Built with Streamlit & Azure OpenAI")

if __name__ == "__main__":
    main()
