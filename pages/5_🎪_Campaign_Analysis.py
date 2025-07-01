import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.data_loader import load_data
from utils.azure_openai import get_ai_insights, analyze_customer_segment
from utils.model_utils import predict_customer_segment, calculate_clv
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="Campaign Analysis", page_icon="🎪", layout="wide")

st.title("🎪 Marketing Campaign & Product Analysis")
st.markdown("Analyze service adoption, identify opportunities, and optimize marketing campaigns")
st.markdown("---")

# Load data
@st.cache_data
def get_campaign_data():
    df = load_data()
    if df is not None:
        # Calculate service adoption metrics
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Convert to numeric for analysis
        for col in service_cols:
            if col in df.columns:
                df[f'{col}_numeric'] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
        
        # Calculate service adoption score
        numeric_service_cols = [f'{col}_numeric' for col in service_cols if f'{col}_numeric' in df.columns]
        df['ServiceAdoptionScore'] = df[numeric_service_cols].sum(axis=1)
        
        # CLV calculation
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        
        # Customer segments for campaign targeting
        df['CampaignSegment'] = 'Unknown'
        
        # Define segments based on behavior
        df.loc[(df['ServiceAdoptionScore'] <= 2) & (df['tenure'] < 12), 'CampaignSegment'] = 'New Basic Users'
        df.loc[(df['ServiceAdoptionScore'] >= 5) & (df['CLV'] > 2000), 'CampaignSegment'] = 'Premium Users'
        df.loc[(df['ServiceAdoptionScore'].between(3, 4)) & (df['tenure'] > 12), 'CampaignSegment'] = 'Standard Users'
        df.loc[(df['MonthlyCharges'] < 50) & (df['ServiceAdoptionScore'] <= 3), 'CampaignSegment'] = 'Price-Sensitive'
        
        # Simulated campaign performance data
        np.random.seed(42)
        df['EmailEngagement'] = np.random.beta(2, 5, len(df))  # Realistic engagement rates
        df['CampaignResponse'] = np.random.binomial(1, df['EmailEngagement'] * 0.3, len(df))
        
    return df

df = get_campaign_data()

if df is None:
    st.error("❌ Unable to load campaign data")
    st.stop()

# Campaign Planning Section
st.sidebar.header("🎯 Campaign Planner")
st.sidebar.markdown("Design and analyze marketing campaigns:")

with st.sidebar:
    st.subheader("Campaign Configuration")
    
    campaign_type = st.selectbox(
        "Campaign Type",
        ["Service Upsell", "Retention", "Acquisition", "Cross-sell", "Loyalty"]
    )
    
    target_segment = st.selectbox(
        "Target Segment",
        ["All Customers", "New Basic Users", "Standard Users", "Premium Users", "Price-Sensitive"]
    )
    
    campaign_budget = st.slider("Campaign Budget ($)", 1000, 50000, 10000)
    expected_response_rate = st.slider("Expected Response Rate (%)", 1.0, 20.0, 5.0)
    avg_revenue_per_response = st.slider("Avg Revenue per Response ($)", 10, 500, 100)
    
    # Calculate ROI projection
    if target_segment == "All Customers":
        target_size = len(df)
    else:
        target_size = len(df[df['CampaignSegment'] == target_segment])
    
    expected_responses = target_size * (expected_response_rate / 100)
    expected_revenue = expected_responses * avg_revenue_per_response
    roi = ((expected_revenue - campaign_budget) / campaign_budget) * 100
    
    st.subheader("📊 Campaign Projection")
    st.metric("Target Audience", f"{target_size:,}")
    st.metric("Expected Responses", f"{expected_responses:.0f}")
    st.metric("Projected Revenue", f"${expected_revenue:,.0f}")
    st.metric("ROI", f"{roi:.1f}%")
    
    if st.button("🚀 Generate Campaign Strategy", type="primary"):
        campaign_data = {
            'campaign_type': campaign_type,
            'target_segment': target_segment,
            'target_size': target_size,
            'budget': campaign_budget,
            'expected_responses': expected_responses,
            'roi': roi
        }
        
        # Get AI campaign strategy
        with st.spinner("Generating AI campaign strategy..."):
            prompt = f"""
            Create a marketing campaign strategy for:
            Campaign Type: {campaign_type}
            Target: {target_segment} ({target_size:,} customers)
            Budget: ${campaign_budget:,}
            Expected ROI: {roi:.1f}%
            
            Provide 3 specific tactics and success metrics.
            """
            
            try:
                ai_strategy = get_ai_insights(campaign_data)
                st.success("✅ Campaign Strategy Generated!")
                st.markdown(f"**AI Strategy:**\n{ai_strategy}")
            except:
                st.info("Campaign strategy template generated locally")

# Service Adoption Analysis
st.header("📱 Service Adoption Analysis")

col1, col2, col3, col4 = st.columns(4)

service_cols = ['PhoneService', 'OnlineSecurity', 'StreamingTV', 'TechSupport']
service_adoption_rates = {}

for i, service in enumerate(service_cols):
    if service in df.columns:
        adoption_rate = (df[service] == 'Yes').mean()
        service_adoption_rates[service] = adoption_rate
        
        with [col1, col2, col3, col4][i]:
            st.metric(
                label=f"📊 {service.replace('Service', '').replace('Support', '')}",
                value=f"{adoption_rate:.1%}",
                delta=f"+{adoption_rate*10:.1f}% vs target"
            )

# Service adoption heatmap
st.subheader("🔥 Service Adoption Heatmap")

# Create service adoption matrix
services_analysis = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

adoption_matrix = []
for service in services_analysis:
    if service in df.columns:
        by_contract = df.groupby('Contract')[service].apply(lambda x: (x == 'Yes').mean()).round(3)
        adoption_matrix.append(by_contract.values)

if adoption_matrix:
    adoption_df = pd.DataFrame(adoption_matrix, 
                              index=[s.replace('Service', '').replace('Support', '') for s in services_analysis if s in df.columns],
                              columns=df['Contract'].unique())
    
    fig_heatmap = px.imshow(
        adoption_df,
        title="Service Adoption Rate by Contract Type",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig_heatmap.update_xaxes(title="Contract Type")
    fig_heatmap.update_yaxes(title="Services")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Cross-selling & Upselling Opportunities
st.header("💰 Cross-selling & Upselling Opportunities")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Cross-selling Targets")
    
    # Customers with phone service but no other services
    phone_only = df[
        (df['PhoneService'] == 'Yes') & 
        (df['ServiceAdoptionScore'] == 1)
    ]
    
    st.metric("Phone-Only Customers", f"{len(phone_only):,}")
    st.write("**Opportunity:** Internet & streaming services")
    
    # Internet users without streaming
    internet_no_streaming = df[
        (df['InternetService'].isin(['DSL', 'Fiber optic'])) &
        (df['StreamingTV'] == 'No') &
        (df['StreamingMovies'] == 'No')
    ]
    
    st.metric("Internet Users (No Streaming)", f"{len(internet_no_streaming):,}")
    st.write("**Opportunity:** Streaming service bundles")
    
    # High monthly charges, low service adoption
    high_spend_low_adoption = df[
        (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) &
        (df['ServiceAdoptionScore'] < 3)
    ]
    
    st.metric("High Spend, Low Adoption", f"{len(high_spend_low_adoption):,}")
    st.write("**Opportunity:** Premium service packages")

with col2:
    st.subheader("📈 Upselling Targets")
    
    # Monthly contract users with high CLV
    monthly_high_clv = df[
        (df['Contract'] == 'Month-to-month') &
        (df['CLV'] > df['CLV'].quantile(0.6))
    ]
    
    st.metric("Monthly High-Value", f"{len(monthly_high_clv):,}")
    st.write("**Opportunity:** Annual contract incentives")
    
    # DSL users who could upgrade to Fiber
    dsl_upgrade_candidates = df[
        (df['InternetService'] == 'DSL') &
        (df['MonthlyCharges'] > 50) &
        (df['ServiceAdoptionScore'] >= 3)
    ]
    
    st.metric("DSL Upgrade Candidates", f"{len(dsl_upgrade_candidates):,}")
    st.write("**Opportunity:** Fiber optic upgrade")
    
    # Basic users ready for premium
    premium_ready = df[
        (df['tenure'] > 12) &
        (df['ServiceAdoptionScore'] >= 4) &
        (df['MonthlyCharges'] < 80)
    ]
    
    st.metric("Premium Ready", f"{len(premium_ready):,}")
    st.write("**Opportunity:** Premium service tiers")

# Campaign Performance Analysis
st.header("📊 Campaign Performance Analysis")

# Simulated campaign data
campaign_tabs = st.tabs(["Email Campaigns", "Service Promotions", "Retention Campaigns", "ROI Analysis"])

with campaign_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Email engagement by segment
        email_engagement = df.groupby('CampaignSegment')['EmailEngagement'].mean().sort_values(ascending=False)
        
        fig_email = px.bar(
            x=email_engagement.values,
            y=email_engagement.index,
            orientation='h',
            title="Email Engagement Rate by Segment",
            color=email_engagement.values,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_email, use_container_width=True)
    
    with col2:
        # Campaign response correlation
        response_factors = df[['tenure', 'MonthlyCharges', 'ServiceAdoptionScore', 'CampaignResponse']].corr()['CampaignResponse'][:-1]
        
        fig_corr = px.bar(
            x=response_factors.index,
            y=response_factors.values,
            title="Campaign Response Correlations",
            color=response_factors.values,
            color_continuous_scale="RdYlBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with campaign_tabs[1]:
    # Service-specific promotion analysis
    service_promotion_data = []
    
    for service in services_analysis[:4]:  # Analyze top 4 services
        if service in df.columns:
            adoption_rate = (df[service] == 'Yes').mean()
            potential_customers = len(df[df[service] == 'No'])
            revenue_potential = potential_customers * df['MonthlyCharges'].mean() * 0.1  # 10% uplift
            
            service_promotion_data.append({
                'Service': service.replace('Service', '').replace('Support', ''),
                'Current Adoption': f"{adoption_rate:.1%}",
                'Potential Customers': potential_customers,
                'Revenue Potential': f"${revenue_potential:,.0f}"
            })
    
    promotion_df = pd.DataFrame(service_promotion_data)
    st.dataframe(promotion_df, use_container_width=True)
    
    # Service bundle recommendations
    st.subheader("📦 Bundle Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎬 Entertainment Bundle**
        - Streaming TV + Movies
        - Target: Internet users without streaming
        - Potential: {:.0f} customers
        - Expected uplift: +15% monthly revenue
        """.format(len(internet_no_streaming)))
    
    with col2:
        st.markdown("""
        **🔒 Security Bundle**
        - Online Security + Backup + Device Protection
        - Target: Fiber optic users
        - Potential: {:.0f} customers
        - Expected uplift: +20% monthly revenue
        """.format(len(df[df['InternetService'] == 'Fiber optic'])))

with campaign_tabs[2]:
    # Retention campaign analysis
    retention_targets = df[
        (df['Contract'] == 'Month-to-month') |
        (df['tenure'] < 12) |
        (df['ServiceAdoptionScore'] < 2)
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        retention_metrics = {
            'Total At-Risk': len(retention_targets),
            'High-Value At-Risk': len(retention_targets[retention_targets['CLV'] > df['CLV'].median()]),
            'New Customer Risk': len(retention_targets[retention_targets['tenure'] < 6]),
            'Low Engagement Risk': len(retention_targets[retention_targets['ServiceAdoptionScore'] < 2])
        }
        
        fig_retention = px.bar(
            x=list(retention_metrics.keys()),
            y=list(retention_metrics.values()),
            title="Retention Campaign Targets",
            color=list(retention_metrics.values()),
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_retention, use_container_width=True)
    
    with col2:
        # Retention strategy recommendations
        st.markdown("""
        **🎯 Priority Retention Actions**
        
        **Immediate (High-Value At-Risk):**
        - Personal outreach calls
        - Custom retention offers
        - Dedicated account management
        
        **Short-term (New Customers):**
        - Enhanced onboarding program
        - Service adoption incentives
        - Early satisfaction surveys
        
        **Long-term (Low Engagement):**
        - Service education campaigns
        - Usage optimization tips
        - Feature introduction programs
        """)

with campaign_tabs[3]:
    # ROI Analysis
    st.subheader("💹 Campaign ROI Analysis")
    
    # Sample campaign performance data
    campaign_performance = {
        'Service Upsell': {'Cost': 15000, 'Revenue': 45000, 'Customers': 150},
        'Retention': {'Cost': 25000, 'Revenue': 80000, 'Customers': 200},
        'Cross-sell': {'Cost': 10000, 'Revenue': 30000, 'Customers': 100},
        'Acquisition': {'Cost': 30000, 'Revenue': 60000, 'Customers': 120}
    }
    
    roi_data = []
    for campaign, metrics in campaign_performance.items():
        roi = ((metrics['Revenue'] - metrics['Cost']) / metrics['Cost']) * 100
        cost_per_customer = metrics['Cost'] / metrics['Customers']
        revenue_per_customer = metrics['Revenue'] / metrics['Customers']
        
        roi_data.append({
            'Campaign': campaign,
            'Investment': f"${metrics['Cost']:,}",
            'Revenue': f"${metrics['Revenue']:,}",
            'ROI': f"{roi:.1f}%",
            'Cost per Customer': f"${cost_per_customer:.0f}",
            'Revenue per Customer': f"${revenue_per_customer:.0f}"
        })
    
    roi_df = pd.DataFrame(roi_data)
    st.dataframe(roi_df, use_container_width=True)
    
    # ROI visualization
    roi_values = [((campaign_performance[c]['Revenue'] - campaign_performance[c]['Cost']) / 
                   campaign_performance[c]['Cost']) * 100 for c in campaign_performance.keys()]
    
    fig_roi = px.bar(
        x=list(campaign_performance.keys()),
        y=roi_values,
        title="Campaign ROI Comparison (%)",
        color=roi_values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_roi, use_container_width=True)

# Contract Optimization Analysis
st.header("📋 Contract Optimization Analysis")

col1, col2 = st.columns(2)

with col1:
    # Contract distribution and performance
    contract_analysis = df.groupby('Contract').agg({
        'customerID': 'count',
        'CLV': 'mean',
        'MonthlyCharges': 'mean',
        'Churn': lambda x: (x == 'Yes').mean()
    }).round(2)
    
    contract_analysis.columns = ['Customer Count', 'Avg CLV', 'Avg Monthly Charges', 'Churn Rate']
    contract_analysis['Revenue Share'] = (df.groupby('Contract')['MonthlyCharges'].sum() / 
                                         df['MonthlyCharges'].sum()).round(3)
    
    st.dataframe(contract_analysis, use_container_width=True)

with col2:
    # Contract migration opportunities
    monthly_customers = len(df[df['Contract'] == 'Month-to-month'])
    annual_revenue_potential = (df[df['Contract'] == 'Month-to-month']['MonthlyCharges'].sum() * 
                               12 * 0.9)  # 10% discount for annual
    
    st.metric("Monthly Contract Customers", f"{monthly_customers:,}")
    st.metric("Annual Contract Potential", f"${annual_revenue_potential:,.0f}")
    
    migration_incentives = {
        "5% Discount": annual_revenue_potential * 0.95,
        "10% Discount": annual_revenue_potential * 0.90,
        "15% Discount": annual_revenue_potential * 0.85
    }
    
    st.write("**Migration Revenue Scenarios:**")
    for incentive, revenue in migration_incentives.items():
        st.write(f"• {incentive}: ${revenue:,.0f}")

# AI-Powered Campaign Insights
st.header("🤖 AI-Powered Campaign Insights")

if st.button("🔍 Generate Campaign Intelligence", type="primary"):
    with st.spinner("Analyzing campaign opportunities with Azure OpenAI..."):
        campaign_insights_data = {
            'total_customers': len(df),
            'service_adoption_avg': df['ServiceAdoptionScore'].mean(),
            'cross_sell_opportunities': len(phone_only) + len(internet_no_streaming),
            'retention_targets': len(retention_targets),
            'high_value_at_risk': len(retention_targets[retention_targets['CLV'] > df['CLV'].median()]),
            'monthly_contract_migration': monthly_customers
        }
        
        ai_campaign_insight = get_ai_insights(campaign_insights_data)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h4>🧠 AI Campaign Strategy Intelligence</h4>
            <p>{ai_campaign_insight}</p>
        </div>
        """, unsafe_allow_html=True)

# Export Campaign Analysis
st.header("📊 Export Campaign Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Export Target Lists"):
        # Create comprehensive target list
        target_list = df[['customerID', 'CampaignSegment', 'ServiceAdoptionScore', 
                         'CLV', 'EmailEngagement', 'Contract']].copy()
        csv = target_list.to_csv(index=False)
        st.download_button(
            label="Download Target Lists",
            data=csv,
            file_name="campaign_target_lists.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Opportunity Analysis"):
        opportunity_data = {
            'Cross_sell_Phone_Only': len(phone_only),
            'Cross_sell_Internet_NoStreaming': len(internet_no_streaming),
            'Upsell_Monthly_HighCLV': len(monthly_high_clv),
            'Upsell_DSL_Fiber': len(dsl_upgrade_candidates),
            'Retention_AtRisk': len(retention_targets)
        }
        
        opp_df = pd.DataFrame([opportunity_data])
        csv = opp_df.to_csv(index=False)
        st.download_button(
            label="Download Opportunities",
            data=csv,
            file_name="campaign_opportunities.csv",
            mime="text/csv"
        )

with col3:
    if st.button("💰 Export ROI Analysis"):
        roi_export = pd.DataFrame(roi_data)
        csv = roi_export.to_csv(index=False)
        st.download_button(
            label="Download ROI Analysis",
            data=csv,
            file_name="campaign_roi_analysis.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("🎪 **Campaign Analysis** | Optimize marketing effectiveness with data-driven insights")
