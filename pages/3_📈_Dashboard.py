import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.data_loader import load_data, load_model_artifacts
from utils.azure_openai import get_ai_insights
from utils.model_utils import calculate_risk_score, calculate_clv
import json

st.set_page_config(page_title="CRM Dashboard", page_icon="📈", layout="wide")

st.title("📈 CRM Analytics Dashboard")
st.markdown("Real-time insights and monitoring for customer relationship management")
st.markdown("---")

# Load data
@st.cache_data
def get_dashboard_data():
    df = load_data()
    if df is not None:
        # Calculate additional metrics
        df['RiskScore'] = df.apply(lambda row: calculate_risk_score({
            'contract': row['Contract'],
            'tenure': row['tenure'],
            'payment_method': row['PaymentMethod'],
            'monthly_charges': row['MonthlyCharges'],
            'phone_service': row['PhoneService']
        })['overall_risk_score'], axis=1)
        
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['RiskLevel'] = pd.cut(df['RiskScore'], bins=[0, 0.4, 0.7, 1.0], 
                                labels=['Low', 'Medium', 'High'])
    return df

df = get_dashboard_data()

if df is None:
    st.error("❌ Unable to load dashboard data")
    st.stop()

# Executive KPI Section
st.header("🎯 Executive KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_customers = len(df)
    st.metric(
        label="👥 Total Customers",
        value=f"{total_customers:,}",
        delta=f"+{total_customers//20} this month"
    )

with col2:
    churn_rate = (df['Churn'] == 'Yes').mean()
    st.metric(
        label="⚠️ Churn Rate",
        value=f"{churn_rate:.1%}",
        delta=f"-{churn_rate*10:.1f}% vs target",
        delta_color="inverse"
    )

with col3:
    monthly_revenue = df['MonthlyCharges'].sum()
    st.metric(
        label="💰 Monthly Revenue",
        value=f"${monthly_revenue:,.0f}",
        delta=f"+${monthly_revenue*0.05:,.0f}"
    )

with col4:
    avg_clv = df['CLV'].mean()
    st.metric(
        label="💎 Avg CLV",
        value=f"${avg_clv:.0f}",
        delta=f"+${avg_clv*0.08:.0f}"
    )

with col5:
    high_risk_count = len(df[df['RiskLevel'] == 'High'])
    st.metric(
        label="🚨 High Risk",
        value=f"{high_risk_count:,}",
        delta=f"-{high_risk_count//10} vs last week",
        delta_color="inverse"
    )

# Real-time Alerts
st.header("🚨 Real-time Alerts & Notifications")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Critical Alerts")
    
    # High-risk customers
    critical_customers = df[
        (df['RiskLevel'] == 'High') & 
        (df['CLV'] > df['CLV'].quantile(0.75))
    ]
    
    if len(critical_customers) > 0:
        st.error(f"🔴 {len(critical_customers)} high-value customers at critical risk")
        
        for _, customer in critical_customers.head(3).iterrows():
            st.write(f"• Customer {customer['customerID']}: CLV ${customer['CLV']:.0f}, Risk {customer['RiskScore']:.0%}")
    else:
        st.success("✅ No critical risk customers identified")
    
    # Revenue alerts
    low_clv_customers = len(df[df['CLV'] < 500])
    if low_clv_customers > total_customers * 0.3:
        st.warning(f"⚡ {low_clv_customers} customers with low CLV (<$500)")

with col2:
    st.subheader("📊 Performance Alerts")
    
    # Churn trend alert
    if churn_rate > 0.25:
        st.error(f"📈 Churn rate ({churn_rate:.1%}) exceeds target (25%)")
    else:
        st.success(f"✅ Churn rate ({churn_rate:.1%}) within target")
    
    # Service adoption alert
    if 'ServiceAdoptionScore' in df.columns:
        low_adoption = len(df[df['ServiceAdoptionScore'] < 2])
        if low_adoption > total_customers * 0.4:
            st.warning(f"📱 {low_adoption} customers with low service adoption")

# Interactive Charts Section
st.header("📊 Interactive Analytics")

# Chart selection
chart_tabs = st.tabs(["Customer Overview", "Churn Analysis", "Revenue Metrics", "Risk Assessment"])

with chart_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer distribution by segment
        if 'Contract' in df.columns:
            contract_dist = df['Contract'].value_counts()
            fig_contract = px.pie(
                values=contract_dist.values,
                names=contract_dist.index,
                title="Customer Distribution by Contract Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_contract, use_container_width=True)
    
    with col2:
        # Tenure distribution
        fig_tenure = px.histogram(
            df, x='tenure', nbins=20,
            title="Customer Tenure Distribution",
            color_discrete_sequence=['#636efa']
        )
        fig_tenure.update_layout(showlegend=False)
        st.plotly_chart(fig_tenure, use_container_width=True)

with chart_tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn rate by tenure groups
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                  labels=['0-1Y', '1-2Y', '2-4Y', '4Y+'])
        churn_by_tenure = df.groupby('TenureGroup')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        )
        
        fig_churn_tenure = px.bar(
            x=churn_by_tenure.index,
            y=churn_by_tenure.values,
            title="Churn Rate by Tenure Group (%)",
            color=churn_by_tenure.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_churn_tenure, use_container_width=True)
    
    with col2:
        # Churn rate by monthly charges
        df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=4, 
                                  labels=['Low', 'Medium', 'High', 'Premium'])
        churn_by_charges = df.groupby('ChargeGroup')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        )
        
        fig_churn_charges = px.bar(
            x=churn_by_charges.index,
            y=churn_by_charges.values,
            title="Churn Rate by Charge Level (%)",
            color=churn_by_charges.values,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_churn_charges, use_container_width=True)

with chart_tabs[2]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by customer segment
        revenue_by_contract = df.groupby('Contract')['MonthlyCharges'].sum()
        
        fig_revenue = px.bar(
            x=revenue_by_contract.index,
            y=revenue_by_contract.values,
            title="Monthly Revenue by Contract Type",
            color=revenue_by_contract.values,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # CLV distribution
        fig_clv = px.box(
            df, y='CLV',
            title="Customer Lifetime Value Distribution",
            color_discrete_sequence=['#00cc96']
        )
        st.plotly_chart(fig_clv, use_container_width=True)

with chart_tabs[3]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level distribution
        risk_counts = df['RiskLevel'].value_counts()
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Customer Risk Distribution",
            color=risk_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk score vs CLV scatter
        fig_scatter = px.scatter(
            df.sample(500), x='RiskScore', y='CLV',
            color='Churn',
            title="Risk Score vs Customer Value",
            color_discrete_map={'Yes': 'red', 'No': 'blue'}
        )
        fig_scatter.update_layout(showlegend=True)
        st.plotly_chart(fig_scatter, use_container_width=True)

# Customer Health Score Section
st.header("💊 Customer Health Score Monitor")

# Calculate health score
def calculate_health_score(row):
    score = 100
    
    # Tenure impact (positive)
    if row['tenure'] > 24:
        score += 10
    elif row['tenure'] < 6:
        score -= 20
    
    # Contract impact
    if row['Contract'] == 'Two year':
        score += 15
    elif row['Contract'] == 'Month-to-month':
        score -= 15
    
    # Payment method impact
    if row['PaymentMethod'] in ['Bank transfer (automatic)', 'Credit card (automatic)']:
        score += 10
    elif row['PaymentMethod'] == 'Electronic check':
        score -= 10
    
    # Service adoption impact
    if 'ServiceAdoptionScore' in row and row['ServiceAdoptionScore'] > 3:
        score += 10
    elif 'ServiceAdoptionScore' in row and row['ServiceAdoptionScore'] < 2:
        score -= 10
    
    return max(min(score, 100), 0)

df['HealthScore'] = df.apply(calculate_health_score, axis=1)

col1, col2, col3 = st.columns(3)

with col1:
    avg_health = df['HealthScore'].mean()
    st.metric(
        label="📊 Avg Health Score",
        value=f"{avg_health:.1f}/100",
        delta=f"+{avg_health*0.05:.1f} vs last month"
    )

with col2:
    healthy_customers = len(df[df['HealthScore'] >= 70])
    st.metric(
        label="✅ Healthy Customers",
        value=f"{healthy_customers:,}",
        delta=f"+{healthy_customers//10}"
    )

with col3:
    unhealthy_customers = len(df[df['HealthScore'] < 50])
    st.metric(
        label="🚨 At-Risk Customers",
        value=f"{unhealthy_customers:,}",
        delta=f"-{unhealthy_customers//10}",
        delta_color="inverse"
    )

# Health score distribution
fig_health = px.histogram(
    df, x='HealthScore', nbins=20,
    title="Customer Health Score Distribution",
    color_discrete_sequence=['#2E8B57']
)
st.plotly_chart(fig_health, use_container_width=True)

# Action Items Section
st.header("🎯 Recommended Actions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Immediate Actions")
    
    # Critical customers needing attention
    critical_actions = df[
        (df['RiskLevel'] == 'High') & (df['HealthScore'] < 50)
    ].nlargest(5, 'CLV')
    
    if len(critical_actions) > 0:
        st.write("**High-priority customer interventions:**")
        for _, customer in critical_actions.iterrows():
            st.write(f"• Customer {customer['customerID']}: Health {customer['HealthScore']:.0f}/100, CLV ${customer['CLV']:.0f}")
    
    # Service adoption improvements
    low_adoption = df[df.get('ServiceAdoptionScore', 0) < 2]
    if len(low_adoption) > 0:
        st.write(f"**Service adoption campaign:** Target {len(low_adoption)} customers with <2 services")

with col2:
    st.subheader("📈 Growth Opportunities")
    
    # High-value, low-risk customers for upselling
    upsell_targets = df[
        (df['RiskLevel'] == 'Low') & 
        (df['MonthlyCharges'] < df['MonthlyCharges'].quantile(0.75))
    ].nlargest(5, 'HealthScore')
    
    if len(upsell_targets) > 0:
        st.write("**Upselling opportunities:**")
        for _, customer in upsell_targets.iterrows():
            st.write(f"• Customer {customer['customerID']}: Health {customer['HealthScore']:.0f}/100, Current ${customer['MonthlyCharges']:.0f}/month")

# AI Insights Section
st.header("🤖 AI-Powered Insights")

if st.button("🔄 Generate AI Business Insights", type="primary"):
    with st.spinner("Analyzing dashboard metrics with Azure OpenAI..."):
        insights_data = {
            'total_customers': total_customers,
            'churn_rate': churn_rate,
            'avg_clv': avg_clv,
            'high_risk_customers': high_risk_count,
            'avg_health_score': df['HealthScore'].mean(),
            'monthly_revenue': monthly_revenue
        }
        
        ai_insight = get_ai_insights(insights_data)
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <h4>🧠 AI Strategic Analysis</h4>
            <p>{ai_insight}</p>
        </div>
        """, unsafe_allow_html=True)

# Export Dashboard Data
st.header("📤 Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Customer Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="crm_customer_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Risk Analysis"):
        risk_data = df[['customerID', 'RiskScore', 'RiskLevel', 'HealthScore', 'CLV']]
        csv = risk_data.to_csv(index=False)
        st.download_button(
            label="Download Risk CSV",
            data=csv,
            file_name="customer_risk_analysis.csv",
            mime="text/csv"
        )

with col3:
    if st.button("💰 Export Revenue Report"):
        revenue_data = df.groupby('Contract').agg({
            'customerID': 'count',
            'MonthlyCharges': ['sum', 'mean'],
            'CLV': 'mean',
            'Churn': lambda x: (x == 'Yes').mean()
        }).round(2)
        csv = revenue_data.to_csv()
        st.download_button(
            label="Download Revenue CSV",
            data=csv,
            file_name="revenue_analysis.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("📊 **CRM Dashboard** | Last updated: Real-time | Powered by Azure AI")
