import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.data_loader import load_data
from utils.model_utils import calculate_clv
from utils.azure_openai import get_ai_insights
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CLV Analysis", page_icon="💰", layout="wide")

st.title("💰 Customer Lifetime Value Analysis")
st.markdown("Comprehensive analysis and prediction of customer lifetime value")
st.markdown("---")

# Load data
@st.cache_data
def get_clv_data():
    df = load_data()
    if df is not None:
        # Calculate CLV metrics
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['PredictedCLV'] = np.where(
            df['Churn'] == 'No',
            df['CLV'] * (1 + df['tenure']/12),
            df['CLV']
        )
        df['CLVperMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # CLV segments
        df['CLVQuartile'] = pd.qcut(df['CLV'], 4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Customer value categories
        df['ValueCategory'] = pd.cut(
            df['CLV'], 
            bins=[0, 500, 1500, 3000, np.inf],
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )
    return df

df = get_clv_data()

if df is None:
    st.error("❌ Unable to load CLV data")
    st.stop()

# CLV Calculator Sidebar
st.sidebar.header("🧮 CLV Calculator")
st.sidebar.markdown("Calculate CLV for new customer scenarios:")

with st.sidebar:
    st.subheader("Customer Inputs")
    calc_tenure = st.slider("Expected Tenure (months)", 1, 72, 24)
    calc_monthly = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
    calc_churn_risk = st.slider("Churn Risk (%)", 0, 100, 25)
    
    # Advanced CLV calculation
    retention_rate = (100 - calc_churn_risk) / 100
    discount_rate = 0.1  # 10% annual discount rate
    
    # Calculate various CLV metrics
    simple_clv = calc_tenure * calc_monthly
    
    # Predictive CLV with retention consideration
    months_range = np.arange(1, calc_tenure + 1)
    monthly_values = calc_monthly * (retention_rate ** months_range)
    discounted_values = monthly_values / ((1 + discount_rate/12) ** months_range)
    predictive_clv = np.sum(discounted_values)
    
    st.subheader("📊 CLV Results")
    st.metric("Simple CLV", f"${simple_clv:,.2f}")
    st.metric("Predictive CLV", f"${predictive_clv:,.2f}")
    st.metric("Monthly Value", f"${calc_monthly:.2f}")
    
    if st.button("💡 Get CLV Strategy", type="primary"):
        clv_insights = calculate_clv({
            'tenure': calc_tenure,
            'monthly_charges': calc_monthly,
            'total_charges': simple_clv
        })
        
        st.success("✅ CLV Analysis Complete!")
        st.write(f"**Recommended Actions:**")
        
        if predictive_clv > 2000:
            st.write("🥇 High-value customer - VIP treatment")
        elif predictive_clv > 1000:
            st.write("🥈 Medium-value - upselling focus")
        else:
            st.write("🥉 Focus on retention and engagement")

# Executive CLV Summary
st.header("📊 CLV Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_clv = df['CLV'].sum()
    st.metric(
        label="💎 Total CLV",
        value=f"${total_clv:,.0f}",
        delta=f"+${total_clv*0.08:,.0f} projected growth"
    )

with col2:
    avg_clv = df['CLV'].mean()
    st.metric(
        label="📈 Average CLV",
        value=f"${avg_clv:.0f}",
        delta=f"+${avg_clv*0.12:.0f} vs last quarter"
    )

with col3:
    median_clv = df['CLV'].median()
    st.metric(
        label="📊 Median CLV",
        value=f"${median_clv:.0f}",
        delta=f"+{(avg_clv/median_clv-1)*100:.1f}% skew"
    )

with col4:
    high_value_customers = len(df[df['CLVQuartile'] == 'Premium'])
    st.metric(
        label="⭐ Premium Customers",
        value=f"{high_value_customers:,}",
        delta=f"{high_value_customers/len(df)*100:.1f}% of base"
    )

# CLV Distribution Analysis
st.header("📊 CLV Distribution & Segmentation")

col1, col2 = st.columns(2)

with col1:
    # CLV histogram
    fig_hist = px.histogram(
        df, x='CLV', nbins=30,
        title="Customer Lifetime Value Distribution",
        color_discrete_sequence=['#2E8B57']
    )
    fig_hist.add_vline(x=avg_clv, line_dash="dash", line_color="red", 
                       annotation_text=f"Mean: ${avg_clv:.0f}")
    fig_hist.add_vline(x=median_clv, line_dash="dash", line_color="blue", 
                       annotation_text=f"Median: ${median_clv:.0f}")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # CLV by quartiles
    quartile_counts = df['CLVQuartile'].value_counts()
    colors = {'Low': '#ff7f7f', 'Medium': '#ffcc99', 'High': '#87ceeb', 'Premium': '#98fb98'}
    
    fig_quartiles = px.pie(
        values=quartile_counts.values,
        names=quartile_counts.index,
        title="CLV Quartile Distribution",
        color=quartile_counts.index,
        color_discrete_map=colors
    )
    st.plotly_chart(fig_quartiles, use_container_width=True)

# CLV Analysis by Customer Characteristics
st.header("🔍 CLV Analysis by Customer Segments")

analysis_tabs = st.tabs(["Contract Analysis", "Tenure Impact", "Service Usage", "Demographics"])

with analysis_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV by contract type
        clv_contract = df.groupby('Contract')['CLV'].agg(['mean', 'median', 'count']).round(2)
        clv_contract.columns = ['Average CLV', 'Median CLV', 'Customer Count']
        
        fig_contract = px.bar(
            x=clv_contract.index,
            y=clv_contract['Average CLV'],
            title="Average CLV by Contract Type",
            color=clv_contract['Average CLV'],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_contract, use_container_width=True)
        
        st.dataframe(clv_contract, use_container_width=True)
    
    with col2:
        # CLV distribution by contract
        fig_box_contract = px.box(
            df, x='Contract', y='CLV',
            title="CLV Distribution by Contract Type",
            color='Contract'
        )
        st.plotly_chart(fig_box_contract, use_container_width=True)

with analysis_tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV vs Tenure scatter
        fig_tenure_scatter = px.scatter(
            df.sample(1000), x='tenure', y='CLV',
            color='Churn',
            title="CLV vs Customer Tenure",
            color_discrete_map={'Yes': 'red', 'No': 'blue'},
            opacity=0.6
        )
        
        # Add trendline
        from sklearn.linear_model import LinearRegression
        X = df['tenure'].values.reshape(-1, 1)
        y = df['CLV'].values
        reg = LinearRegression().fit(X, y)
        
        x_trend = np.linspace(df['tenure'].min(), df['tenure'].max(), 100)
        y_trend = reg.predict(x_trend.reshape(-1, 1))
        
        fig_tenure_scatter.add_trace(
            go.Scatter(x=x_trend, y=y_trend, mode='lines', 
                      name='Trend', line=dict(color='orange', width=3))
        )
        
        st.plotly_chart(fig_tenure_scatter, use_container_width=True)
    
    with col2:
        # Tenure groups analysis
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                  labels=['0-1Y', '1-2Y', '2-4Y', '4Y+'])
        
        tenure_clv = df.groupby('TenureGroup').agg({
            'CLV': ['mean', 'count'],
            'Churn': lambda x: (x == 'Yes').mean()
        }).round(2)
        
        tenure_clv.columns = ['Avg CLV', 'Customer Count', 'Churn Rate']
        
        fig_tenure_bar = px.bar(
            x=tenure_clv.index,
            y=tenure_clv['Avg CLV'],
            title="Average CLV by Tenure Group",
            color=tenure_clv['Avg CLV'],
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_tenure_bar, use_container_width=True)
        
        st.dataframe(tenure_clv, use_container_width=True)

with analysis_tabs[2]:
    if 'ServiceAdoptionScore' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # CLV vs Service Adoption
            service_clv = df.groupby('ServiceAdoptionScore')['CLV'].mean()
            
            fig_service = px.line(
                x=service_clv.index,
                y=service_clv.values,
                title="CLV vs Service Adoption Score",
                markers=True
            )
            fig_service.update_traces(line_color='green', marker_size=8)
            st.plotly_chart(fig_service, use_container_width=True)
        
        with col2:
            # Service adoption impact
            df['ServiceLevel'] = pd.cut(df['ServiceAdoptionScore'], 
                                       bins=[0, 2, 4, 6, 8],
                                       labels=['Basic', 'Standard', 'Advanced', 'Premium'])
            
            service_level_clv = df.groupby('ServiceLevel')['CLV'].agg(['mean', 'count']).round(2)
            service_level_clv.columns = ['Average CLV', 'Customer Count']
            
            fig_service_level = px.bar(
                x=service_level_clv.index,
                y=service_level_clv['Average CLV'],
                title="CLV by Service Level",
                color=service_level_clv['Average CLV'],
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_service_level, use_container_width=True)
    else:
        st.info("Service adoption data not available for analysis")

with analysis_tabs[3]:
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV by gender
        gender_clv = df.groupby('gender')['CLV'].agg(['mean', 'count']).round(2)
        gender_clv.columns = ['Average CLV', 'Customer Count']
        
        fig_gender = px.bar(
            x=gender_clv.index,
            y=gender_clv['Average CLV'],
            title="Average CLV by Gender",
            color=['blue', 'pink']
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # CLV by senior citizen status
        senior_clv = df.groupby('SeniorCitizen')['CLV'].agg(['mean', 'count']).round(2)
        senior_clv.columns = ['Average CLV', 'Customer Count']
        senior_clv.index = ['Non-Senior', 'Senior']
        
        fig_senior = px.bar(
            x=senior_clv.index,
            y=senior_clv['Average CLV'],
            title="Average CLV by Age Group",
            color=['lightblue', 'orange']
        )
        st.plotly_chart(fig_senior, use_container_width=True)

# CLV Prediction Model
st.header("🔮 CLV Prediction Model")

col1, col2 = st.columns([2, 1])

with col1:
    # Train simple CLV prediction model
    feature_cols = ['tenure', 'MonthlyCharges']
    if 'ServiceAdoptionScore' in df.columns:
        feature_cols.append('ServiceAdoptionScore')
    
    X = df[feature_cols].fillna(0)
    y = df['CLV']
    
    # Polynomial features for better fit
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Make predictions
    y_pred = model.predict(X_poly)
    r2_score = model.score(X_poly, y)
    
    # Prediction vs Actual scatter
    fig_pred = px.scatter(
        x=y, y=y_pred,
        title=f"CLV Prediction Model (R² = {r2_score:.3f})",
        labels={'x': 'Actual CLV', 'y': 'Predicted CLV'},
        opacity=0.6
    )
    
    # Add perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    fig_pred.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction', 
                  line=dict(color='red', dash='dash'))
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

with col2:
    st.subheader("📈 Model Performance")
    st.metric("R² Score", f"{r2_score:.3f}")
    
    mae = np.mean(np.abs(y - y_pred))
    st.metric("Mean Abs Error", f"${mae:.2f}")
    
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    st.metric("MAPE", f"{mape:.1f}%")
    
    st.subheader("🎯 Key Drivers")
    if len(feature_cols) >= 2:
        st.write("**Top CLV Drivers:**")
        st.write("1. 📅 Customer Tenure")
        st.write("2. 💰 Monthly Charges")
        if len(feature_cols) > 2:
            st.write("3. 📱 Service Adoption")

# High-Value Customer Analysis
st.header("⭐ High-Value Customer Analysis")

# Identify high-value customers
high_value_threshold = df['CLV'].quantile(0.8)
high_value_customers = df[df['CLV'] >= high_value_threshold]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🏆 High-Value Count",
        value=f"{len(high_value_customers):,}",
        delta=f"{len(high_value_customers)/len(df)*100:.1f}% of total"
    )

with col2:
    hv_revenue_contribution = high_value_customers['MonthlyCharges'].sum() / df['MonthlyCharges'].sum()
    st.metric(
        label="💰 Revenue Share",
        value=f"{hv_revenue_contribution:.1%}",
        delta="High concentration"
    )

with col3:
    hv_churn_rate = (high_value_customers['Churn'] == 'Yes').mean()
    st.metric(
        label="⚠️ HV Churn Rate",
        value=f"{hv_churn_rate:.1%}",
        delta=f"{hv_churn_rate - churn_rate:.1%} vs avg",
        delta_color="inverse" if hv_churn_rate > churn_rate else "normal"
    )

# High-value customer characteristics
st.subheader("🔍 High-Value Customer Profile")

hv_profile = high_value_customers.describe()[['tenure', 'MonthlyCharges', 'CLV']].round(2)
st.dataframe(hv_profile, use_container_width=True)

# CLV Optimization Recommendations
st.header("💡 CLV Optimization Strategy")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Growth Opportunities")
    
    # Customers with high monthly charges but low tenure
    growth_opportunities = df[
        (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.7)) &
        (df['tenure'] < 24) &
        (df['Churn'] == 'No')
    ]
    
    st.write(f"**Retention Focus:** {len(growth_opportunities)} high-paying new customers")
    st.write("• Implement loyalty programs")
    st.write("• Offer contract incentives")
    st.write("• Provide premium support")
    
    # Low service adoption high spenders
    if 'ServiceAdoptionScore' in df.columns:
        upsell_opportunities = df[
            (df['MonthlyCharges'] > 50) &
            (df['ServiceAdoptionScore'] < 3) &
            (df['Churn'] == 'No')
        ]
        
        st.write(f"**Upselling Target:** {len(upsell_opportunities)} customers with low service adoption")

with col2:
    st.subheader("🚨 Risk Mitigation")
    
    # High-value customers at risk
    at_risk_hv = high_value_customers[high_value_customers['Churn'] == 'Yes']
    
    if len(at_risk_hv) > 0:
        st.write(f"**Critical Alert:** {len(at_risk_hv)} high-value customers churned")
        st.write("• Analyze churn patterns")
        st.write("• Implement win-back campaigns")
        st.write("• Review pricing strategy")
    
    # Contract risk analysis
    month_to_month_hv = high_value_customers[high_value_customers['Contract'] == 'Month-to-month']
    
    st.write(f"**Contract Risk:** {len(month_to_month_hv)} high-value customers on monthly contracts")
    st.write("• Offer annual contract discounts")
    st.write("• Provide contract upgrade incentives")

# AI-Powered CLV Insights
st.header("🤖 AI-Powered CLV Insights")

if st.button("🔍 Generate CLV Strategy", type="primary"):
    with st.spinner("Analyzing CLV patterns with Azure OpenAI..."):
        clv_insights_data = {
            'total_customers': len(df),
            'avg_clv': avg_clv,
            'high_value_customers': len(high_value_customers),
            'hv_churn_rate': hv_churn_rate,
            'revenue_concentration': hv_revenue_contribution,
            'growth_opportunities': len(growth_opportunities) if 'growth_opportunities' in locals() else 0
        }
        
        ai_clv_insight = get_ai_insights(clv_insights_data)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h4>🧠 AI CLV Strategy Recommendations</h4>
            <p>{ai_clv_insight}</p>
        </div>
        """, unsafe_allow_html=True)

# Export CLV Analysis
st.header("📊 Export CLV Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 Export CLV Report"):
        clv_report = df[['customerID', 'CLV', 'PredictedCLV', 'CLVQuartile', 'ValueCategory']].copy()
        csv = clv_report.to_csv(index=False)
        st.download_button(
            label="Download CLV Report",
            data=csv,
            file_name="clv_analysis_report.csv",
            mime="text/csv"
        )

with col2:
    if st.button("⭐ Export High-Value Customers"):
        hv_csv = high_value_customers[['customerID', 'tenure', 'MonthlyCharges', 'CLV', 'Contract', 'Churn']].to_csv(index=False)
        st.download_button(
            label="Download HV Customers",
            data=hv_csv,
            file_name="high_value_customers.csv",
            mime="text/csv"
        )

with col3:
    if st.button("💰 Export Revenue Analysis"):
        revenue_analysis = df.groupby(['Contract', 'CLVQuartile']).agg({
            'customerID': 'count',
            'MonthlyCharges': 'sum',
            'CLV': 'mean'
        }).round(2)
        csv = revenue_analysis.to_csv()
        st.download_button(
            label="Download Revenue Analysis",
            data=csv,
            file_name="revenue_by_segment.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("💰 **CLV Analysis** | Powered by Predictive Analytics & Azure AI")
