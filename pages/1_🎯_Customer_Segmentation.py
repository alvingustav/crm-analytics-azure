import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_loader import load_data, load_model_artifacts
from utils.azure_openai import analyze_customer_segment

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")

st.title("🎯 Customer Segmentation Analysis")
st.markdown("---")

# Load data
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_models():
    return load_model_artifacts()

df = get_data()
artifacts = get_models()

if df is None or artifacts is None:
    st.error("❌ Failed to load data or models")
    st.stop()

# Sidebar for manual input
st.sidebar.header("🔧 Manual Customer Input")
st.sidebar.markdown("Enter customer details to see segment prediction:")

with st.sidebar:
    # Input fields for segmentation
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 1500.0)
    
    # Service adoption
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    
    # Calculate service adoption score
    services = [phone_service, online_security, online_backup]
    service_adoption_score = sum(1 for s in services if s == "Yes")
    digital_engagement = service_adoption_score / len(services)
    
    if st.button("🔮 Predict Customer Segment", type="primary"):
        try:
            # Prepare input data for segmentation
            input_data = np.array([[tenure, monthly_charges, total_charges, 
                                  service_adoption_score, digital_engagement]])
            
            # Scale the input
            input_scaled = artifacts['segment_scaler'].transform(input_data)
            
            # Predict segment
            segment = artifacts['segment_model'].predict(input_scaled)[0]
            
            segment_names = {
                0: 'Price-Sensitive Basic Users',
                1: 'High-Value Long-Term Customers', 
                2: 'New Digital Adopters',
                3: 'Premium Service Users'
            }
            
            predicted_segment = segment_names.get(segment, 'Unknown')
            
            st.success(f"🎯 **Predicted Segment:** {predicted_segment}")
            
            # Get AI insights for this segment
            segment_data = {
                'segment_name': predicted_segment,
                'avg_clv': tenure * monthly_charges,
                'churn_rate': 0.15,  # Default estimate
                'customer_count': 1,
                'avg_tenure': tenure
            }
            
            ai_insight = analyze_customer_segment(segment_data)
            st.markdown(f"🤖 **AI Strategy Recommendation:**\n{ai_insight}")
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Main content - Existing customer analysis
st.header("📊 Current Customer Segmentation")

# Perform segmentation on existing data
segmentation_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                        'ServiceAdoptionScore']

if all(col in df.columns for col in segmentation_features):
    # Create segments based on business logic (since we don't have pre-trained segments)
    df['CustomerSegment'] = 0  # Default
    
    # Segment 0: Price-Sensitive Basic Users
    df.loc[(df['MonthlyCharges'] < 50) & (df['ServiceAdoptionScore'] <= 2), 'CustomerSegment'] = 0
    
    # Segment 1: High-Value Long-Term Customers
    df.loc[(df['tenure'] > 24) & (df['MonthlyCharges'] > 60), 'CustomerSegment'] = 1
    
    # Segment 2: New Digital Adopters
    df.loc[(df['tenure'] < 12) & (df['ServiceAdoptionScore'] >= 3), 'CustomerSegment'] = 2
    
    # Segment 3: Premium Service Users
    df.loc[(df['MonthlyCharges'] > 80) & (df['ServiceAdoptionScore'] >= 4), 'CustomerSegment'] = 3
    
    segment_names = {
        0: 'Price-Sensitive Basic Users',
        1: 'High-Value Long-Term Customers', 
        2: 'New Digital Adopters',
        3: 'Premium Service Users'
    }
    
    df['SegmentName'] = df['CustomerSegment'].map(segment_names)

# Segment analysis
col1, col2 = st.columns([1, 1])

with col1:
    # Segment distribution
    segment_counts = df['SegmentName'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_segments, use_container_width=True)

with col2:
    # Churn rate by segment
    churn_by_segment = df.groupby('SegmentName')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).round(1)
    
    fig_churn = px.bar(
        x=churn_by_segment.index,
        y=churn_by_segment.values,
        title="Churn Rate by Segment (%)",
        color=churn_by_segment.values,
        color_continuous_scale="Reds"
    )
    fig_churn.update_layout(showlegend=False)
    st.plotly_chart(fig_churn, use_container_width=True)

# Detailed segment analysis
st.header("📈 Segment Performance Analysis")

segment_stats = df.groupby('SegmentName').agg({
    'customerID': 'count',
    'tenure': 'mean',
    'MonthlyCharges': 'mean',
    'TotalCharges': 'mean',
    'CLV': 'mean',
    'ServiceAdoptionScore': 'mean',
    'Churn': lambda x: (x == 'Yes').mean()
}).round(2)

segment_stats.columns = ['Customer Count', 'Avg Tenure (months)', 'Avg Monthly Charges ($)', 
                        'Avg Total Charges ($)', 'Avg CLV ($)', 'Avg Service Adoption', 'Churn Rate']

st.dataframe(segment_stats, use_container_width=True)

# Demographic segmentation
st.header("👥 Demographic Segmentation")

col1, col2, col3 = st.columns(3)

with col1:
    # Gender distribution
    gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index') * 100
    fig_gender = px.bar(
        gender_churn,
        title="Churn Rate by Gender (%)",
        color_discrete_sequence=['#00cc96', '#ef553b']
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    # Senior citizen analysis
    senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index') * 100
    fig_senior = px.bar(
        senior_churn,
        title="Churn Rate by Age Group (%)",
        color_discrete_sequence=['#00cc96', '#ef553b']
    )
    fig_senior.update_xaxis(ticktext=['Non-Senior', 'Senior'], tickvals=[0, 1])
    st.plotly_chart(fig_senior, use_container_width=True)

with col3:
    # Contract type analysis
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    fig_contract = px.bar(
        contract_churn,
        title="Churn Rate by Contract Type (%)",
        color_discrete_sequence=['#00cc96', '#ef553b']
    )
    st.plotly_chart(fig_contract, use_container_width=True)

# Value-based segmentation
st.header("💰 Value-Based Segmentation")

# CLV quartiles
df['CLV_Quartile'] = pd.qcut(df['CLV'], 4, labels=['Low Value', 'Medium Value', 'High Value', 'Premium Value'])

col1, col2 = st.columns(2)

with col1:
    clv_distribution = df['CLV_Quartile'].value_counts()
    fig_clv = px.bar(
        x=clv_distribution.index,
        y=clv_distribution.values,
        title="Customer Distribution by Value Segment",
        color=clv_distribution.values,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_clv, use_container_width=True)

with col2:
    clv_churn = df.groupby('CLV_Quartile')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    fig_clv_churn = px.bar(
        x=clv_churn.index,
        y=clv_churn.values,
        title="Churn Rate by Value Segment (%)",
        color=clv_churn.values,
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_clv_churn, use_container_width=True)

# Business insights
st.header("💡 Segment-Specific Recommendations")

recommendations = {
    'Price-Sensitive Basic Users': {
        'strategy': 'Focus on value proposition and basic service reliability',
        'tactics': ['Offer competitive pricing', 'Emphasize service quality', 'Provide basic support packages']
    },
    'High-Value Long-Term Customers': {
        'strategy': 'Maintain satisfaction and explore upselling opportunities',
        'tactics': ['VIP customer service', 'Exclusive offers', 'Premium feature trials']
    },
    'New Digital Adopters': {
        'strategy': 'Accelerate onboarding and feature adoption',
        'tactics': ['Digital tutorials', 'Progressive feature rollout', 'Early adopter rewards']
    },
    'Premium Service Users': {
        'strategy': 'Maximize revenue and ensure premium experience',
        'tactics': ['White-glove service', 'Beta feature access', 'Personal account management']
    }
}

for segment, rec in recommendations.items():
    with st.expander(f"📋 {segment} Strategy"):
        st.write(f"**Strategy:** {rec['strategy']}")
        st.write("**Tactics:**")
        for tactic in rec['tactics']:
            st.write(f"• {tactic}")
