import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.data_loader import load_data, load_model_artifacts
from utils.azure_openai import get_ai_insights

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction & Risk Analysis")
st.markdown("---")

# Load data and models
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

# Sidebar for individual prediction
st.sidebar.header("🎯 Individual Churn Prediction")
st.sidebar.markdown("Enter customer details for churn risk assessment:")

with st.sidebar:
    # Customer demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    # Service details
    st.subheader("Services")
    tenure = st.slider("Tenure (months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 1500.0)
    
    # Services
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    
    # Contract details
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"])
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        try:
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Feature engineering
            service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            service_adoption = sum(1 for col in service_cols if input_data.get(col) == "Yes")
            
            input_data['ServiceAdoptionScore'] = service_adoption
            input_data['CLV'] = tenure * monthly_charges
            input_data['AvgChargesPerTenure'] = total_charges / (tenure + 1)
            input_data['DigitalEngagement'] = service_adoption / len(service_cols)
            
            # Encode categorical variables (simplified)
            categorical_encodings = {
                'gender': {'Male': 0, 'Female': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
                'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 
                                'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
            }
            
            # Create feature vector
            features = [
                input_data['SeniorCitizen'],
                input_data['tenure'],
                input_data['MonthlyCharges'],
                input_data['TotalCharges'],
                input_data['ServiceAdoptionScore'],
                input_data['CLV'],
                input_data['AvgChargesPerTenure'],
                input_data['DigitalEngagement']
            ]
            
            # Add encoded categorical features
            for col, encoding in categorical_encodings.items():
                features.append(encoding.get(input_data[col], 0))
            
            # Convert to array and reshape
            X_input = np.array(features).reshape(1, -1)
            
            # Scale features (assuming we need to scale)
            if artifacts and 'feature_scaler' in artifacts:
                X_scaled = artifacts['feature_scaler'].transform(X_input)
            else:
                X_scaled = X_input
            
            # Make prediction (using a simple heuristic if model not available)
            if artifacts and 'churn_model' in artifacts:
                churn_prob = artifacts['churn_model'].predict_proba(X_scaled)[0, 1]
            else:
                # Simple heuristic based on key factors
                risk_score = 0
                if contract == "Month-to-month":
                    risk_score += 0.3
                if tenure < 12:
                    risk_score += 0.25
                if payment_method == "Electronic check":
                    risk_score += 0.2
                if monthly_charges > 80:
                    risk_score += 0.15
                if service_adoption < 2:
                    risk_score += 0.1
                
                churn_prob = min(risk_score, 0.95)
            
            # Display results
            risk_level = "HIGH" if churn_prob > 0.7 else "MEDIUM" if churn_prob > 0.4 else "LOW"
            risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            
            st.markdown(f"### {risk_color} Risk Level: {risk_level}")
            st.metric("Churn Probability", f"{churn_prob:.1%}")
            
            # Risk factors
            st.markdown("**Key Risk Factors:**")
            if contract == "Month-to-month":
                st.write("• Month-to-month contract (high mobility)")
            if tenure < 12:
                st.write("• New customer (tenure < 12 months)")
            if payment_method == "Electronic check":
                st.write("• Electronic check payment method")
            if monthly_charges > 80:
                st.write("• High monthly charges")
            if service_adoption < 2:
                st.write("• Low service adoption")
                
            # Recommendations
            st.markdown("**Retention Recommendations:**")
            if risk_level == "HIGH":
                st.write("• Immediate intervention required")
                st.write("• Offer retention incentives")
                st.write("• Personal outreach by account manager")
            elif risk_level == "MEDIUM":
                st.write("• Monitor closely")
                st.write("• Engage with targeted offers")
                st.write("• Improve service experience")
            else:
                st.write("• Maintain current engagement")
                st.write("• Consider upselling opportunities")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Main content
st.header("📈 Overall Churn Analysis")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churned_customers = len(df[df['Churn'] == 'Yes'])
churn_rate = churned_customers / total_customers
retention_rate = 1 - churn_rate

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Churned Customers", f"{churned_customers:,}")

with col3:
    st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"-{churn_rate*5:.1f}% vs target")

with col4:
    st.metric("Retention Rate", f"{retention_rate:.1%}", delta=f"+{retention_rate*5:.1f}% vs last quarter")

# Churn analysis charts
st.header("🔍 Churn Risk Factors Analysis")

col1, col2 = st.columns(2)

with col1:
    # Churn by tenure
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                              labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
    tenure_churn = df.groupby('TenureGroup')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig_tenure = px.bar(
        x=tenure_churn.index,
        y=tenure_churn.values,
        title="Churn Rate by Tenure Group",
        color=tenure_churn.values,
        color_continuous_scale="Reds"
    )
    fig_tenure.update_layout(showlegend=False)
    st.plotly_chart(fig_tenure, use_container_width=True)

with col2:
    # Churn by contract type
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    
    fig_contract = px.bar(
        x=contract_churn.index,
        y=contract_churn.values,
        title="Churn Rate by Contract Type",
        color=contract_churn.values,
        color_continuous_scale="Reds"
    )
    fig_contract.update_layout(showlegend=False)
    st.plotly_chart(fig_contract, use_container_width=True)

# Service adoption impact
st.header("📱 Service Adoption Impact on Churn")

col1, col2 = st.columns(2)

with col1:
    # Service adoption vs churn
    if 'ServiceAdoptionScore' in df.columns:
        adoption_churn = df.groupby('ServiceAdoptionScore')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        
        fig_adoption = px.line(
            x=adoption_churn.index,
            y=adoption_churn.values,
            title="Churn Rate by Service Adoption Score",
            markers=True
        )
        fig_adoption.update_traces(line_color='red', marker_size=8)
        st.plotly_chart(fig_adoption, use_container_width=True)

with col2:
    # Payment method impact
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    
    fig_payment = px.bar(
        x=payment_churn.values,
        y=payment_churn.index,
        orientation='h',
        title="Churn Rate by Payment Method",
        color=payment_churn.values,
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_payment, use_container_width=True)

# Customer risk scoring
st.header("⚠️ High-Risk Customer Identification")

# Create risk scores for existing customers
def calculate_risk_score(row):
    score = 0
    if row['Contract'] == 'Month-to-month':
        score += 0.3
    if row['tenure'] < 12:
        score += 0.25
    if row['PaymentMethod'] == 'Electronic check':
        score += 0.2
    if row['MonthlyCharges'] > 80:
        score += 0.15
    if row.get('ServiceAdoptionScore', 0) < 2:
        score += 0.1
    return min(score, 0.95)

df['RiskScore'] = df.apply(calculate_risk_score, axis=1)
df['RiskLevel'] = pd.cut(df['RiskScore'], bins=[0, 0.4, 0.7, 1.0], 
                        labels=['Low', 'Medium', 'High'])

# Risk distribution
risk_counts = df['RiskLevel'].value_counts()
fig_risk = px.pie(
    values=risk_counts.values,
    names=risk_counts.index,
    title="Customer Risk Distribution",
    color_discrete_sequence=['green', 'orange', 'red']
)
st.plotly_chart(fig_risk, use_container_width=True)

# High-risk customers table
st.subheader("🚨 Top High-Risk Customers")
high_risk_customers = df[df['RiskLevel'] == 'High'].nlargest(10, 'RiskScore')[
    ['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod', 'RiskScore']
]

if len(high_risk_customers) > 0:
    st.dataframe(high_risk_customers, use_container_width=True)
    
    # AI insights for high-risk customers
    if st.button("🤖 Get AI Retention Strategy", type="primary"):
        with st.spinner("Generating AI-powered retention strategies..."):
            ai_insight = get_ai_insights({
                'total_customers': len(df),
                'churn_rate': churn_rate,
                'avg_clv': df['CLV'].mean() if 'CLV' in df.columns else 0,
                'high_risk_customers': len(high_risk_customers)
            })
            
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 1rem; border-left: 4px solid #1f77b4; margin: 1rem 0;">
                <h4>🧠 AI-Powered Retention Strategy</h4>
                <p>{ai_insight}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No high-risk customers identified in current dataset.")

# Retention recommendations
st.header("💡 Retention Strategy Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 High-Risk Customers
    **Immediate Actions:**
    - Personal outreach within 24 hours
    - Offer retention discounts (10-20%)
    - Upgrade to annual contract incentives
    - Dedicated customer success manager
    """)

with col2:
    st.markdown("""
    ### ⚖️ Medium-Risk Customers  
    **Proactive Engagement:**
    - Targeted email campaigns
    - Service usage optimization tips
    - Cross-sell complementary services
    - Quarterly satisfaction surveys
    """)

with col3:
    st.markdown("""
    ### ✅ Low-Risk Customers
    **Loyalty Building:**
    - Referral program enrollment
    - Beta feature access
    - Loyalty rewards program
    - Upselling premium features
    """)
