import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple, List
import streamlit as st

class CRMPredictor:
    """Main CRM prediction class for churn and segmentation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.config = {}
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all model artifacts"""
        try:
            # Load models
            self.models['churn'] = joblib.load('models/churn_prediction_model.pkl')
            self.models['segment'] = joblib.load('models/customer_segmentation_model.pkl')
            
            # Load scalers
            self.scalers['feature'] = joblib.load('models/feature_scaler.pkl')
            self.scalers['segment'] = joblib.load('models/segment_scaler.pkl')
            
            # Load encoders
            self.encoders = joblib.load('models/label_encoders.pkl')
            
            # Load config
            with open('models/deployment_config.json', 'r') as f:
                self.config = json.load(f)
                
        except Exception as e:
            st.warning(f"Some model artifacts not found: {str(e)}")
            # Initialize empty structures for fallback
            self.models = {}
            self.scalers = {}
            self.encoders = {}
            self.config = {}

def preprocess_input_data(customer_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess customer input data for model prediction
    
    Args:
        customer_data: Dictionary containing customer information
        
    Returns:
        Preprocessed feature array
    """
    try:
        # Feature engineering based on notebook
        features = {}
        
        # Basic features
        features['SeniorCitizen'] = 1 if customer_data.get('senior_citizen') == 'Yes' else 0
        features['tenure'] = customer_data.get('tenure', 0)
        features['MonthlyCharges'] = customer_data.get('monthly_charges', 0)
        features['TotalCharges'] = customer_data.get('total_charges', 0)
        
        # Calculate CLV
        features['CLV'] = features['tenure'] * features['MonthlyCharges']
        
        # Calculate average charges per tenure
        features['AvgChargesPerTenure'] = features['TotalCharges'] / (features['tenure'] + 1)
        
        # Service adoption score
        service_cols = ['phone_service', 'online_security', 'online_backup', 
                       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
        
        service_adoption = sum(1 for col in service_cols 
                             if customer_data.get(col) == 'Yes')
        features['ServiceAdoptionScore'] = service_adoption
        
        # Digital engagement
        features['DigitalEngagement'] = service_adoption / len(service_cols)
        
        # Encode categorical variables
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1},
            'partner': {'No': 0, 'Yes': 1},
            'dependents': {'No': 0, 'Yes': 1},
            'phone_service': {'No': 0, 'Yes': 1},
            'internet_service': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'paperless_billing': {'No': 0, 'Yes': 1},
            'payment_method': {
                'Electronic check': 0, 
                'Mailed check': 1, 
                'Bank transfer (automatic)': 2, 
                'Credit card (automatic)': 3
            }
        }
        
        # Add encoded categorical features
        for col, mapping in categorical_mappings.items():
            encoded_value = mapping.get(customer_data.get(col, 'No'), 0)
            features[f'{col}_encoded'] = encoded_value
        
        # Create feature array in correct order
        feature_order = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'ServiceAdoptionScore', 'CLV', 'AvgChargesPerTenure', 'DigitalEngagement',
            'gender_encoded', 'partner_encoded', 'dependents_encoded',
            'phone_service_encoded', 'internet_service_encoded',
            'contract_encoded', 'paperless_billing_encoded', 'payment_method_encoded'
        ]
        
        feature_array = np.array([features.get(col, 0) for col in feature_order]).reshape(1, -1)
        
        return feature_array
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return np.zeros((1, 16))  # Return default array

def predict_churn(customer_data: Dict[str, Any], predictor: CRMPredictor = None) -> Tuple[float, str]:
    """
    Predict customer churn probability
    
    Args:
        customer_data: Customer information dictionary
        predictor: CRMPredictor instance
        
    Returns:
        Tuple of (churn_probability, risk_level)
    """
    try:
        if predictor is None:
            predictor = CRMPredictor()
        
        # Preprocess input
        X = preprocess_input_data(customer_data)
        
        # Scale features if scaler available
        if 'feature' in predictor.scalers and predictor.scalers['feature']:
            X_scaled = predictor.scalers['feature'].transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        if 'churn' in predictor.models and predictor.models['churn']:
            churn_prob = predictor.models['churn'].predict_proba(X_scaled)[0, 1]
        else:
            # Fallback heuristic
            churn_prob = calculate_heuristic_churn_risk(customer_data)
        
        # Determine risk level
        if churn_prob > 0.7:
            risk_level = "HIGH"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return churn_prob, risk_level
        
    except Exception as e:
        st.error(f"Error in churn prediction: {str(e)}")
        return 0.5, "MEDIUM"

def calculate_heuristic_churn_risk(customer_data: Dict[str, Any]) -> float:
    """Calculate churn risk using business heuristics"""
    risk_score = 0.0
    
    # Contract type risk
    if customer_data.get('contract') == 'Month-to-month':
        risk_score += 0.3
    
    # Tenure risk
    if customer_data.get('tenure', 0) < 12:
        risk_score += 0.25
    
    # Payment method risk
    if customer_data.get('payment_method') == 'Electronic check':
        risk_score += 0.2
    
    # Monthly charges risk
    if customer_data.get('monthly_charges', 0) > 80:
        risk_score += 0.15
    
    # Service adoption risk
    service_cols = ['phone_service', 'online_security', 'online_backup', 
                   'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    service_count = sum(1 for col in service_cols if customer_data.get(col) == 'Yes')
    if service_count < 2:
        risk_score += 0.1
    
    return min(risk_score, 0.95)

def predict_customer_segment(customer_data: Dict[str, Any], predictor: CRMPredictor = None) -> Tuple[int, str]:
    """
    Predict customer segment
    
    Args:
        customer_data: Customer information dictionary
        predictor: CRMPredictor instance
        
    Returns:
        Tuple of (segment_id, segment_name)
    """
    try:
        if predictor is None:
            predictor = CRMPredictor()
        
        # Prepare segmentation features
        segmentation_features = [
            customer_data.get('tenure', 0),
            customer_data.get('monthly_charges', 0),
            customer_data.get('total_charges', 0),
            calculate_service_adoption_score(customer_data),
            calculate_digital_engagement(customer_data)
        ]
        
        X_segment = np.array(segmentation_features).reshape(1, -1)
        
        # Scale features
        if 'segment' in predictor.scalers and predictor.scalers['segment']:
            X_scaled = predictor.scalers['segment'].transform(X_segment)
        else:
            X_scaled = X_segment
        
        # Predict segment
        if 'segment' in predictor.models and predictor.models['segment']:
            segment_id = predictor.models['segment'].predict(X_scaled)[0]
        else:
            # Fallback heuristic segmentation
            segment_id = calculate_heuristic_segment(customer_data)
        
        # Map segment ID to name
        segment_names = {
            0: 'Price-Sensitive Basic Users',
            1: 'High-Value Long-Term Customers',
            2: 'New Digital Adopters',
            3: 'Premium Service Users'
        }
        
        segment_name = segment_names.get(segment_id, 'Unknown Segment')
        
        return segment_id, segment_name
        
    except Exception as e:
        st.error(f"Error in segment prediction: {str(e)}")
        return 0, 'Price-Sensitive Basic Users'

def calculate_service_adoption_score(customer_data: Dict[str, Any]) -> int:
    """Calculate service adoption score"""
    service_cols = ['phone_service', 'online_security', 'online_backup', 
                   'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    return sum(1 for col in service_cols if customer_data.get(col) == 'Yes')

def calculate_digital_engagement(customer_data: Dict[str, Any]) -> float:
    """Calculate digital engagement score"""
    service_count = calculate_service_adoption_score(customer_data)
    return service_count / 7  # Total possible services

def calculate_heuristic_segment(customer_data: Dict[str, Any]) -> int:
    """Calculate segment using business heuristics"""
    tenure = customer_data.get('tenure', 0)
    monthly_charges = customer_data.get('monthly_charges', 0)
    service_adoption = calculate_service_adoption_score(customer_data)
    
    # Segment 0: Price-Sensitive Basic Users
    if monthly_charges < 50 and service_adoption <= 2:
        return 0
    
    # Segment 1: High-Value Long-Term Customers
    elif tenure > 24 and monthly_charges > 60:
        return 1
    
    # Segment 2: New Digital Adopters
    elif tenure < 12 and service_adoption >= 3:
        return 2
    
    # Segment 3: Premium Service Users
    elif monthly_charges > 80 and service_adoption >= 4:
        return 3
    
    else:
        return 0  # Default to Price-Sensitive

def calculate_clv(customer_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate Customer Lifetime Value metrics
    
    Returns:
        Dictionary with CLV calculations
    """
    tenure = customer_data.get('tenure', 0)
    monthly_charges = customer_data.get('monthly_charges', 0)
    total_charges = customer_data.get('total_charges', 0)
    
    # Current CLV
    current_clv = tenure * monthly_charges
    
    # Predicted CLV (assuming average customer lifecycle)
    avg_customer_lifetime = 36  # months
    predicted_clv = monthly_charges * avg_customer_lifetime
    
    # CLV per tenure month
    clv_per_month = total_charges / (tenure + 1) if tenure > 0 else monthly_charges
    
    return {
        'current_clv': current_clv,
        'predicted_clv': predicted_clv,
        'clv_per_month': clv_per_month,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges
    }

def calculate_risk_score(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive customer risk score
    
    Returns:
        Dictionary with risk analysis
    """
    # Get churn probability
    churn_prob, risk_level = predict_churn(customer_data)
    
    # Calculate individual risk factors
    risk_factors = {
        'contract_risk': 0.3 if customer_data.get('contract') == 'Month-to-month' else 0.0,
        'tenure_risk': 0.25 if customer_data.get('tenure', 0) < 12 else 0.0,
        'payment_risk': 0.2 if customer_data.get('payment_method') == 'Electronic check' else 0.0,
        'price_risk': 0.15 if customer_data.get('monthly_charges', 0) > 80 else 0.0,
        'service_risk': 0.1 if calculate_service_adoption_score(customer_data) < 2 else 0.0
    }
    
    # Risk factor descriptions
    risk_descriptions = {
        'contract_risk': 'Month-to-month contract increases mobility',
        'tenure_risk': 'New customers have higher churn risk',
        'payment_risk': 'Electronic check payment method shows higher churn',
        'price_risk': 'High monthly charges may cause price sensitivity',
        'service_risk': 'Low service adoption indicates weak engagement'
    }
    
    # Active risk factors
    active_risks = {k: v for k, v in risk_factors.items() if v > 0}
    
    return {
        'overall_risk_score': churn_prob,
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'active_risks': active_risks,
        'risk_descriptions': risk_descriptions,
        'recommendations': generate_risk_recommendations(active_risks, risk_level)
    }

def generate_risk_recommendations(active_risks: Dict[str, float], risk_level: str) -> List[str]:
    """Generate recommendations based on risk factors"""
    recommendations = []
    
    if 'contract_risk' in active_risks:
        recommendations.append("Offer incentives for annual contract upgrade")
    
    if 'tenure_risk' in active_risks:
        recommendations.append("Implement new customer onboarding program")
    
    if 'payment_risk' in active_risks:
        recommendations.append("Encourage automatic payment setup")
    
    if 'price_risk' in active_risks:
        recommendations.append("Review pricing strategy and value proposition")
    
    if 'service_risk' in active_risks:
        recommendations.append("Promote additional service adoption")
    
    # General recommendations based on risk level
    if risk_level == "HIGH":
        recommendations.extend([
            "Immediate personal outreach required",
            "Offer retention discount or incentive",
            "Assign dedicated account manager"
        ])
    elif risk_level == "MEDIUM":
        recommendations.extend([
            "Proactive engagement campaign",
            "Monitor usage patterns closely",
            "Targeted service recommendations"
        ])
    else:
        recommendations.extend([
            "Focus on upselling opportunities",
            "Maintain regular satisfaction check-ins",
            "Enroll in loyalty program"
        ])
    
    return recommendations
