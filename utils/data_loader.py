import pandas as pd
import joblib
import json
import streamlit as st
import os

@st.cache_data
def load_data(file_path='data/churn.csv'):
    """Load and preprocess the churn dataset"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Feature engineering
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['AvgChargesPerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Service adoption score
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
        
        df['ServiceAdoptionScore'] = df[service_cols].sum(axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        artifacts = {
            'churn_model': joblib.load('models/churn_prediction_model.pkl'),
            'segment_model': joblib.load('models/customer_segmentation_model.pkl'),
            'feature_scaler': joblib.load('models/feature_scaler.pkl'),
            'segment_scaler': joblib.load('models/segment_scaler.pkl'),
            'label_encoders': joblib.load('models/label_encoders.pkl')
        }
        
        # Load config
        with open('models/deployment_config.json', 'r') as f:
            artifacts['config'] = json.load(f)
            
        return artifacts
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None
