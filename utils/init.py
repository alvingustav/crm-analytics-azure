"""
CRM Analytics Utilities Package
"""

__version__ = "1.0.0"
__author__ = "CRM Analytics Team"

# Import commonly used functions
from .data_loader import load_data, load_model_artifacts
from .model_utils import (
    preprocess_input_data, 
    predict_churn, 
    predict_customer_segment,
    calculate_clv,
    calculate_risk_score
)
from .azure_openai import get_ai_insights, analyze_customer_segment

__all__ = [
    'load_data',
    'load_model_artifacts', 
    'preprocess_input_data',
    'predict_churn',
    'predict_customer_segment',
    'calculate_clv',
    'calculate_risk_score',
    'get_ai_insights',
    'analyze_customer_segment'
]
