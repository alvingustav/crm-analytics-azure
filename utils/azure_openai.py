import openai
import streamlit as st
import os
from typing import Dict, Any

def get_ai_insights(data: Dict[str, Any]) -> str:
    """Generate AI insights using Azure OpenAI"""
    try:
        # Configure Azure OpenAI
        openai.api_type = "azure"
        openai.api_base = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
        openai.api_version = "2023-12-01-preview"
        openai.api_key = st.secrets.get("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"))
        
        prompt = f"""
        As a CRM Analytics expert, analyze the following customer data and provide actionable business insights:
        
        - Total Customers: {data['total_customers']:,}
        - Churn Rate: {data['churn_rate']:.1%}
        - Average CLV: ${data['avg_clv']:.2f}
        - High-Risk Customers: {data['high_risk_customers']}
        
        Provide 3 key insights and 2 specific recommendations for improving customer retention and revenue growth.
        Keep the response concise and business-focused.
        """
        
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",  # Replace with your deployment name
            messages=[
                {"role": "system", "content": "You are a senior CRM analyst with expertise in customer retention and revenue optimization."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI insights temporarily unavailable. Error: {str(e)}"

def analyze_customer_segment(segment_data: Dict[str, Any]) -> str:
    """Generate segment-specific insights"""
    try:
        openai.api_type = "azure"
        openai.api_base = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
        openai.api_version = "2023-12-01-preview"
        openai.api_key = st.secrets.get("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"))
        
        prompt = f"""
        Analyze this customer segment and provide targeted marketing strategies:
        
        Segment: {segment_data['segment_name']}
        - Average CLV: ${segment_data['avg_clv']:.2f}
        - Churn Rate: {segment_data['churn_rate']:.1%}
        - Customer Count: {segment_data['customer_count']}
        - Avg Tenure: {segment_data['avg_tenure']:.1f} months
        
        Provide 2 specific marketing strategies and 1 retention tactic for this segment.
        """
        
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing strategist specializing in customer segmentation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Segment analysis temporarily unavailable. Error: {str(e)}"
