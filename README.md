🏢 CRM Analytics Platform
[
[
[
[

📋 Overview
CRM Analytics Platform adalah aplikasi web berbasis Streamlit yang menyediakan analisis customer relationship management (CRM) yang komprehensif. Platform ini menggunakan machine learning untuk prediksi churn, segmentasi pelanggan, dan analisis customer lifetime value (CLV), serta terintegrasi dengan Azure OpenAI untuk insights berbasis AI.

🎯 Key Features
🎯 Customer Segmentation: Analisis demografis, behavioral, dan value-based segmentation

📊 Churn Prediction: Model ML untuk prediksi risiko churn dengan scoring system

📈 Real-time Dashboard: Monitoring KPI dan metrics secara real-time

💰 CLV Analysis: Analisis dan prediksi Customer Lifetime Value

🎪 Campaign Analysis: Optimasi marketing campaign dan product analysis

🤖 AI-Powered Insights: Rekomendasi strategis menggunakan Azure OpenAI

🏗️ Architecture
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Python Backend  │────│  Azure OpenAI   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       
         │              ┌────────────────┐               
         └──────────────│  ML Models     │               
                        │  - Churn       │               
                        │  - Segmentation│               
                        │  - CLV         │               
                        └────────────────┘               
🚀 Quick Start
Prerequisites
Python 3.11+

Azure CLI

Docker (optional)

Azure subscription dengan OpenAI service

Local Development
Clone repository

git clone https://github.com/your-username/crm-analytics-azure.git
cd crm-analytics-azure
Install dependencies

pip install -r requirements.txt
Configure secrets

bash
mkdir .streamlit
cat > .streamlit/secrets.toml << EOF
AZURE_OPENAI_ENDPOINT = "https://your-openai-service.openai.azure.com/"
AZURE_OPENAI_KEY = "your-openai-api-key"
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo"
EOF
Run application

bash
streamlit run app.py
☁️ Azure Deployment
Using Azure Container Apps
Create resource group

bash
az group create --name crm-analytics-rg --location eastus
Create Azure Container Registry

bash
az acr create --resource-group crm-analytics-rg --name crmanalyticacr --sku Basic --admin-enabled true
Create Container Apps environment

bash
az containerapp env create \
  --name crm-analytics-env \
  --resource-group crm-analytics-rg \
  --location eastus
Build and deploy

bash
# Build dengan unique tag
TIMESTAMP=$(date +%Y%m%d%H%M%S)
az acr build --registry crmanalyticacr --image crm-analytic:$TIMESTAMP .

# Deploy ke Container Apps
az containerapp create \
  --name crm-analytic-platform \
  --resource-group crm-analytics-rg \
  --environment crm-analytics-env \
  --image crmanalyticacr.azurecr.io/crm-analytic:$TIMESTAMP \
  --target-port 8501 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi
Get application URL

bash
az containerapp show \
  --name crm-analytic-platform \
  --resource-group crm-analytics-rg \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
Using GitHub Codespaces
Open in Codespaces dari GitHub repository

Install Azure CLI (sudah tersedia di devcontainer)

Login ke Azure

bash
az login --use-device-code
Follow deployment steps di atas

📊 Data Requirements
Platform ini menggunakan dataset telco customer churn dengan struktur:

text
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, 
PhoneService, MultipleLines, InternetService, OnlineSecurity, 
OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
MonthlyCharges, TotalCharges, Churn
Sample Data Format
text
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,DSL,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,DSL,56.95,1889.5,No
🤖 Machine Learning Models
Churn Prediction
Algorithm: Weighted Random Forest, Gradient Boosting, Logistic Regression

Features: 15 engineered features including CLV, service adoption score

Performance: AUC Score > 0.85, F1 Score > 0.75

Customer Segmentation
Algorithm: K-Means Clustering

Segments: 4 customer segments (Price-Sensitive, High-Value, Digital Adopters, Premium)

Features: Tenure, charges, service adoption, digital engagement

CLV Prediction
Algorithm: Polynomial Regression

Features: Tenure, monthly charges, service adoption

Metrics: R² Score, MAPE, MAE

📱 Application Structure
text
crm-analytics-azure/
├── app.py                          # Main Streamlit application
├── pages/                          # Multi-page structure
│   ├── 1_🎯_Customer_Segmentation.py
│   ├── 2_📊_Churn_Prediction.py
│   ├── 3_📈_Dashboard.py
│   ├── 4_💰_CLV_Analysis.py
│   └── 5_🎪_Campaign_Analysis.py
├── utils/                          # Utility modules
│   ├── data_loader.py             # Data loading functions
│   ├── model_utils.py             # ML model utilities
│   └── azure_openai.py            # Azure OpenAI integration
├── models/                         # Pre-trained models
│   ├── churn_prediction_model.pkl
│   ├── customer_segmentation_model.pkl
│   └── *.pkl                      # Other model artifacts
├── data/                          # Dataset
│   └── churn.csv
├── Dockerfile                     # Container configuration
├── requirements.txt               # Python dependencies
└── .streamlit/                    # Streamlit configuration
    ├── config.toml
    └── secrets.toml
🔧 Configuration
Environment Variables
bash
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo
Streamlit Configuration
text
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
📈 Performance & Monitoring
Key Metrics
Response Time: < 2 seconds untuk predictions

Throughput: 100+ concurrent users

Availability: 99.9% uptime target

Model Accuracy: AUC > 0.85 untuk churn prediction

Monitoring
Azure Monitor: Application insights dan logging

Container Health: Liveness dan readiness probes

Custom Metrics: Business KPIs tracking

🛠️ Development
Code Quality
Linting: flake8, black

Type Hints: mypy

Testing: pytest

Documentation: Docstrings dan comments

CI/CD Pipeline
text
# .github/workflows/azure-deploy.yml
name: Deploy to Azure Container Apps
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Azure
        uses: azure/container-apps-deploy-action@v1
📚 API Documentation
Prediction Endpoints
python
# Churn prediction
predict_churn(customer_data: Dict) -> Tuple[float, str]

# Customer segmentation  
predict_customer_segment(customer_data: Dict) -> Tuple[int, str]

# CLV calculation
calculate_clv(customer_data: Dict) -> Dict[str, float]
🔒 Security
Authentication: Azure AD integration ready

Data Privacy: No PII stored in logs

API Security: Rate limiting dan input validation

Container Security: Non-root user, minimal base image

🤝 Contributing
Fork repository

Create feature branch: git checkout -b feature/amazing-feature

Commit changes: git commit -m 'Add amazing feature'

Push to branch: git push origin feature/amazing-feature

Open Pull Request

Development Guidelines
Follow PEP 8 style guide

Add unit tests untuk new features

Update documentation sesuai changes

Ensure backward compatibility

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Azure OpenAI untuk AI-powered insights

Streamlit untuk rapid web app development

Scikit-learn untuk machine learning capabilities

Plotly untuk interactive visualizations

📞 Support
Issues: GitHub Issues

Documentation: Wiki

Email: your-email@domain.com

🚀 Roadmap
 Real-time streaming data integration

 Advanced ML models (Deep Learning, AutoML)

 Multi-tenant architecture

 Mobile app companion

 API Gateway integration

 Advanced security features

Built with ❤️ using Azure Container Apps, Streamlit, and Machine Learning

Last updated: January 2025
