# CRM Analytics Platform

A modern, cloud-ready Customer Relationship Management (CRM) analytics application built with **Streamlit**, **scikit-learn**, and **Azure OpenAI**. This platform enables business users and analysts to perform customer segmentation, churn prediction, CLV analysis, campaign insights, and real-time dashboarding, all deployable to Azure Container Apps.

---

## 🚀 Features

- **Customer Segmentation**
  - Demographic, behavioral, and value-based segmentation
  - Predict customer segment from manual input or dataset

- **Churn Analysis & Prediction**
  - Individual churn risk scoring with ML model
  - Churn risk explanations and retention recommendations

- **Customer Lifetime Value (CLV)**
  - CLV calculation and predictive analytics
  - High-value customer identification

- **Campaign & Product Analysis**
  - Service adoption and upsell/cross-sell opportunity analysis
  - Campaign ROI and targeting simulation

- **Interactive Dashboard**
  - Real-time KPI monitoring and customer health scoring
  - Exportable analytics and AI-powered insights

- **Azure OpenAI Integration**
  - LLM-generated business insights and recommendations

---

## 🗂️ Project Structure

crm-analytics-azure/

├── app.py

├── Dockerfile
├── containerapp.yaml
├── requirements.txt
├── README.md
├── .devcontainer/
│ └── devcontainer.json
├── .github/
│ └── workflows/
│ └── azure-deploy.yml
├── .streamlit/
│ ├── config.toml
│ └── secrets.toml
├── models/
│ ├── churn_prediction_model.pkl
│ ├── customer_segmentation_model.pkl
│ ├── feature_scaler.pkl
│ ├── segment_scaler.pkl
│ ├── label_encoders.pkl
│ └── deployment_config.json
├── data/
│ └── churn.csv
├── pages/
│ ├── 1_🎯_Customer_Segmentation.py
│ ├── 2_📊_Churn_Prediction.py
│ ├── 3_📈_Dashboard.py
│ ├── 4_💰_CLV_Analysis.py
│ └── 5_🎪_Campaign_Analysis.py
├── utils/
│ ├── init.py
│ ├── data_loader.py
│ ├── model_utils.py
│ └── azure_openai.py
└── startup.sh

text

---

## ⚡ Quick Start (Local)

1. **Clone the repository**
git clone https://github.com/yourusername/crm-analytics-azure.git
cd crm-analytics-azure

text

2. **Install dependencies**
pip install -r requirements.txt

text

3. **Set up Streamlit secrets**
- Edit `.streamlit/secrets.toml` with your Azure OpenAI endpoint, key, and deployment name.

4. **Run the app locally**
streamlit run app.py

text

---

## 🐳 Docker Usage

1. **Build Docker image**
docker build -t crm-analytics .

text

2. **Run container**
docker run -p 8501:8501 --env-file .env crm-analytics

text

---

## ☁️ Azure Container Apps Deployment

### 1. Build & Push Image to Azure Container Registry (ACR)
az acr build --registry <acr-name> --image crm-analytics:latest .

text

### 2. Deploy to Azure Container Apps
az containerapp create
--name crm-analytics-platform
--resource-group <resource-group>
--environment <containerapp-env>
--image <acr-name>.azurecr.io/crm-analytics:latest
--target-port 8501
--ingress external
--cpu 1.0
--memory 2.0Gi

text

### 3. Update Container App (Redeploy)
az containerapp update
--name crm-analytics-platform
--resource-group <resource-group>
--image <acr-name>.azurecr.io/crm-analytics:latest

text

### 4. Get Public URL
az containerapp show
--name crm-analytics-platform
--resource-group <resource-group>
--query "properties.configuration.ingress.fqdn"
--output tsv

text

---

## 🔑 Azure OpenAI Integration

- Set your Azure OpenAI credentials in `.streamlit/secrets.toml`:
AZURE_OPENAI_ENDPOINT = "https://<your-openai-resource>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>"

text

---

## 🧑‍💻 Development in GitHub Codespaces

- Open the repo in Codespaces.
- Use the provided `.devcontainer/devcontainer.json` for a ready-to-use environment with Docker and Azure CLI.
- All ports and dependencies are pre-configured for rapid development and cloud deployment.

---

## 📝 Dataset

- Place your customer data as `data/churn.csv` (Telco churn format).
- Example columns: `customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, ...`

---

## 🛠️ Model Artifacts

- Pre-trained models and scalers are stored in `models/`:
- `churn_prediction_model.pkl`
- `customer_segmentation_model.pkl`
- `feature_scaler.pkl`
- `segment_scaler.pkl`
- `label_encoders.pkl`
- `deployment_config.json`

---

## 🧩 Customization

- **Add new pages**: Place Streamlit files in `/pages/`
- **Add new models**: Retrain and save to `/models/`
- **Extend data pipeline**: Edit `utils/model_utils.py` and `utils/data_loader.py`

---

## 🏆 Best Practices

- Always use the same feature order and preprocessing at inference as during model training.
- Use version tags for Docker images to avoid cache issues.
- Use managed identity or ACR credentials for secure container registry access.
- Keep your Azure OpenAI deployment name consistent and verify in Azure Portal.

---

## 📄 License

MIT License

---

## 🙋 FAQ

**Q: Why do I get "X has 16 features, but StandardScaler is expecting 15 features"?**  
A: Ensure your feature engineering and order matches exactly between training and inference. Use the provided preprocessing functions.

**Q: How do I update the app after code/model changes?**  
A: Rebuild the Docker image, push to ACR, and run `az containerapp update ...` as shown above.

**Q: Why does AI insights fail with deployment error?**  
A: Check that your Azure OpenAI deployment name matches exactly (case-sensitive) and is available in your resource/region.

---

## 👨‍💻 Authors

- CRM Analytics Team

---

## 🌐 References

- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
