# CRM Analytics Platform

A modern, cloud-ready Customer Relationship Management (CRM) analytics application built with **Streamlit**, **scikit-learn**, and **Azure OpenAI**. This platform enables business users and analysts to perform customer segmentation, churn prediction, CLV analysis, campaign insights, and real-time dashboarding — all deployable to Azure Container Apps.

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

```
crm-analytics-azure/
├── app.py
├── Dockerfile
├── containerapp.yaml
├── requirements.txt
├── README.md
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── azure-deploy.yml
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── models/
│   ├── churn_prediction_model.pkl
│   ├── customer_segmentation_model.pkl
│   ├── feature_scaler.pkl
│   ├── segment_scaler.pkl
│   ├── label_encoders.pkl
│   └── deployment_config.json
├── data/
│   └── churn.csv
├── pages/
│   ├── 1_🎯_Customer_Segmentation.py
│   ├── 2_📊_Churn_Prediction.py
│   ├── 3_📈_Dashboard.py
│   ├── 4_💰_CLV_Analysis.py
│   └── 5_🎪_Campaign_Analysis.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_utils.py
│   └── azure_openai.py
└── startup.sh
```

---

## ⚡ Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crm-analytics-azure.git
cd crm-analytics-azure
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Streamlit secrets

Edit the file `.streamlit/secrets.toml` and add your Azure OpenAI credentials:

```toml
AZURE_OPENAI_ENDPOINT = "https://<your-openai-resource>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>"
```

### 4. Run the app locally

```bash
streamlit run app.py
```

---

## 🐳 Docker Usage

### 1. Build Docker image

```bash
docker build -t crm-analytics .
```

### 2. Run container

```bash
docker run -p 8501:8501 --env-file .env crm-analytics
```

---

## ☁️ Azure Container Apps Deployment

### 1. Build & Push Image to Azure Container Registry (ACR)

```bash
az acr build --registry <acr-name> --image crm-analytics:latest .
```

### 2. Deploy to Azure Container Apps

```bash
az containerapp create   --name crm-analytics-platform   --resource-group <resource-group>   --environment <containerapp-env>   --image <acr-name>.azurecr.io/crm-analytics:latest   --target-port 8501   --ingress external   --cpu 1.0   --memory 2.0Gi
```

### 3. Update Container App (Redeploy)

```bash
az containerapp update   --name crm-analytics-platform   --resource-group <resource-group>   --image <acr-name>.azurecr.io/crm-analytics:latest
```

### 4. Get Public URL

```bash
az containerapp show   --name crm-analytics-platform   --resource-group <resource-group>   --query "properties.configuration.ingress.fqdn"   --output tsv
```

---

## 🔑 Azure OpenAI Integration

Edit your `.streamlit/secrets.toml` file:

```toml
AZURE_OPENAI_ENDPOINT = "https://<your-openai-resource>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-azure-openai-key>"
AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>"
```

---

## 🧑‍💻 Development in GitHub Codespaces

- Open the repo in Codespaces.
- Use `.devcontainer/devcontainer.json` for preconfigured Docker + Azure CLI.
- All ports and dependencies are ready to go for local/cloud development.

---

## 📝 Dataset

- Place your dataset in `data/churn.csv` (Telco churn format).
- Example columns: `customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, ...`

---

## 🛠️ Model Artifacts

Stored in the `models/` folder:

- `churn_prediction_model.pkl`
- `customer_segmentation_model.pkl`
- `feature_scaler.pkl`
- `segment_scaler.pkl`
- `label_encoders.pkl`
- `deployment_config.json`

---

## 🧩 Customization

- **Add new pages**: Place new `.py` files in `/pages/`
- **Add new models**: Train and save to `/models/`
- **Extend pipeline**: Modify `utils/model_utils.py` and `utils/data_loader.py`

---

## 🏆 Best Practices

- Ensure consistent feature order & preprocessing during training and inference.
- Tag Docker images (`:v1`, `:latest`) to avoid cache/stale issues.
- Use managed identity or secure ACR credentials.
- Keep Azure OpenAI deployment names consistent and case-sensitive.

---

## 📄 License

MIT License

---

## 🙋 FAQ

**Q: How do I update the app after code/model changes?**  
A: Rebuild the Docker image, push to ACR, and redeploy using `az containerapp update ...`.

**Q: Why does AI insights fail with deployment error?**  
A: Ensure your Azure OpenAI deployment name is **exactly** correct (case-sensitive) and exists in the correct region.

---

## 👨‍💻 Authors

- CRM Analytics Team

---

## 🌐 References

- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
