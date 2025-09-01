# 🚀 HR Performance Analytics Pro - Setup Guide

## Quick Start (5 minutes)

### 1. Prerequisites
- Python 3.9+ installed
- Git installed
- Supabase account (optional - works with sample data)

### 2. Installation

```bash
# Navigate to project directory
cd "E:\New folder\hr-performance-analytics-pro"

# Install dependencies
pip install -r requirements.txt

# Setup environment (creates directories and .env file)
python scripts/setup_environment.py

# Initialize database (optional - creates sample data if no Supabase)
python scripts/init_database.py
```

### 3. Configuration

Edit `.env` file with your Supabase credentials (optional):
```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_KEY=your_supabase_service_role_key_here
```

### 4. Run Application

**Option A: Single Command (Recommended)**
```bash
python run_app.py
```

**Option B: Separate Services**
```bash
# Terminal 1 - API Backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Dashboard
streamlit run src/dashboard/main.py --server.port 8501
```

### 5. Access Application

- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🔧 Advanced Setup

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
```

### Model Training
```bash
# Train ensemble models
python src/models/train_ensemble.py

# Or use API endpoint
curl -X POST "http://localhost:8000/models/train" \
  -H "Content-Type: application/json" \
  -d '{"retrain": true, "hyperparameter_tuning": false}'
```

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

## 🎯 Features Overview

### Core Capabilities
- **🤖 Ensemble ML Models**: Random Forest, XGBoost, LightGBM, Neural Networks
- **📊 92+ Engineered Features**: Temporal, demographic, behavioral, contextual
- **⚖️ Bias Detection**: Real-time fairness monitoring with industry-standard metrics
- **🧠 Explainable AI**: SHAP and LIME explanations for all predictions
- **🔒 Privacy Protection**: PII masking, differential privacy, federated learning
- **📈 Professional Dashboard**: Modern Streamlit UI with interactive visualizations

### API Endpoints
- `POST /predict/single` - Single employee prediction
- `POST /predict/batch` - Batch predictions
- `POST /bias/audit` - Comprehensive bias audit
- `GET /models/status` - Model health and metadata
- `POST /models/train` - Train/retrain models
- `GET /metrics/fairness` - Real-time fairness metrics

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project root directory
cd "E:\New folder\hr-performance-analytics-pro"

# Install missing dependencies
pip install -r requirements.txt
```

**2. Database Connection Issues**
- Application works with sample data if no database configured
- Check `.env` file for correct Supabase credentials
- Verify Supabase project is active

**3. Model Training Fails**
```bash
# Generate sample data first
python -c "from src.data.sample_generator import SampleDataGenerator; SampleDataGenerator().generate_complete_dataset(1000)"
```

**4. Port Already in Use**
```bash
# Change ports in run_app.py or use different ports:
uvicorn src.api.main:app --port 8001
streamlit run src/dashboard/main.py --server.port 8502
```

### Performance Optimization
- For large datasets (>10K employees), consider using Docker deployment
- Enable hyperparameter tuning for better model performance
- Use batch predictions for multiple employees

## 📊 Sample Usage

### 1. Single Prediction via Dashboard
1. Navigate to "🔮 Predictions" tab
2. Fill in employee information
3. Click "🔮 Predict Performance"
4. View prediction, confidence, and explanations

### 2. API Usage
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict/single", json={
    "employee_data": {
        "employee_id": "EMP001",
        "name": "John Doe",
        "department": "Engineering",
        "position": "Senior Developer",
        "age": 32,
        "salary": 85000,
        # ... other fields
    },
    "explain": True,
    "include_bias_check": True
})

result = response.json()
print(f"Predicted Performance: {result['prediction']:.1f}")
print(f"Confidence: {result['confidence']:.1f}%")
```

### 3. Bias Audit
```python
# Run bias audit
response = requests.post("http://localhost:8000/bias/audit", json={
    "model_version": "ensemble_v1.0",
    "dataset_sample_size": 1000
})

audit = response.json()
print(f"Overall Fairness: {audit['overall_fairness']}")
```

## 🔐 Security & Privacy

### Data Protection Features
- **Automatic PII Detection**: Identifies and masks personal information
- **Differential Privacy**: Adds calibrated noise to protect individual privacy
- **Secure Hashing**: SHA-256 hashing for sensitive identifiers
- **GDPR Compliance**: Right to erasure and data portability
- **Federated Learning**: Train models without sharing raw data

### Best Practices
1. Always use `.env` file for credentials (never commit to version control)
2. Enable PII masking for production data
3. Regular bias audits (recommended: weekly)
4. Monitor fairness metrics in real-time
5. Use HTTPS in production deployments

## 📈 Monitoring & Maintenance

### Health Checks
- API Health: `GET /health`
- Model Status: `GET /models/status`
- Fairness Metrics: `GET /metrics/fairness`

### Logs
- Application logs: `logs/hr_analytics.log`
- Model training logs: Console output
- API access logs: Uvicorn logs

### Model Retraining
- Automatic drift detection triggers retraining recommendations
- Manual retraining via API or dashboard
- Model versioning and rollback capabilities

## 🆘 Support

For issues and questions:
1. Check this setup guide
2. Review application logs in `logs/` directory
3. Test with sample data first
4. Verify all dependencies are installed
5. Check API health endpoint

## 🎉 Success!

If you see:
- ✅ Dashboard at http://localhost:8501
- ✅ API docs at http://localhost:8000/docs
- ✅ Health check returns "healthy"

Your HR Performance Analytics Pro system is ready! 🚀
