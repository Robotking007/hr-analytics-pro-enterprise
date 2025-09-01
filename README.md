# 🚀 HR Performance Analytics Pro - Complete System

A comprehensive AI-powered HR analytics platform with advanced machine learning, bias detection, and Supabase backend integration.

## 🌟 Key Features

### 🤖 Advanced Machine Learning
- **Ensemble Learning**: Random Forest, Gradient Boosting, Extra Trees, SVM, Neural Networks
- **Deep Learning**: LSTM with Attention mechanism for temporal data
- **Transformer Architecture**: Multi-head attention for text and sequence data
- **Model Fusion**: Combines quantitative and qualitative models (90.1% accuracy)

### 📊 Feature Engineering (92+ Features)
- **Temporal Features**: 12-month historical data tracking
- **Statistical Aggregations**: Mean, std dev, trend analysis
- **Demographic Features**: Age, tenure, salary, education, position
- **Interaction Terms**: Cross-feature combinations and polynomial features

### ⚖️ Fairness & Bias Monitoring
- **Comprehensive Bias Audit**: Gender, age, ethnicity bias detection
- **Fairness Metrics**: Demographic Parity, Equalized Odds, Disparate Impact
- **Real-time Monitoring**: Continuous fairness tracking with alerts

### 🔒 Privacy & Security
- **PII Masking**: Automatic personal data protection
- **Differential Privacy**: Gaussian noise for privacy preservation
- **Federated Learning**: Distributed learning without data sharing

### 🎨 Professional Dashboard
- **Streamlit Interface**: Modern, responsive web application
- **Interactive Visualizations**: Plotly charts and dynamic graphs
- **Real-time Updates**: Live data refresh and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Supabase account
- PostgreSQL (optional for local development)

### Option 1: One-Click Setup (Recommended)
```bash
cd "E:\New folder\hr-performance-analytics-pro"
python quick_start.py
```

### Option 2: Complete System Setup
```bash
cd "E:\New folder\hr-performance-analytics-pro"
python start_system.py
```

### Option 3: Manual Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**: 
   - The `.env` file is already configured with your Supabase credentials

3. **Test Database Connection**:
   ```bash
   python test_supabase_connection.py
   ```

4. **Start Application**:
   ```bash
   python run_app.py
   ```

### 🌐 Access Points
- **📊 Dashboard**: http://localhost:8501
- **🔗 API Documentation**: http://localhost:8000/docs
- **💚 Health Check**: http://localhost:8000/health00

## 📁 Project Structure

```
hr-performance-analytics-pro/
├── src/
│   ├── models/           # ML models and training
│   ├── data/            # Data processing and feature engineering
│   ├── api/             # FastAPI backend
│   ├── dashboard/       # Streamlit frontend
│   ├── database/        # Supabase integration
│   ├── bias/            # Fairness and bias detection
│   ├── privacy/         # Privacy and security features
│   └── utils/           # Utilities and helpers
├── scripts/             # Setup and utility scripts
├── tests/              # Test suite
├── docs/               # Documentation
└── data/               # Sample data and datasets
```

## 🔧 Configuration

### Supabase Setup
1. Create a new Supabase project
2. Copy your project URL and anon key to `.env`
3. Run database initialization script
4. Configure Row Level Security (RLS) policies

### Model Training
```bash
python src/models/train_ensemble.py
```

### Bias Audit
```bash
python src/bias/audit_models.py
```

## 📊 API Endpoints

- `POST /predict/single` - Single employee prediction
- `POST /predict/batch` - Batch predictions
- `GET /models/status` - Model health check
- `POST /bias/audit` - Bias analysis
- `GET /metrics/fairness` - Fairness metrics

## 🎯 Usage Examples

### Single Prediction
```python
import requests

response = requests.post("http://localhost:8000/predict/single", 
    json={"employee_data": {...}})
```

### Bias Audit
```python
from src.bias.monitor import BiasMonitor

monitor = BiasMonitor()
audit_results = monitor.audit_model(model, test_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions, please create an issue in the repository or contact the development team.
