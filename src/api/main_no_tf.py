"""
FastAPI Backend for HR Performance Analytics (No TensorFlow)
Main API application with all endpoints
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ensemble_models_no_tf import EnsemblePerformancePredictor, ModelRegistry
from src.data.feature_engineering import FeatureEngineer
from src.bias.fairness_monitor import FairnessMonitor
from src.explainability.model_explainer import ModelExplainer
from src.privacy.data_protection import DataProtectionManager
from src.database.supabase_client import SupabaseClient
from src.utils.config import settings
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="HR Performance Analytics Pro API",
    description="Advanced AI-powered HR analytics with bias detection and explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
model_registry = ModelRegistry()
feature_engineer = FeatureEngineer()
fairness_monitor = FairnessMonitor()
explainer = ModelExplainer()
privacy_manager = DataProtectionManager()
db_client = SupabaseClient()

# Pydantic models
class EmployeeData(BaseModel):
    employee_id: str
    name: str
    department: str
    position: str
    salary: float
    age: int
    gender: str
    ethnicity: str
    education_level: str
    hire_date: str
    manager_id: Optional[str] = None

class PerformanceData(BaseModel):
    employee_id: str
    task_completion_rate: float = Field(..., ge=0, le=1)
    efficiency_score: float = Field(..., ge=0, le=1)
    quality_score: float = Field(..., ge=0, le=1)
    collaboration_score: float = Field(..., ge=0, le=1)
    innovation_score: float = Field(..., ge=0, le=1)
    leadership_score: float = Field(..., ge=0, le=1)
    communication_score: float = Field(..., ge=0, le=1)
    problem_solving_score: float = Field(..., ge=0, le=1)
    adaptability_score: float = Field(..., ge=0, le=1)
    goal_achievement_rate: float = Field(..., ge=0, le=1)

class PredictionRequest(BaseModel):
    employee_data: EmployeeData
    performance_history: Optional[List[PerformanceData]] = None
    explain: bool = False

class BatchPredictionRequest(BaseModel):
    employees: List[PredictionRequest]
    explain: bool = False

class BiasAuditRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    protected_attributes: List[str] = ["gender", "ethnicity", "age_group"]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting HR Performance Analytics API...")
    
    # Try to load existing model
    try:
        model = model_registry.get_model("default")
        if model and model.is_trained:
            logger.info("Loaded existing trained model")
        else:
            logger.info("No trained model found. Train a model using /models/train endpoint")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_available": model_registry.list_models()
    }

# Prediction endpoints
@app.post("/predict/single")
async def predict_single_employee(request: PredictionRequest):
    """Predict performance for a single employee"""
    try:
        model = model_registry.get_model("default")
        if not model or not model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model available")
        
        # Convert employee data to features
        employee_df = pd.DataFrame([request.employee_data.dict()])
        
        # Add performance history if available
        if request.performance_history:
            perf_df = pd.DataFrame([p.dict() for p in request.performance_history])
            # Merge with employee data (simplified)
            latest_perf = perf_df.iloc[-1] if len(perf_df) > 0 else {}
            for key, value in latest_perf.items():
                if key != 'employee_id':
                    employee_df[key] = value
        
        # Engineer features
        features_df = feature_engineer.engineer_all_features(employee_df)
        
        # Make prediction
        prediction_result = model.predict_single(features_df.iloc[0].to_dict())
        
        # Add explanation if requested
        if request.explain:
            explanation = explainer.explain_prediction(
                model, features_df.iloc[0].to_dict()
            )
            prediction_result['explanation'] = explanation
        
        return {
            "employee_id": request.employee_data.employee_id,
            "prediction": prediction_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch_employees(request: BatchPredictionRequest):
    """Predict performance for multiple employees"""
    try:
        model = model_registry.get_model("default")
        if not model or not model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model available")
        
        results = []
        for emp_request in request.employees:
            try:
                # Convert to single prediction
                single_request = PredictionRequest(
                    employee_data=emp_request.employee_data,
                    performance_history=emp_request.performance_history,
                    explain=request.explain
                )
                
                result = await predict_single_employee(single_request)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting for employee {emp_request.employee_data.employee_id}: {e}")
                results.append({
                    "employee_id": emp_request.employee_data.employee_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return {
            "predictions": results,
            "total_processed": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Bias and fairness endpoints
@app.post("/bias/audit")
async def audit_bias(request: BiasAuditRequest):
    """Perform comprehensive bias audit"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.dataset)
        
        # Perform bias analysis
        bias_results = fairness_monitor.analyze_bias(
            df, 
            protected_attributes=request.protected_attributes
        )
        
        return {
            "bias_analysis": bias_results,
            "timestamp": datetime.now().isoformat(),
            "compliance_status": "COMPLIANT" if bias_results.get('overall_fairness', 0) > 0.8 else "NON_COMPLIANT"
        }
        
    except Exception as e:
        logger.error(f"Bias audit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/fairness")
async def get_fairness_metrics():
    """Get current fairness metrics"""
    try:
        # Get recent predictions from database
        predictions = db_client.get_recent_predictions(limit=1000)
        
        if not predictions:
            return {"message": "No recent predictions available for fairness analysis"}
        
        # Analyze fairness
        df = pd.DataFrame(predictions)
        fairness_results = fairness_monitor.analyze_bias(df)
        
        return {
            "fairness_metrics": fairness_results,
            "data_points_analyzed": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fairness metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/models/status")
async def get_model_status():
    """Get status of all models"""
    try:
        models_info = []
        for model_name in model_registry.list_models():
            info = model_registry.get_model_info(model_name)
            models_info.append(info)
        
        return {
            "models": models_info,
            "total_models": len(models_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train a new model with latest data"""
    try:
        # Get training data from database
        employees = db_client.get_all_employees()
        performance = db_client.get_performance_data()
        
        if not employees or not performance:
            raise HTTPException(status_code=400, detail="Insufficient training data")
        
        # Start training in background
        background_tasks.add_task(train_model_background, employees, performance)
        
        return {
            "message": "Model training started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(employees: List[Dict], performance: List[Dict]):
    """Background task for model training"""
    try:
        logger.info("Starting background model training...")
        
        # Convert to DataFrames
        employees_df = pd.DataFrame(employees)
        performance_df = pd.DataFrame(performance)
        
        # Merge data
        latest_performance = performance_df.groupby('employee_id').last().reset_index()
        merged_data = employees_df.merge(latest_performance, on='employee_id', how='left')
        
        # Engineer features
        features_df = feature_engineer.engineer_all_features(merged_data)
        
        # Prepare training data
        performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
        y = features_df[performance_cols].mean(axis=1)
        
        exclude_cols = ['employee_id', 'name', 'email'] + performance_cols
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].select_dtypes(include=[np.number])
        
        # Train model
        model = EnsemblePerformancePredictor()
        results = model.train_ensemble(X, y)
        
        # Register model
        model_registry.register_model("default", model)
        
        logger.info(f"Model training completed. R² score: {results['ensemble_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")

# Privacy endpoints
@app.post("/privacy/audit")
async def privacy_audit(data: Dict[str, Any]):
    """Perform privacy compliance audit"""
    try:
        audit_results = privacy_manager.audit_data_privacy(data)
        
        return {
            "privacy_audit": audit_results,
            "timestamp": datetime.now().isoformat(),
            "gdpr_compliant": audit_results.get('gdpr_compliant', False)
        }
        
    except Exception as e:
        logger.error(f"Privacy audit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data management endpoints
@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload employee or performance data"""
    try:
        # Read uploaded file
        content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(content)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Privacy check
        privacy_results = privacy_manager.detect_pii(df)
        if privacy_results['pii_detected']:
            df = privacy_manager.mask_pii(df)
        
        # Determine data type and insert
        if 'employee_id' in df.columns and 'name' in df.columns:
            # Employee data
            for _, row in df.iterrows():
                await db_client.insert_employee(row.to_dict())
            data_type = "employees"
        elif 'employee_id' in df.columns and 'task_completion_rate' in df.columns:
            # Performance data
            for _, row in df.iterrows():
                await db_client.insert_performance_data(row.to_dict())
            data_type = "performance"
        else:
            raise HTTPException(status_code=400, detail="Unknown data format")
        
        return {
            "message": f"Successfully uploaded {len(df)} {data_type} records",
            "records_processed": len(df),
            "privacy_check": privacy_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/employees")
async def get_employees(limit: int = 100):
    """Get employee data"""
    try:
        employees = db_client.get_all_employees(limit=limit)
        return {
            "employees": employees,
            "count": len(employees),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get employees error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/performance")
async def get_performance_data(employee_id: Optional[str] = None, limit: int = 100):
    """Get performance data"""
    try:
        if employee_id:
            performance = db_client.get_employee_performance(employee_id)
        else:
            performance = db_client.get_performance_data(limit=limit)
        
        return {
            "performance_data": performance,
            "count": len(performance),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get performance data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Explanation endpoints
@app.post("/explain/prediction")
async def explain_prediction(request: PredictionRequest):
    """Get detailed explanation for a prediction"""
    try:
        model = model_registry.get_model("default")
        if not model or not model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model available")
        
        # Convert to features
        employee_df = pd.DataFrame([request.employee_data.dict()])
        features_df = feature_engineer.engineer_all_features(employee_df)
        
        # Get explanation
        explanation = explainer.explain_prediction(
            model, features_df.iloc[0].to_dict()
        )
        
        return {
            "employee_id": request.employee_data.employee_id,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/dashboard")
async def get_dashboard_data():
    """Get data for dashboard"""
    try:
        # Get recent data
        employees = db_client.get_all_employees(limit=1000)
        performance = db_client.get_performance_data(limit=5000)
        
        # Calculate basic analytics
        analytics = {
            "total_employees": len(employees),
            "total_performance_records": len(performance),
            "departments": {},
            "average_performance": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if employees:
            emp_df = pd.DataFrame(employees)
            analytics["departments"] = emp_df['department'].value_counts().to_dict()
        
        if performance:
            perf_df = pd.DataFrame(performance)
            performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
            for col in performance_cols:
                if col in perf_df.columns:
                    analytics["average_performance"][col] = float(perf_df[col].mean())
        
        return analytics
        
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
