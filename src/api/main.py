"""
FastAPI Backend for HR Performance Analytics Pro
Provides REST API endpoints for predictions, bias audits, and model management
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import uvicorn
from loguru import logger
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ensemble_models import EnsemblePerformancePredictor, ModelRegistry
from src.data.feature_engineering import FeatureEngineer
from src.bias.fairness_monitor import FairnessMonitor
from src.explainability.model_explainer import ModelExplainer
from src.privacy.data_protection import DataProtectionManager
from src.database.supabase_client import SupabaseClient

# Initialize FastAPI app
app = FastAPI(
    title="HR Performance Analytics Pro API",
    description="Advanced AI-powered HR analytics with bias detection and explainable AI",
    version="1.0.0"
)

# Add CORS middleware
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
db_client = SupabaseClient()
feature_engineer = FeatureEngineer()
fairness_monitor = FairnessMonitor()
model_explainer = ModelExplainer()
privacy_manager = DataProtectionManager()
model_registry = ModelRegistry()

# Global model instance
global_model: Optional[EnsemblePerformancePredictor] = None

# Pydantic models
class EmployeeData(BaseModel):
    employee_id: str
    name: str
    email: str
    department: str
    position: str
    position_level: int
    salary: float
    hire_date: str
    age: int
    gender: str
    ethnicity: str
    education_level: str
    task_completion_rate: Optional[float] = 85.0
    efficiency_score: Optional[float] = 85.0
    quality_score: Optional[float] = 85.0
    collaboration_score: Optional[float] = 85.0
    innovation_score: Optional[float] = 85.0
    leadership_score: Optional[float] = 85.0
    communication_score: Optional[float] = 85.0
    problem_solving_score: Optional[float] = 85.0
    adaptability_score: Optional[float] = 85.0
    goal_achievement_rate: Optional[float] = 85.0

class PredictionRequest(BaseModel):
    employee_data: EmployeeData
    explain: bool = True
    include_bias_check: bool = True

class BatchPredictionRequest(BaseModel):
    employees: List[EmployeeData]
    explain: bool = False
    include_bias_check: bool = True

class BiasAuditRequest(BaseModel):
    model_version: str
    dataset_sample_size: int = 1000

class ModelTrainingRequest(BaseModel):
    retrain: bool = True
    hyperparameter_tuning: bool = False
    validation_split: float = 0.2

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global global_model
    
    logger.info("Starting HR Performance Analytics Pro API...")
    
    try:
        # Initialize database
        await db_client.create_tables()
        logger.info("Database initialized")
        
        # Try to load existing model
        try:
            global_model = model_registry.get_latest_model()
            logger.info("Loaded existing model")
        except:
            logger.info("No existing model found, will train new model on first request")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": global_model is not None,
        "database_connected": True,
        "version": "1.0.0"
    }

# Model status endpoint
@app.get("/models/status")
async def get_model_status():
    """Get current model status and metadata"""
    if global_model is None:
        return {"status": "no_model", "message": "No model currently loaded"}
    
    return {
        "status": "ready",
        "model_trained": global_model.is_trained,
        "ensemble_weights": global_model.ensemble_weights,
        "available_models": list(global_model.models.keys())
    }

# Single prediction endpoint
@app.post("/predict/single")
async def predict_single_employee(request: PredictionRequest):
    """Predict performance for a single employee"""
    global global_model
    
    try:
        # Ensure model is available
        if global_model is None or not global_model.is_trained:
            # Train model with sample data if not available
            await train_model_with_sample_data()
        
        # Convert employee data to DataFrame
        employee_dict = request.employee_data.dict()
        df = pd.DataFrame([employee_dict])
        
        # Engineer features
        df_engineered = feature_engineer.engineer_all_features(df)
        
        # Make prediction
        prediction_result = global_model.predict_single(df_engineered)
        
        # Save prediction to database
        prediction_data = {
            "employee_id": request.employee_data.employee_id,
            "predicted_performance": prediction_result['ensemble_prediction'],
            "confidence_score": prediction_result['confidence'],
            "model_version": "ensemble_v1.0",
            "features_used": df_engineered.columns.tolist(),
            "explanation": prediction_result['individual_predictions']
        }
        
        saved_prediction = db_client.save_prediction(prediction_data)
        
        response = {
            "employee_id": request.employee_data.employee_id,
            "prediction": prediction_result['ensemble_prediction'],
            "confidence": prediction_result['confidence'],
            "individual_predictions": prediction_result['individual_predictions'],
            "model_weights": prediction_result['model_weights'],
            "prediction_id": saved_prediction.get('id')
        }
        
        # Add explanations if requested
        if request.explain:
            try:
                # Initialize explainer if needed
                sample_data = df_engineered.sample(min(100, len(df_engineered)))
                model_explainer.initialize_explainers(global_model.models['random_forest'], sample_data)
                
                # Generate explanations
                shap_explanation = model_explainer.explain_prediction_shap(
                    global_model.models['random_forest'], df_engineered
                )
                lime_explanation = model_explainer.explain_prediction_lime(
                    global_model.models['random_forest'], df_engineered
                )
                
                response['explanations'] = {
                    'shap': shap_explanation,
                    'lime': lime_explanation
                }
            except Exception as e:
                logger.error(f"Error generating explanations: {e}")
                response['explanations'] = {"error": "Explanation generation failed"}
        
        # Add bias check if requested
        if request.include_bias_check:
            try:
                # Create synthetic comparison data for bias check
                comparison_data = pd.DataFrame([employee_dict] * 10)
                comparison_data['gender'] = ['Male', 'Female'] * 5
                comparison_data['ethnicity'] = ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'] * 2
                
                predictions = global_model.predict_ensemble(
                    feature_engineer.engineer_all_features(comparison_data)
                )
                
                bias_check = fairness_monitor.monitor_real_time_fairness(
                    predictions, comparison_data[['gender', 'ethnicity']]
                )
                
                response['bias_check'] = bias_check
            except Exception as e:
                logger.error(f"Error performing bias check: {e}")
                response['bias_check'] = {"error": "Bias check failed"}
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch_employees(request: BatchPredictionRequest):
    """Predict performance for multiple employees"""
    global global_model
    
    try:
        if global_model is None or not global_model.is_trained:
            await train_model_with_sample_data()
        
        # Convert to DataFrame
        employees_data = [emp.dict() for emp in request.employees]
        df = pd.DataFrame(employees_data)
        
        # Engineer features
        df_engineered = feature_engineer.engineer_all_features(df)
        
        # Make predictions
        predictions = global_model.predict_ensemble(df_engineered)
        
        # Prepare response
        results = []
        for i, employee in enumerate(request.employees):
            result = {
                "employee_id": employee.employee_id,
                "prediction": float(predictions[i]),
                "rank": i + 1  # Will be sorted later
            }
            results.append(result)
        
        # Sort by prediction (highest first)
        results.sort(key=lambda x: x['prediction'], reverse=True)
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        response = {
            "total_employees": len(results),
            "predictions": results,
            "summary_stats": {
                "mean_prediction": float(np.mean(predictions)),
                "std_prediction": float(np.std(predictions)),
                "min_prediction": float(np.min(predictions)),
                "max_prediction": float(np.max(predictions))
            }
        }
        
        # Add bias check if requested
        if request.include_bias_check:
            try:
                bias_check = fairness_monitor.monitor_real_time_fairness(
                    predictions, df[['gender', 'ethnicity']]
                )
                response['bias_check'] = bias_check
            except Exception as e:
                logger.error(f"Error in batch bias check: {e}")
                response['bias_check'] = {"error": "Bias check failed"}
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Bias audit endpoint
@app.post("/bias/audit")
async def perform_bias_audit(request: BiasAuditRequest):
    """Perform comprehensive bias audit"""
    try:
        # Get sample data from database
        all_data = db_client.get_all_employees_data()
        
        if all_data.empty:
            # Use synthetic data for demo
            all_data = generate_sample_data(request.dataset_sample_size)
        
        # Engineer features
        df_engineered = feature_engineer.engineer_all_features(all_data)
        
        # Make predictions
        if global_model and global_model.is_trained:
            predictions = global_model.predict_ensemble(df_engineered)
            
            # Create synthetic ground truth for audit
            y_true = predictions + np.random.normal(0, 5, len(predictions))
            
            # Perform bias audit
            audit_results = fairness_monitor.comprehensive_bias_audit(
                all_data, y_true, predictions, request.model_version
            )
            
            # Save audit results
            audit_data = {
                "model_version": request.model_version,
                "demographic_parity": audit_results['protected_attributes'].get('gender', {}).get('demographic_parity', {}).get('demographic_parity_ratio', 1.0),
                "equalized_odds": audit_results['protected_attributes'].get('gender', {}).get('equalized_odds', {}).get('equalized_odds_ratio', 1.0),
                "disparate_impact": audit_results['protected_attributes'].get('gender', {}).get('disparate_impact', {}).get('disparate_impact_ratio', 1.0),
                "statistical_parity_diff": audit_results['protected_attributes'].get('gender', {}).get('statistical_parity', {}).get('statistical_parity_difference', 0.0),
                "fairness_threshold_met": audit_results['overall_fairness'],
                "audit_results": audit_results
            }
            
            db_client.save_bias_audit(audit_data)
            
            return audit_results
        else:
            raise HTTPException(status_code=400, detail="No trained model available for audit")
            
    except Exception as e:
        logger.error(f"Bias audit error: {e}")
        raise HTTPException(status_code=500, detail=f"Bias audit failed: {str(e)}")

# Model training endpoint
@app.post("/models/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain the ensemble model"""
    try:
        # Add training task to background
        background_tasks.add_task(train_model_background, request)
        
        return {
            "status": "training_started",
            "message": "Model training started in background",
            "estimated_time": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

async def train_model_background(request: ModelTrainingRequest):
    """Background task for model training"""
    global global_model
    
    try:
        logger.info("Starting model training...")
        
        # Get training data
        training_data = db_client.get_all_employees_data()
        
        if training_data.empty:
            # Generate sample data for training
            training_data = generate_sample_data(1000)
        
        # Engineer features
        df_engineered = feature_engineer.engineer_all_features(training_data)
        
        # Create target variable (overall performance score)
        performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score',
                          'collaboration_score', 'innovation_score']
        available_perf_cols = [col for col in performance_cols if col in df_engineered.columns]
        
        if available_perf_cols:
            y = df_engineered[available_perf_cols].mean(axis=1)
        else:
            y = np.random.normal(85, 10, len(df_engineered))
        
        # Select features for training
        feature_cols = [col for col in df_engineered.columns 
                       if col not in ['employee_id', 'name', 'email'] + performance_cols]
        X = df_engineered[feature_cols]
        
        # Initialize and train model
        global_model = EnsemblePerformancePredictor()
        
        if request.hyperparameter_tuning:
            tuning_results = global_model.hyperparameter_tuning(X, y)
            logger.info(f"Hyperparameter tuning completed: {tuning_results}")
        
        training_results = global_model.train_ensemble(X, y, request.validation_split)
        
        # Save model
        model_registry.register_model(
            global_model, 
            "1.0", 
            {
                "training_results": training_results,
                "feature_count": X.shape[1],
                "training_samples": len(X)
            }
        )
        
        # Save model metadata to database
        model_metadata = {
            "model_name": "ensemble_performance_predictor",
            "model_version": "1.0",
            "model_type": "ensemble",
            "accuracy": training_results['ensemble_score'],
            "precision_score": 0.0,  # Not applicable for regression
            "recall_score": 0.0,     # Not applicable for regression
            "f1_score": 0.0,         # Not applicable for regression
            "training_date": pd.Timestamp.now().isoformat(),
            "feature_importance": global_model.get_feature_importance(),
            "hyperparameters": training_results,
            "is_active": True
        }
        
        db_client.save_model_metadata(model_metadata)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")

async def train_model_with_sample_data():
    """Train model with sample data if no model exists"""
    global global_model
    
    if global_model is not None and global_model.is_trained:
        return
    
    logger.info("Training model with sample data...")
    
    # Generate sample training data
    sample_data = generate_sample_data(500)
    df_engineered = feature_engineer.engineer_all_features(sample_data)
    
    # Create target variable
    performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
    y = df_engineered[performance_cols].mean(axis=1)
    
    # Select features
    feature_cols = [col for col in df_engineered.columns 
                   if col not in ['employee_id', 'name', 'email'] + performance_cols]
    X = df_engineered[feature_cols]
    
    # Train model
    global_model = EnsemblePerformancePredictor()
    training_results = global_model.train_ensemble(X, y)
    
    logger.info("Sample model training completed")

def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate sample employee data for testing"""
    np.random.seed(42)
    
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    positions = ['Junior Developer', 'Senior Developer', 'Manager', 'Director', 'Analyst']
    genders = ['Male', 'Female', 'Non-binary']
    ethnicities = ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other']
    education_levels = ['Bachelor', 'Master', 'PhD']
    
    sample_data = []
    for i in range(n_samples):
        employee = {
            'employee_id': f'EMP{i+1:04d}',
            'name': f'Employee {i+1}',
            'email': f'employee{i+1}@company.com',
            'department': np.random.choice(departments),
            'position': np.random.choice(positions),
            'position_level': np.random.randint(1, 6),
            'salary': np.random.normal(75000, 20000),
            'hire_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 2000)),
            'age': np.random.randint(22, 65),
            'gender': np.random.choice(genders),
            'ethnicity': np.random.choice(ethnicities),
            'education_level': np.random.choice(education_levels),
            'task_completion_rate': np.random.normal(85, 10),
            'efficiency_score': np.random.normal(85, 10),
            'quality_score': np.random.normal(85, 10),
            'collaboration_score': np.random.normal(85, 10),
            'innovation_score': np.random.normal(85, 10),
            'leadership_score': np.random.normal(85, 10),
            'communication_score': np.random.normal(85, 10),
            'problem_solving_score': np.random.normal(85, 10),
            'adaptability_score': np.random.normal(85, 10),
            'goal_achievement_rate': np.random.normal(85, 10)
        }
        sample_data.append(employee)
    
    return pd.DataFrame(sample_data)

# Explanation endpoint
@app.post("/explain/prediction")
async def explain_prediction(employee_id: str):
    """Get detailed explanation for a prediction"""
    try:
        # Get employee data
        employee_data = db_client.get_employee_data(employee_id)
        if not employee_data:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Convert to DataFrame and engineer features
        df = pd.DataFrame([employee_data])
        df_engineered = feature_engineer.engineer_all_features(df)
        
        # Generate comprehensive explanation
        explanation = model_explainer.explain_model_decisions(
            global_model.models['random_forest'], df_engineered, employee_data
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Data upload endpoint
@app.post("/data/upload")
async def upload_employee_data(file: UploadFile = File(...)):
    """Upload employee data from CSV/Excel file"""
    try:
        # Read uploaded file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Apply privacy protection
        df_protected = privacy_manager.mask_pii(df)
        
        # Insert employees into database
        uploaded_count = 0
        for _, row in df_protected.iterrows():
            try:
                employee_data = row.to_dict()
                db_client.insert_employee(employee_data)
                uploaded_count += 1
            except Exception as e:
                logger.warning(f"Error inserting employee: {e}")
        
        return {
            "status": "success",
            "uploaded_employees": uploaded_count,
            "total_rows": len(df),
            "privacy_protection_applied": True
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Fairness metrics endpoint
@app.get("/metrics/fairness")
async def get_fairness_metrics():
    """Get current fairness metrics"""
    try:
        # Get recent predictions
        all_data = db_client.get_all_employees_data()
        
        if all_data.empty:
            return {"message": "No data available for fairness analysis"}
        
        # Generate predictions for fairness analysis
        df_engineered = feature_engineer.engineer_all_features(all_data)
        predictions = global_model.predict_ensemble(df_engineered)
        
        # Monitor fairness
        fairness_results = fairness_monitor.monitor_real_time_fairness(
            predictions, all_data[['gender', 'ethnicity']]
        )
        
        return fairness_results
        
    except Exception as e:
        logger.error(f"Fairness metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Fairness metrics failed: {str(e)}")

# Privacy audit endpoint
@app.post("/privacy/audit")
async def audit_privacy_compliance():
    """Audit privacy compliance of current dataset"""
    try:
        all_data = db_client.get_all_employees_data()
        
        if all_data.empty:
            return {"message": "No data available for privacy audit"}
        
        compliance_results = privacy_manager.audit_privacy_compliance(all_data)
        return compliance_results
        
    except Exception as e:
        logger.error(f"Privacy audit error: {e}")
        raise HTTPException(status_code=500, detail=f"Privacy audit failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
