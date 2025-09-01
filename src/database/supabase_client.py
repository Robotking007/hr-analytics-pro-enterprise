"""
Supabase client configuration and database operations
"""
import os
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd
from loguru import logger

load_dotenv()

class SupabaseClient:
    """Supabase client for HR Analytics Pro"""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.service_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        self.admin_client: Client = create_client(self.url, self.service_key) if self.service_key else None
        
    def create_tables_sync(self):
        """Create all necessary tables for HR analytics"""
        
        # Employees table
        employees_schema = """
        CREATE TABLE IF NOT EXISTS employees (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            employee_id VARCHAR UNIQUE NOT NULL,
            name VARCHAR NOT NULL,
            email VARCHAR UNIQUE NOT NULL,
            department VARCHAR NOT NULL,
            position VARCHAR NOT NULL,
            position_level INTEGER NOT NULL,
            salary DECIMAL(10,2),
            hire_date DATE NOT NULL,
            age INTEGER,
            gender VARCHAR(20),
            ethnicity VARCHAR(50),
            education_level VARCHAR(50),
            education_score INTEGER,
            manager_id VARCHAR,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Performance metrics table
        performance_schema = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            employee_id VARCHAR NOT NULL REFERENCES employees(employee_id),
            metric_date DATE NOT NULL,
            task_completion_rate DECIMAL(5,2),
            efficiency_score DECIMAL(5,2),
            quality_score DECIMAL(5,2),
            collaboration_score DECIMAL(5,2),
            innovation_score DECIMAL(5,2),
            leadership_score DECIMAL(5,2),
            communication_score DECIMAL(5,2),
            problem_solving_score DECIMAL(5,2),
            adaptability_score DECIMAL(5,2),
            goal_achievement_rate DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Predictions table
        predictions_schema = """
        CREATE TABLE IF NOT EXISTS predictions (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            employee_id VARCHAR NOT NULL REFERENCES employees(employee_id),
            prediction_date TIMESTAMP DEFAULT NOW(),
            predicted_performance DECIMAL(5,2) NOT NULL,
            confidence_score DECIMAL(5,2) NOT NULL,
            model_version VARCHAR NOT NULL,
            features_used JSONB,
            explanation JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Bias audit table
        bias_audit_schema = """
        CREATE TABLE IF NOT EXISTS bias_audits (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            audit_date TIMESTAMP DEFAULT NOW(),
            model_version VARCHAR NOT NULL,
            protected_attribute VARCHAR NOT NULL,
            demographic_parity DECIMAL(5,4),
            equalized_odds DECIMAL(5,4),
            disparate_impact DECIMAL(5,4),
            statistical_parity_diff DECIMAL(5,4),
            fairness_threshold_met BOOLEAN,
            audit_results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Model metadata table
        models_schema = """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_version VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            training_date TIMESTAMP NOT NULL,
            feature_importance JSONB,
            hyperparameters JSONB,
            is_active BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        schemas = [employees_schema, performance_schema, predictions_schema, 
                  bias_audit_schema, models_schema]
        
        for schema in schemas:
            try:
                if self.admin_client:
                    result = self.admin_client.rpc('exec_sql', {'sql': schema}).execute()
                    logger.info(f"Created table successfully")
                else:
                    logger.warning("Admin client not available, using regular client")
                    # Note: This might not work without proper permissions
            except Exception as e:
                logger.error(f"Error creating table: {e}")
    
    def insert_employee(self, employee_data: Dict) -> Dict:
        """Insert new employee record"""
        try:
            result = self.client.table("employees").insert(employee_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error inserting employee: {e}")
            raise
    
    def insert_performance_metrics(self, metrics_data: List[Dict]) -> List[Dict]:
        """Insert performance metrics"""
        try:
            result = self.client.table("performance_metrics").insert(metrics_data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting performance metrics: {e}")
            raise
    
    def get_employee_data(self, employee_id: str) -> Optional[Dict]:
        """Get employee data by ID"""
        try:
            result = self.client.table("employees").select("*").eq("employee_id", employee_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting employee data: {e}")
            return None
    
    def get_performance_history(self, employee_id: str, months: int = 12) -> pd.DataFrame:
        """Get performance history for an employee"""
        try:
            result = self.client.table("performance_metrics")\
                .select("*")\
                .eq("employee_id", employee_id)\
                .order("metric_date", desc=True)\
                .limit(months)\
                .execute()
            
            return pd.DataFrame(result.data) if result.data else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction_data: Dict) -> Dict:
        """Save prediction results"""
        try:
            result = self.client.table("predictions").insert(prediction_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            raise
    
    def save_bias_audit(self, audit_data: Dict) -> Dict:
        """Save bias audit results"""
        try:
            result = self.client.table("bias_audits").insert(audit_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error saving bias audit: {e}")
            raise
    
    def get_all_employees_data(self) -> pd.DataFrame:
        """Get all employees data for training"""
        try:
            # Get employees
            employees_result = self.client.table("employees").select("*").execute()
            employees_df = pd.DataFrame(employees_result.data)
            
            if employees_df.empty:
                return pd.DataFrame()
            
            # Get performance metrics for all employees
            metrics_result = self.client.table("performance_metrics").select("*").execute()
            metrics_df = pd.DataFrame(metrics_result.data)
            
            if metrics_df.empty:
                return employees_df
            
            # Merge data
            merged_df = employees_df.merge(
                metrics_df.groupby('employee_id').last().reset_index(),
                on='employee_id',
                how='left'
            )
            
            return merged_df
        except Exception as e:
            logger.error(f"Error getting all employees data: {e}")
            return pd.DataFrame()
    
    def save_model_metadata(self, model_data: Dict) -> Dict:
        """Save model training metadata"""
        try:
            result = self.client.table("model_metadata").insert(model_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            raise
    
    def get_active_model(self) -> Optional[Dict]:
        """Get currently active model"""
        try:
            result = self.client.table("model_metadata")\
                .select("*")\
                .eq("is_active", True)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None
