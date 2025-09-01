"""
Direct PostgreSQL Client for Supabase Database
Uses psycopg2 for direct database connection
"""
import os
import psycopg2
import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class PostgreSQLClient:
    """Direct PostgreSQL connection to Supabase"""
    
    def __init__(self):
        self.host = os.getenv("DB_HOST", "db.ybtcbjycttvuvqkdgxtr.supabase.co")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "postgres")
        self.database = os.getenv("DB_NAME", "postgres")
        self.password = os.getenv("DB_PASSWORD")
        
        if not self.password:
            logger.warning("DB_PASSWORD not set. You'll need to provide the database password.")
        
        self.connection = None
        
    def connect(self, password: Optional[str] = None):
        """Connect to PostgreSQL database"""
        try:
            db_password = password or self.password
            if not db_password:
                raise ValueError("Database password is required")
            
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=db_password,
                database=self.database
            )
            
            logger.info(f"✅ Connected to PostgreSQL at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False
    
    def create_tables(self):
        """Create all necessary tables"""
        if not self.connection:
            logger.error("No database connection. Call connect() first.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create employees table
            employees_sql = """
            CREATE TABLE IF NOT EXISTS public.employees (
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
            
            # Create performance_data table
            performance_sql = """
            CREATE TABLE IF NOT EXISTS public.performance_data (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                employee_id VARCHAR NOT NULL,
                evaluation_date DATE NOT NULL,
                task_completion_rate DECIMAL(3,2),
                efficiency_score DECIMAL(3,2),
                quality_score DECIMAL(3,2),
                collaboration_score DECIMAL(3,2),
                innovation_score DECIMAL(3,2),
                leadership_score DECIMAL(3,2),
                communication_score DECIMAL(3,2),
                problem_solving_score DECIMAL(3,2),
                adaptability_score DECIMAL(3,2),
                goal_achievement_rate DECIMAL(3,2),
                projects_completed INTEGER,
                training_hours INTEGER,
                meeting_frequency INTEGER,
                overtime_hours INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            
            # Create predictions table
            predictions_sql = """
            CREATE TABLE IF NOT EXISTS public.predictions (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                employee_id VARCHAR NOT NULL,
                predicted_performance DECIMAL(3,2),
                prediction_confidence DECIMAL(3,2),
                model_version VARCHAR,
                features_used JSONB,
                prediction_date TIMESTAMP DEFAULT NOW()
            );
            """
            
            # Create bias_audits table
            bias_audits_sql = """
            CREATE TABLE IF NOT EXISTS public.bias_audits (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                audit_date TIMESTAMP DEFAULT NOW(),
                protected_attribute VARCHAR NOT NULL,
                demographic_parity DECIMAL(3,2),
                equalized_odds DECIMAL(3,2),
                disparate_impact DECIMAL(3,2),
                overall_fairness DECIMAL(3,2),
                sample_size INTEGER,
                audit_results JSONB
            );
            """
            
            # Execute table creation
            tables = [
                ("employees", employees_sql),
                ("performance_data", performance_sql),
                ("predictions", predictions_sql),
                ("bias_audits", bias_audits_sql)
            ]
            
            for table_name, sql in tables:
                cursor.execute(sql)
                logger.info(f"✅ Created/verified table: {table_name}")
            
            self.connection.commit()
            cursor.close()
            
            logger.info("✅ All tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def insert_employee(self, employee_data: Dict[str, Any]) -> bool:
        """Insert employee data"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare insert statement
            columns = list(employee_data.keys())
            placeholders = ['%s'] * len(columns)
            
            sql = f"""
            INSERT INTO public.employees ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (employee_id) DO UPDATE SET
            {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'employee_id'])}
            """
            
            cursor.execute(sql, list(employee_data.values()))
            self.connection.commit()
            cursor.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert employee: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def insert_performance_data(self, perf_data: Dict[str, Any]) -> bool:
        """Insert performance data"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare insert statement
            columns = list(perf_data.keys())
            placeholders = ['%s'] * len(columns)
            
            sql = f"""
            INSERT INTO public.performance_data ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            
            cursor.execute(sql, list(perf_data.values()))
            self.connection.commit()
            cursor.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert performance data: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_all_employees(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all employees"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM public.employees LIMIT {limit}")
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get employees: {e}")
            return []
    
    def get_performance_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get performance data"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM public.performance_data LIMIT {limit}")
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
