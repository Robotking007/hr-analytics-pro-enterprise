"""
Direct SQL Table Creation for Supabase
Creates tables using direct PostgreSQL connection
"""
import psycopg2
import os
import getpass
from dotenv import load_dotenv

load_dotenv()

def create_tables_direct():
    """Create tables using direct PostgreSQL connection"""
    
    # Get database password
    password = getpass.getpass("Enter your Supabase database password: ")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="db.ybtcbjycttvuvqkdgxtr.supabase.co",
            port="5432",
            user="postgres",
            password=password,
            database="postgres"
        )
        
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL database")
        
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
        
        # Execute table creation
        tables = [
            ("employees", employees_sql),
            ("performance_data", performance_sql),
            ("predictions", predictions_sql)
        ]
        
        for table_name, sql in tables:
            cursor.execute(sql)
            print(f"✅ Created table: {table_name}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("🎉 All tables created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Direct SQL Table Creation for Supabase")
    print("=" * 50)
    create_tables_direct()
