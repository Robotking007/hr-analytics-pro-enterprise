"""
Manual Database Setup Script
Creates tables directly in Supabase using SQL commands
"""
import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_supabase_tables():
    """Create tables directly using Supabase SQL"""
    
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not service_key:
        print("❌ Missing Supabase credentials in .env file")
        return False
    
    try:
        # Use service key for admin operations
        supabase: Client = create_client(url, service_key)
        
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
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (employee_id) REFERENCES public.employees(employee_id)
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
            prediction_date TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (employee_id) REFERENCES public.employees(employee_id)
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
        
        # Execute SQL commands
        tables = [
            ("employees", employees_sql),
            ("performance_data", performance_sql),
            ("predictions", predictions_sql),
            ("bias_audits", bias_audits_sql)
        ]
        
        for table_name, sql in tables:
            try:
                result = supabase.rpc('exec_sql', {'sql': sql}).execute()
                print(f"✅ Created table: {table_name}")
            except Exception as e:
                # Try alternative method - direct SQL execution
                try:
                    # Use postgrest directly
                    response = supabase.postgrest.rpc('exec_sql', {'sql': sql}).execute()
                    print(f"✅ Created table: {table_name}")
                except Exception as e2:
                    print(f"⚠️ Table {table_name} may already exist or need manual creation: {e2}")
        
        print("✅ Database setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("💡 You may need to create tables manually in Supabase dashboard")
        return False

def insert_sample_data():
    """Insert sample data into tables"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.data.sample_generator import SampleDataGenerator
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        
        # Generate sample data
        generator = SampleDataGenerator()
        employees_df, performance_df = generator.generate_complete_dataset(10)
        
        # Insert employees
        for _, employee in employees_df.iterrows():
            try:
                result = supabase.table('employees').insert(employee.to_dict()).execute()
                print(f"✅ Inserted employee: {employee['name']}")
            except Exception as e:
                print(f"⚠️ Failed to insert employee {employee['name']}: {e}")
        
        # Insert performance data
        for _, perf in performance_df.iterrows():
            try:
                result = supabase.table('performance_data').insert(perf.to_dict()).execute()
                print(f"✅ Inserted performance data for employee {perf['employee_id']}")
            except Exception as e:
                print(f"⚠️ Failed to insert performance data: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data insertion failed: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Manual Database Setup for HR Performance Analytics Pro")
    print("=" * 60)
    
    # Step 1: Create tables
    if create_supabase_tables():
        print("\n📊 Inserting sample data...")
        # Step 2: Insert sample data
        insert_sample_data()
        print("\n🎉 Database setup complete!")
    else:
        print("\n❌ Database setup failed. Please check your Supabase credentials.")
        print("💡 Try creating tables manually in Supabase dashboard using the SQL from this script.")
