"""
Database Setup with Password from Environment Variable
Avoids terminal password input issues
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_database_with_env_password():
    """Setup database using password from .env file"""
    
    # Check if password is in environment
    db_password = os.getenv("DB_PASSWORD")
    
    if not db_password:
        print("❌ DB_PASSWORD not found in .env file")
        print("💡 Please add your database password to .env file:")
        print("DB_PASSWORD=your_actual_password")
        return False
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="db.ybtcbjycttvuvqkdgxtr.supabase.co",
            port="5432",
            user="postgres",
            password=db_password,
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
        
        # Execute table creation
        cursor.execute(employees_sql)
        print("✅ Created employees table")
        
        cursor.execute(performance_sql)
        print("✅ Created performance_data table")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("🎉 Database setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def insert_sample_data():
    """Insert sample data using direct PostgreSQL"""
    db_password = os.getenv("DB_PASSWORD")
    
    if not db_password:
        print("❌ DB_PASSWORD not found in .env file")
        return False
    
    try:
        # Add src to path for imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.data.sample_generator import SampleDataGenerator
        
        # Connect to database
        conn = psycopg2.connect(
            host="db.ybtcbjycttvuvqkdgxtr.supabase.co",
            port="5432",
            user="postgres",
            password=db_password,
            database="postgres"
        )
        
        cursor = conn.cursor()
        
        # Generate sample data
        generator = SampleDataGenerator()
        employees_df, performance_df = generator.generate_complete_dataset(10)
        
        # Insert employees
        for _, employee in employees_df.iterrows():
            try:
                emp_data = employee.to_dict()
                columns = list(emp_data.keys())
                values = list(emp_data.values())
                placeholders = ['%s'] * len(values)
                
                sql = f"""
                INSERT INTO public.employees ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (employee_id) DO NOTHING
                """
                
                cursor.execute(sql, values)
                print(f"✅ Inserted employee: {employee['name']}")
                
            except Exception as e:
                print(f"⚠️ Failed to insert employee {employee['name']}: {e}")
        
        # Insert performance data
        for _, perf in performance_df.iterrows():
            try:
                perf_data = perf.to_dict()
                columns = list(perf_data.keys())
                values = list(perf_data.values())
                placeholders = ['%s'] * len(values)
                
                sql = f"""
                INSERT INTO public.performance_data ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                """
                
                cursor.execute(sql, values)
                
            except Exception as e:
                print(f"⚠️ Failed to insert performance data: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Sample data inserted successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Sample data insertion failed: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Database Setup with Environment Password")
    print("=" * 50)
    
    if setup_database_with_env_password():
        print("\n📊 Inserting sample data...")
        insert_sample_data()
    else:
        print("\n💡 Please add your database password to .env file:")
        print("DB_PASSWORD=your_actual_password")
