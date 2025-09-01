"""
System Startup with Direct PostgreSQL Connection
Uses direct database connection instead of Supabase client
"""
import sys
import os
import subprocess
import time
import getpass
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_database_postgres():
    """Setup database using direct PostgreSQL connection"""
    print("🗄️ Setting up database with direct PostgreSQL connection...")
    
    try:
        from src.database.postgres_client import PostgreSQLClient
        from src.data.sample_generator import SampleDataGenerator
        
        # Get password from user
        password = getpass.getpass("Enter your Supabase database password: ")
        
        # Create client and connect
        client = PostgreSQLClient()
        
        if client.connect(password):
            print("✅ Database connection successful!")
            
            # Create tables
            if client.create_tables():
                print("✅ Database tables created successfully")
                
                # Generate and insert sample data
                generator = SampleDataGenerator()
                employees_df, performance_df = generator.generate_complete_dataset(20)
                
                # Insert employees
                success_count = 0
                for _, employee in employees_df.iterrows():
                    if client.insert_employee(employee.to_dict()):
                        success_count += 1
                
                # Insert performance data
                perf_success = 0
                for _, perf in performance_df.iterrows():
                    if client.insert_performance_data(perf.to_dict()):
                        perf_success += 1
                
                print(f"✅ Inserted {success_count} employees and {perf_success} performance records")
                client.close()
                return True
            else:
                print("❌ Failed to create database tables")
                client.close()
                return False
        else:
            print("❌ Failed to connect to database")
            return False
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def start_services():
    """Start FastAPI and Streamlit services"""
    print("🚀 Starting services...")
    
    try:
        # Start using the TensorFlow-free launcher
        subprocess.run([sys.executable, "run_app_no_tf.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Services stopped by user")
    except Exception as e:
        print(f"❌ Failed to start services: {e}")

def main():
    """Main startup function"""
    print("🚀 HR Performance Analytics Pro - PostgreSQL Direct Connection")
    print("=" * 60)
    
    # Step 1: Setup database
    if not setup_database_postgres():
        print("❌ Database setup failed. Please check your credentials.")
        print("💡 Make sure you have the correct database password.")
        return
    
    # Step 2: Start services
    print("\n🎯 Database ready! Starting web services...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔗 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the system")
    print("=" * 60)
    
    start_services()

if __name__ == "__main__":
    main()
