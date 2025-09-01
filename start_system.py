"""
Complete System Startup Script for HR Performance Analytics Pro
Initializes database, trains models, and starts services
"""
import sys
import os
import subprocess
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_database():
    """Initialize Supabase database with schema and sample data"""
    print("🗄️ Setting up database...")
    try:
        from src.database.supabase_client import SupabaseClient
        from src.data.sample_generator import SampleDataGenerator
        
        # Initialize client and create tables
        client = SupabaseClient()
        if client.create_tables_sync():
            print("✅ Database tables created successfully")
            
            # Generate and insert sample data
            generator = SampleDataGenerator()
            employees_df, performance_df = generator.generate_complete_dataset(20)
            
            # Insert employees
            for _, employee in employees_df.iterrows():
                client.insert_employee(employee.to_dict())
            
            # Insert performance data
            for _, perf in performance_df.iterrows():
                client.insert_performance_data(perf.to_dict())
            
            print(f"✅ Inserted {len(employees_df)} employees and {len(performance_df)} performance records")
            return True
        else:
            print("❌ Failed to create database tables")
            return False
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def train_initial_model():
    """Train initial ML model"""
    print("🤖 Training initial ML model...")
    try:
        from src.models.train_ensemble import main as train_model
        train_model()
        print("✅ Model training completed")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def start_services():
    """Start FastAPI and Streamlit services"""
    print("🚀 Starting services...")
    
    try:
        # Start using the run_app.py launcher
        subprocess.run([sys.executable, "run_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Services stopped by user")
    except Exception as e:
        print(f"❌ Failed to start services: {e}")

def main():
    """Main startup function"""
    print("🚀 HR Performance Analytics Pro - Complete System Startup")
    print("=" * 60)
    
    # Step 1: Setup database
    if not setup_database():
        print("❌ Database setup failed. Please check your Supabase credentials.")
        return
    
    # Step 2: Train model (skip TensorFlow for now)
    print("⚠️ Skipping TensorFlow model training due to DLL issues. Using scikit-learn models only.")
    
    # Step 3: Start services
    print("\n🎯 System ready! Starting web services...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔗 API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the system")
    print("=" * 60)
    
    start_services()

if __name__ == "__main__":
    main()
