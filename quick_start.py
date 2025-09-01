"""
Quick Start Script for HR Performance Analytics Pro
Sets up and runs the complete system with minimal configuration
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def create_env_file():
    """Create .env file with default values"""
    env_content = """# Supabase Configuration (Optional - system works with sample data)
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_KEY=

# Security
SECRET_KEY=hr-analytics-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=50MB

# Model Settings
FAIRNESS_THRESHOLD=0.8
PRIVACY_BUDGET=1.0
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file")

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "models", "data/raw", "data/processed"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Created directories")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_services():
    """Start FastAPI and Streamlit services"""
    print("🚀 Starting HR Performance Analytics Pro...")
    
    # Start FastAPI backend
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # Start Streamlit dashboard
    dashboard_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/main.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    
    try:
        backend_process = subprocess.Popen(backend_cmd)
        time.sleep(3)  # Give backend time to start
        
        dashboard_process = subprocess.Popen(dashboard_cmd)
        
        print("\n" + "="*60)
        print("🎉 HR Performance Analytics Pro is now running!")
        print("="*60)
        print("📊 Dashboard: http://localhost:8501")
        print("🔗 API Docs: http://localhost:8000/docs")
        print("💡 Press Ctrl+C to stop both services")
        print("="*60)
        
        # Wait for processes
        try:
            backend_process.wait()
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")
            backend_process.terminate()
            dashboard_process.terminate()
            print("✅ Services stopped successfully")
    
    except Exception as e:
        print(f"❌ Error starting services: {e}")

def main():
    """Main quick start function"""
    print("🚀 HR Performance Analytics Pro - Quick Start")
    print("=" * 50)
    
    # Setup
    if not os.path.exists(".env"):
        create_env_file()
    
    create_directories()
    
    if not install_dependencies():
        print("❌ Setup failed. Please install dependencies manually:")
        print("pip install -r requirements.txt")
        return
    
    print("\n✅ Setup complete! Starting services...")
    start_services()

if __name__ == "__main__":
    main()
