"""
Application Launcher for HR Performance Analytics Pro (No TensorFlow)
Starts both FastAPI backend and Streamlit dashboard without TensorFlow dependencies
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def start_backend():
    """Start FastAPI backend without TensorFlow"""
    print("🔧 Starting FastAPI backend (no TensorFlow)...")
    
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main_no_tf:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    return subprocess.Popen(backend_cmd)

def start_dashboard():
    """Start Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    
    dashboard_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/main.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    
    return subprocess.Popen(dashboard_cmd)

def main():
    """Main application launcher"""
    print("🚀 HR Performance Analytics Pro - Application Launcher")
    print("=" * 60)
    print("⚠️ Running without TensorFlow to avoid DLL issues")
    print("✅ Using scikit-learn, XGBoost, and LightGBM models")
    print("=" * 60)
    
    try:
        # Start backend
        backend_process = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start dashboard
        dashboard_process = start_dashboard()
        
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

if __name__ == "__main__":
    main()
