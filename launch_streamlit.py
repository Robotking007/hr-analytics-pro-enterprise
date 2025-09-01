"""
Simple Streamlit Launcher - No FastAPI
"""
import subprocess
import sys
import os

def install_streamlit_deps():
    """Install only Streamlit dependencies"""
    print("📦 Installing Streamlit dependencies...")
    
    packages = [
        "streamlit",
        "supabase", 
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn",
        "python-dotenv",
        "loguru"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package}")
        except:
            print(f"⚠️ {package} failed")

def launch_streamlit():
    """Launch Streamlit app"""
    print("🚀 Starting HR Performance Analytics Pro...")
    print("📊 Streamlit + Supabase Edition")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")

if __name__ == "__main__":
    install_streamlit_deps()
    launch_streamlit()
