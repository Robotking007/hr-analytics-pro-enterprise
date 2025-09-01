"""
Streamlit-Only HR Analytics Launcher
No FastAPI, no complex dependencies
"""
import subprocess
import sys
import os

def main():
    """Launch Streamlit app directly"""
    print("🚀 HR Performance Analytics Pro - Streamlit Only")
    print("=" * 50)
    print("📊 Starting dashboard...")
    
    try:
        # Launch Streamlit directly
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_offline.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
