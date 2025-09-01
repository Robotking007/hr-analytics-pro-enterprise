import os
import subprocess
import time
from dotenv import load_dotenv

def run_command(command, cwd=None):
    """Run a shell command"""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Check for errors
        _, stderr = process.communicate()
        if stderr:
            print(f"Error: {stderr}")
            
        return process.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting HR Analytics Pro - Enterprise Edition Setup 🚀")
    print("=" * 60)
    
    # Check if required environment variables are set
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "JWT_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with these variables or set them in your environment.")
        return
    
    print("✅ Environment variables loaded successfully")
    
    # Install required packages
    print("\n🔧 Installing required packages...")
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install required packages")
        return
    print("✅ Packages installed successfully")
    
    # Set up authentication tables
    print("\n🔐 Setting up authentication system...")
    if not run_command("python setup_auth_tables.py"):
        print("❌ Failed to set up authentication tables")
        return
    print("✅ Authentication system set up successfully")
    
    # Set up database schema and initial data
    print("\n💾 Setting up database...")
    if not run_command("python setup_database.py"):
        print("❌ Failed to set up database")
        return
    print("✅ Database set up successfully")
    
    # Start the Streamlit app
    print("\n🚀 Starting HR Analytics Pro - Enterprise Edition...")
    print("=" * 60)
    print("\n🔑 Default Admin Credentials:")
    print("   Email: admin@hranytics.com")
    print("   Password: admin123\n")
    print("🌐 Opening in your default browser...")
    
    # Start Streamlit app
    if not run_command("streamlit run main_enterprise.py"):
        print("❌ Failed to start the application")
        return

if __name__ == "__main__":
    main()
