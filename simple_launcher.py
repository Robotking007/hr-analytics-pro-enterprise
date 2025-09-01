"""
Simple System Launcher - Minimal Dependencies
Works without TensorFlow, complex async operations, or problematic imports
"""
import subprocess
import sys
import time
import os

def install_minimal_deps():
    """Install only essential dependencies"""
    print("📦 Installing minimal dependencies...")
    
    essential_packages = [
        "streamlit",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "loguru",
        "psycopg2-binary"
    ]
    
    for package in essential_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")

def create_simple_api():
    """Create a simple API without complex dependencies"""
    api_content = '''"""
Simple FastAPI Backend - Minimal Dependencies
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="HR Analytics API", version="1.0.0")

class EmployeeData(BaseModel):
    name: str
    department: str
    position: str
    salary: float
    age: int

@app.get("/")
async def root():
    return {"message": "HR Performance Analytics Pro API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict_performance(employee: EmployeeData):
    # Simple prediction logic
    base_score = 0.7
    
    # Adjust based on salary (normalized)
    salary_factor = min(employee.salary / 100000, 1.0) * 0.1
    
    # Adjust based on age (experience curve)
    age_factor = 0.1 if employee.age > 30 else 0.05
    
    # Department factor
    dept_factors = {
        "Engineering": 0.15,
        "Sales": 0.12,
        "Marketing": 0.10,
        "HR": 0.08,
        "Finance": 0.09
    }
    dept_factor = dept_factors.get(employee.department, 0.08)
    
    predicted_score = base_score + salary_factor + age_factor + dept_factor
    predicted_score = min(predicted_score, 1.0)
    
    return {
        "employee": employee.name,
        "predicted_performance": round(predicted_score, 3),
        "confidence": 0.85,
        "factors": {
            "salary_factor": salary_factor,
            "experience_factor": age_factor,
            "department_factor": dept_factor
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("simple_api.py", "w", encoding="utf-8") as f:
        f.write(api_content)
    print("✅ Created simple API")

def create_simple_dashboard():
    """Create a simple Streamlit dashboard"""
    dashboard_content = '''"""
Simple HR Analytics Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

st.set_page_config(
    page_title="HR Performance Analytics Pro",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Performance Analytics Pro")
st.markdown("### AI-Powered Employee Performance Prediction")

# Sidebar
st.sidebar.header("Employee Information")

name = st.sidebar.text_input("Employee Name", "John Doe")
department = st.sidebar.selectbox("Department", 
    ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"])
position = st.sidebar.text_input("Position", "Software Engineer")
salary = st.sidebar.number_input("Annual Salary ($)", min_value=30000, max_value=200000, value=75000)
age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30)

if st.sidebar.button("🔮 Predict Performance"):
    try:
        # Make API call
        response = requests.post("http://localhost:8000/predict", json={
            "name": name,
            "department": department,
            "position": position,
            "salary": salary,
            "age": age
        })
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Performance", 
                         f"{result['predicted_performance']:.1%}")
            
            with col2:
                st.metric("Confidence Score", 
                         f"{result['confidence']:.1%}")
            
            with col3:
                st.metric("Employee", result['employee'])
            
            # Factors breakdown
            st.subheader("📈 Performance Factors")
            factors_df = pd.DataFrame([
                {"Factor": "Salary Impact", "Value": result['factors']['salary_factor']},
                {"Factor": "Experience Impact", "Value": result['factors']['experience_factor']},
                {"Factor": "Department Impact", "Value": result['factors']['department_factor']}
            ])
            
            fig = px.bar(factors_df, x="Factor", y="Value", 
                        title="Performance Factor Breakdown")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Failed to get prediction from API")
    
    except requests.exceptions.RequestException:
        st.error("API is not running. Please start the backend first.")

# Sample data visualization
st.subheader("📊 Sample Analytics")

# Generate sample data
np.random.seed(42)
sample_data = pd.DataFrame({
    'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'] * 20,
    'Performance': np.random.normal(0.75, 0.15, 100),
    'Salary': np.random.normal(75000, 20000, 100),
    'Age': np.random.randint(25, 55, 100)
})

col1, col2 = st.columns(2)

with col1:
    fig1 = px.box(sample_data, x='Department', y='Performance', 
                  title='Performance by Department')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(sample_data, x='Age', y='Performance', 
                     color='Department', title='Performance vs Age')
    st.plotly_chart(fig2, use_container_width=True)

# API Status
st.subheader("🔧 System Status")
try:
    health_response = requests.get("http://localhost:8000/health", timeout=2)
    if health_response.status_code == 200:
        st.success("✅ API is running and healthy")
    else:
        st.warning("⚠️ API is responding but may have issues")
except:
    st.error("❌ API is not running")

st.markdown("---")
st.markdown("**HR Performance Analytics Pro** - Built with ❤️ using Streamlit and FastAPI")
'''
    
    with open("simple_dashboard.py", "w", encoding="utf-8") as f:
        f.write(dashboard_content)
    print("✅ Created simple dashboard")

def start_simple_system():
    """Start the simplified system"""
    print("🚀 Starting simplified HR Analytics system...")
    
    try:
        # Start API
        api_process = subprocess.Popen([
            sys.executable, "simple_api.py"
        ])
        
        time.sleep(3)  # Wait for API to start
        
        # Start dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "simple_dashboard.py",
            "--server.port", "8501"
        ])
        
        print("\n" + "="*60)
        print("🎉 HR Performance Analytics Pro is running!")
        print("="*60)
        print("📊 Dashboard: http://localhost:8501")
        print("🔗 API: http://localhost:8000")
        print("💡 Press Ctrl+C to stop")
        print("="*60)
        
        # Wait for processes
        try:
            api_process.wait()
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            api_process.terminate()
            dashboard_process.terminate()
            print("✅ Services stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main launcher function"""
    print("🚀 HR Performance Analytics Pro - Simple Launcher")
    print("=" * 60)
    
    # Install dependencies
    install_minimal_deps()
    
    # Create simple files
    create_simple_api()
    create_simple_dashboard()
    
    # Start system
    start_simple_system()

if __name__ == "__main__":
    main()
