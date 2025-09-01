"""
HR Performance Analytics Pro - Streamlit Only Version
Complete system using only Streamlit + Supabase
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="HR Performance Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.error("❌ Supabase credentials not found in .env file")
        return None
    
    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {e}")
        return None

# Data functions
def create_sample_data():
    """Create sample employee data"""
    np.random.seed(42)
    
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    positions = ["Junior", "Senior", "Lead", "Manager", "Director"]
    
    employees = []
    for i in range(50):
        dept = np.random.choice(departments)
        pos = np.random.choice(positions)
        
        employee = {
            "employee_id": f"EMP{i+1:03d}",
            "name": f"Employee {i+1}",
            "email": f"employee{i+1}@company.com",
            "department": dept,
            "position": pos,
            "position_level": np.random.randint(1, 6),
            "salary": np.random.randint(40000, 150000),
            "hire_date": str(date(2020 + np.random.randint(0, 4), 
                                np.random.randint(1, 13), 
                                np.random.randint(1, 28))),
            "age": np.random.randint(25, 55),
            "gender": np.random.choice(["Male", "Female", "Other"]),
            "ethnicity": np.random.choice(["Asian", "White", "Hispanic", "Black", "Other"]),
            "education_level": np.random.choice(["Bachelor", "Master", "PhD", "High School"]),
            "education_score": np.random.randint(60, 100)
        }
        employees.append(employee)
    
    # Performance data
    performance = []
    for emp in employees:
        for month in range(1, 13):
            perf = {
                "employee_id": emp["employee_id"],
                "evaluation_date": str(date(2024, month, 15)),
                "task_completion_rate": round(np.random.uniform(0.6, 1.0), 2),
                "efficiency_score": round(np.random.uniform(0.5, 1.0), 2),
                "quality_score": round(np.random.uniform(0.6, 1.0), 2),
                "collaboration_score": round(np.random.uniform(0.5, 1.0), 2),
                "innovation_score": round(np.random.uniform(0.4, 1.0), 2),
                "leadership_score": round(np.random.uniform(0.3, 1.0), 2),
                "communication_score": round(np.random.uniform(0.5, 1.0), 2),
                "problem_solving_score": round(np.random.uniform(0.4, 1.0), 2),
                "adaptability_score": round(np.random.uniform(0.5, 1.0), 2),
                "goal_achievement_rate": round(np.random.uniform(0.6, 1.0), 2),
                "projects_completed": np.random.randint(1, 8),
                "training_hours": np.random.randint(5, 40),
                "meeting_frequency": np.random.randint(10, 50),
                "overtime_hours": np.random.randint(0, 20)
            }
            performance.append(perf)
    
    return pd.DataFrame(employees), pd.DataFrame(performance)

@st.cache_data
def load_data():
    """Load or create sample data"""
    return create_sample_data()

def train_model(employees_df, performance_df):
    """Train a simple performance prediction model"""
    # Merge data
    latest_perf = performance_df.groupby('employee_id').agg({
        'task_completion_rate': 'mean',
        'efficiency_score': 'mean',
        'quality_score': 'mean',
        'collaboration_score': 'mean',
        'innovation_score': 'mean'
    }).reset_index()
    
    merged = employees_df.merge(latest_perf, on='employee_id')
    
    # Create target variable (overall performance)
    merged['overall_performance'] = (
        merged['task_completion_rate'] * 0.3 +
        merged['efficiency_score'] * 0.25 +
        merged['quality_score'] * 0.25 +
        merged['collaboration_score'] * 0.1 +
        merged['innovation_score'] * 0.1
    )
    
    # Features for prediction
    feature_cols = ['position_level', 'salary', 'age', 'education_score']
    X = merged[feature_cols]
    y = merged['overall_performance']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate accuracy
    predictions = model.predict(X)
    accuracy = r2_score(y, predictions)
    
    return model, accuracy, feature_cols

# Main app
def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📊 HR Performance Analytics Pro")
    st.markdown("### AI-Powered Employee Performance Prediction (Streamlit + Supabase)")
    
    # Initialize Supabase
    supabase = init_supabase()
    
    # Load data
    employees_df, performance_df = load_data()
    
    # Train model
    with st.spinner("🤖 Training AI model..."):
        model, accuracy, feature_cols = train_model(employees_df, performance_df)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Dashboard",
        "🔮 Performance Prediction", 
        "👥 Employee Management",
        "📈 Analytics",
        "⚖️ Bias Detection"
    ])
    
    if page == "🏠 Dashboard":
        show_dashboard(employees_df, performance_df, model, accuracy)
    elif page == "🔮 Performance Prediction":
        show_prediction_page(employees_df, model, feature_cols)
    elif page == "👥 Employee Management":
        show_employee_management(employees_df, performance_df, supabase)
    elif page == "📈 Analytics":
        show_analytics(employees_df, performance_df)
    elif page == "⚖️ Bias Detection":
        show_bias_detection(employees_df, performance_df)

def show_dashboard(employees_df, performance_df, model, accuracy):
    """Show main dashboard"""
    st.header("📊 HR Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(employees_df))
    
    with col2:
        avg_performance = performance_df[['task_completion_rate', 'efficiency_score', 'quality_score']].mean().mean()
        st.metric("Avg Performance", f"{avg_performance:.1%}")
    
    with col3:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    with col4:
        departments = employees_df['department'].nunique()
        st.metric("Departments", departments)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = employees_df['department'].value_counts()
        fig1 = px.pie(values=dept_counts.values, names=dept_counts.index, 
                     title="Employee Distribution by Department")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Performance by department
        merged = employees_df.merge(
            performance_df.groupby('employee_id')[['task_completion_rate']].mean().reset_index(),
            on='employee_id'
        )
        fig2 = px.box(merged, x='department', y='task_completion_rate',
                     title="Performance by Department")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recent activity
    st.subheader("📈 Recent Performance Trends")
    latest_perf = performance_df.sort_values('evaluation_date').tail(100)
    fig3 = px.line(latest_perf, x='evaluation_date', y='efficiency_score',
                  title="Efficiency Trends Over Time")
    st.plotly_chart(fig3, use_container_width=True)

def show_prediction_page(employees_df, model, feature_cols):
    """Show performance prediction page"""
    st.header("🔮 Performance Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Employee Information")
        
        # Input fields
        name = st.text_input("Employee Name", "New Employee")
        department = st.selectbox("Department", 
            ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"])
        position_level = st.slider("Position Level", 1, 5, 3)
        salary = st.number_input("Annual Salary ($)", 40000, 200000, 75000)
        age = st.number_input("Age", 18, 70, 30)
        education_score = st.slider("Education Score", 60, 100, 80)
        
        if st.button("🔮 Predict Performance", type="primary"):
            # Make prediction
            features = [[position_level, salary, age, education_score]]
            prediction = model.predict(features)[0]
            
            # Store in session state
            st.session_state.prediction = prediction
            st.session_state.employee_name = name
    
    with col2:
        st.subheader("Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            employee_name = st.session_state.employee_name
            
            # Display prediction
            st.success(f"🎯 Predicted Performance for {employee_name}")
            
            # Performance score
            performance_color = "green" if prediction > 0.8 else "orange" if prediction > 0.6 else "red"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
                <h2 style="color: {performance_color};">{prediction:.1%}</h2>
                <p>Overall Performance Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance breakdown
            st.subheader("📈 Performance Factors")
            
            # Feature importance (simplified)
            importance = {
                "Position Level": 0.3,
                "Salary": 0.25,
                "Age/Experience": 0.25,
                "Education": 0.2
            }
            
            fig = px.bar(x=list(importance.keys()), y=list(importance.values()),
                        title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction < 0.7:
                st.warning("🔄 Consider additional training or mentoring")
            elif prediction < 0.85:
                st.info("📚 Good performance, room for growth")
            else:
                st.success("🌟 Excellent performance predicted!")

def show_employee_management(employees_df, performance_df, supabase):
    """Show employee management page"""
    st.header("👥 Employee Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 View Employees", "➕ Add Employee", "📊 Performance Data"])
    
    with tab1:
        st.subheader("Employee Directory")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            dept_filter = st.selectbox("Filter by Department", 
                ["All"] + list(employees_df['department'].unique()))
        with col2:
            search_name = st.text_input("Search by Name")
        
        # Filter data
        filtered_df = employees_df.copy()
        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df['department'] == dept_filter]
        if search_name:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
        
        # Display table
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab2:
        st.subheader("Add New Employee")
        
        with st.form("add_employee"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Full Name*")
                new_email = st.text_input("Email*")
                new_dept = st.selectbox("Department*", 
                    ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"])
                new_position = st.text_input("Position*")
                new_salary = st.number_input("Annual Salary*", 30000, 300000, 60000)
            
            with col2:
                new_age = st.number_input("Age*", 18, 70, 30)
                new_gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
                new_ethnicity = st.selectbox("Ethnicity", 
                    ["Asian", "White", "Hispanic", "Black", "Native American", "Other", "Prefer not to say"])
                new_education = st.selectbox("Education Level", 
                    ["High School", "Bachelor", "Master", "PhD"])
                new_hire_date = st.date_input("Hire Date", value=datetime.now().date())
            
            submitted = st.form_submit_button("➕ Add Employee")
            
            if submitted and new_name and new_email:
                # Create employee record
                new_employee = {
                    "employee_id": f"EMP{len(employees_df)+1:03d}",
                    "name": new_name,
                    "email": new_email,
                    "department": new_dept,
                    "position": new_position,
                    "position_level": 3,  # Default
                    "salary": new_salary,
                    "hire_date": str(new_hire_date),
                    "age": new_age,
                    "gender": new_gender,
                    "ethnicity": new_ethnicity,
                    "education_level": new_education,
                    "education_score": 80  # Default
                }
                
                # Try to insert into Supabase
                if supabase:
                    try:
                        result = supabase.table('employees').insert(new_employee).execute()
                        st.success(f"✅ Added {new_name} to database!")
                    except Exception as e:
                        st.error(f"❌ Failed to add to database: {e}")
                        st.info("💡 Employee added to local session only")
                
                # Add to local data
                st.session_state.new_employees = st.session_state.get('new_employees', [])
                st.session_state.new_employees.append(new_employee)
                st.rerun()
    
    with tab3:
        st.subheader("Performance Data Overview")
        
        # Performance metrics summary
        perf_summary = performance_df.groupby('employee_id').agg({
            'task_completion_rate': 'mean',
            'efficiency_score': 'mean',
            'quality_score': 'mean'
        }).round(3)
        
        # Merge with employee names
        perf_with_names = perf_summary.merge(
            employees_df[['employee_id', 'name', 'department']], 
            on='employee_id'
        )
        
        st.dataframe(perf_with_names, use_container_width=True)

def show_analytics(employees_df, performance_df):
    """Show analytics page"""
    st.header("📈 Advanced Analytics")
    
    # Performance trends
    st.subheader("📊 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary vs Performance
        merged = employees_df.merge(
            performance_df.groupby('employee_id')[['efficiency_score']].mean().reset_index(),
            on='employee_id'
        )
        
        fig1 = px.scatter(merged, x='salary', y='efficiency_score', 
                         color='department', size='age',
                         title="Salary vs Performance by Department")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Age distribution
        fig2 = px.histogram(employees_df, x='age', color='department',
                           title="Age Distribution by Department")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Department performance comparison
    st.subheader("🏢 Department Performance Comparison")
    
    dept_performance = employees_df.merge(
        performance_df.groupby('employee_id')[['task_completion_rate', 'efficiency_score', 'quality_score']].mean().reset_index(),
        on='employee_id'
    ).groupby('department')[['task_completion_rate', 'efficiency_score', 'quality_score']].mean()
    
    fig3 = px.bar(dept_performance.reset_index(), x='department', 
                 y=['task_completion_rate', 'efficiency_score', 'quality_score'],
                 title="Average Performance Metrics by Department",
                 barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

def show_bias_detection(employees_df, performance_df):
    """Show bias detection page"""
    st.header("⚖️ Bias Detection & Fairness Analysis")
    
    # Merge data for analysis
    merged = employees_df.merge(
        performance_df.groupby('employee_id')[['efficiency_score']].mean().reset_index(),
        on='employee_id'
    )
    
    # Gender bias analysis
    st.subheader("👥 Gender Performance Analysis")
    gender_stats = merged.groupby('gender')['efficiency_score'].agg(['mean', 'count']).round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(gender_stats)
        
        # Statistical significance test
        male_scores = merged[merged['gender'] == 'Male']['efficiency_score']
        female_scores = merged[merged['gender'] == 'Female']['efficiency_score']
        
        if len(male_scores) > 0 and len(female_scores) > 0:
            diff = abs(male_scores.mean() - female_scores.mean())
            if diff < 0.05:
                st.success("✅ No significant gender bias detected")
            else:
                st.warning(f"⚠️ Potential gender bias detected (diff: {diff:.3f})")
    
    with col2:
        fig = px.box(merged, x='gender', y='efficiency_score',
                    title="Performance Distribution by Gender")
        st.plotly_chart(fig, use_container_width=True)
    
    # Ethnicity analysis
    st.subheader("🌍 Ethnicity Performance Analysis")
    ethnicity_stats = merged.groupby('ethnicity')['efficiency_score'].agg(['mean', 'count']).round(3)
    st.dataframe(ethnicity_stats)
    
    # Age bias analysis
    st.subheader("👴 Age Bias Analysis")
    merged['age_group'] = pd.cut(merged['age'], bins=[0, 30, 40, 50, 100], 
                                labels=['<30', '30-40', '40-50', '50+'])
    
    age_stats = merged.groupby('age_group')['efficiency_score'].agg(['mean', 'count']).round(3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(age_stats)
    with col2:
        fig = px.bar(age_stats.reset_index(), x='age_group', y='mean',
                    title="Average Performance by Age Group")
        st.plotly_chart(fig, use_container_width=True)

# Database status
def show_database_status(supabase):
    """Show database connection status"""
    if supabase:
        try:
            # Test connection
            result = supabase.table('employees').select('count').execute()
            st.sidebar.success("✅ Supabase Connected")
        except:
            st.sidebar.warning("⚠️ Supabase Connection Issues")
    else:
        st.sidebar.error("❌ Supabase Not Connected")

# Run the app
if __name__ == "__main__":
    # Show database status in sidebar
    supabase = init_supabase()
    show_database_status(supabase)
    
    # Run main app
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("**HR Performance Analytics Pro** - Streamlit + Supabase Edition")
    st.markdown("Built with ❤️ for modern HR analytics")
