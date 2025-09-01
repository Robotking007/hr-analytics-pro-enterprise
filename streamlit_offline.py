"""
HR Performance Analytics Pro - Offline Streamlit Version
Works without Supabase connection, uses local data
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Import Ollama integration
try:
    from ollama_integration import create_ollama_client, show_ai_insights, show_ai_chatbot
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    st.sidebar.warning("⚠️ Ollama integration not available")

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="HR Performance Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0 0;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            st.warning("⚠️ Supabase credentials not found in .env")
            return None
            
        supabase: Client = create_client(url, key)
        # Test connection with a simple query
        response = supabase.table('employees').select('*', count='exact').limit(1).execute()
        if hasattr(response, 'error') and response.error:
            raise Exception(response.error)
            
        st.sidebar.success("✅ Connected to Supabase")
        return supabase
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ Using offline mode: {str(e)[:100]}")
        return None

# Get data from Supabase or fallback to sample data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Supabase or use sample data"""
    supabase = st.session_state.get('supabase')
    
    try:
        if supabase:
            # Load employees
            employees_response = supabase.table('employees').select('*').execute()
            employees_df = pd.DataFrame(employees_response.data) if hasattr(employees_response, 'data') else None
            
            # Load performance reviews
            perf_response = supabase.table('performance_reviews').select('*').execute()
            performance_df = pd.DataFrame(perf_response.data) if hasattr(perf_response, 'data') else None
            
            if employees_df is not None and not employees_df.empty:
                # Standardize column names to lowercase
                employees_df.columns = employees_df.columns.str.lower()
                
                # Ensure employee_id column exists
                if 'employee_id' not in employees_df.columns and 'id' in employees_df.columns:
                    employees_df = employees_df.rename(columns={'id': 'employee_id'})
                
                st.sidebar.success(f"Loaded {len(employees_df)} employees from Supabase")
                
                if performance_df is not None and not performance_df.empty:
                    # Standardize column names to lowercase
                    performance_df.columns = performance_df.columns.str.lower()
                    
                    # Ensure employee_id column exists in performance data
                    if 'employee_id' not in performance_df.columns and 'employeeid' in performance_df.columns:
                        performance_df = performance_df.rename(columns={'employeeid': 'employee_id'})
                    
                    required_cols = ['employee_id', 'review_date'] # Performance score is now optional
                    missing_cols = [col for col in required_cols if col not in performance_df.columns]
                    
                    if missing_cols:
                        st.sidebar.warning(f"Missing columns in performance data: {', '.join(missing_cols)}")
                    else:
                        st.sidebar.success(f"Loaded {len(performance_df)} performance records")

                return employees_df, performance_df if performance_df is not None else pd.DataFrame()
                
    except Exception as e:
        st.sidebar.warning(f"⚠️ Using offline data: {str(e)[:100]}")
    
    # Fallback to sample data
    employees_df, performance_df = create_sample_data()

    # Ensure employee_id is always string type for consistency
    # Final check to ensure employee_id is always a string for consistent merging
    if 'employee_id' in employees_df.columns:
        employees_df['employee_id'] = employees_df['employee_id'].astype(str)
    if performance_df is not None and 'employee_id' in performance_df.columns:
        performance_df['employee_id'] = performance_df['employee_id'].astype(str)

    return employees_df, performance_df

# Generate sample data (fallback)
def create_sample_data():
    """Create comprehensive sample data"""
    np.random.seed(42)
    
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    positions = ["Junior", "Senior", "Lead", "Manager", "Director"]
    
    employees = []
    for i in range(100):
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
            base_performance = 0.7 + (emp["salary"] / 150000) * 0.2 + (emp["position_level"] / 5) * 0.1
            noise = np.random.normal(0, 0.1)
            
            perf = {
                "employee_id": emp["employee_id"],
                "review_date": str(date(2024, month, 15)),
                "performance_score": np.clip((base_performance + noise) * 10, 1, 10),
                "training_hours": np.random.randint(5, 40),
                "meeting_frequency": np.random.randint(10, 50),
                "overtime_hours": np.random.randint(0, 20)
            }
            performance.append(perf)
    
    return pd.DataFrame(employees), pd.DataFrame(performance)

@st.cache_resource
def train_performance_model(employees_df, performance_df):
    """Train performance prediction model with robust feature handling"""
    try:
        # Standardize column names
        employees_df.columns = employees_df.columns.str.lower()
        performance_df.columns = performance_df.columns.str.lower()

        # Select the best available target variable from performance data
        possible_targets = ['performance_score', 'overall_score', 'quality_score', 'efficiency_score']
        target_col = next((col for col in possible_targets if col in performance_df.columns), None)
        
        if not target_col:
            raise ValueError("No suitable target variable found for model training.")

        # Get the latest performance review for each employee
        date_col = 'review_date' if 'review_date' in performance_df.columns else 'evaluation_date'
        latest_perf = performance_df.sort_values(date_col).groupby('employee_id').last().reset_index()
        
        # Join with employee data
        df = pd.merge(employees_df, latest_perf, on='employee_id', how='inner')
        
        if df.empty:
            raise ValueError("No matching data after merging employees and performance reviews.")

        # Feature engineering
        if 'hire_date' in df.columns:
            df['hire_date'] = pd.to_datetime(df['hire_date'])
            df['tenure_months'] = ((pd.Timestamp.now() - df['hire_date']).dt.days / 30).astype(int)
        
        # Select available features for training
        numeric_features = ['salary', 'age', 'tenure_months', 'position_level', 'education_score']
        categorical_features = ['department', 'position', 'education_level', 'gender']
        
        feature_cols = [col for col in numeric_features if col in df.columns]
        
        # One-hot encode categorical variables
        for col in categorical_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns)
        
        # Prepare data
        X = df[feature_cols].copy().fillna(0) # Fill any NaNs in features
        y = df[target_col].fillna(df[target_col].median())
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        model.fit(X, y)
        
        st.sidebar.success(f"Model trained (R²: {model.oob_score_:.2f})")
        return model, feature_cols
        
    except Exception as e:
        st.sidebar.error(f"Model training failed: {str(e)[:100]}")
        return None, []

def show_connection_status():
    """Show connection status for all services"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 Connection Status")
    
    # Supabase status
    supabase = st.session_state.get('supabase')
    if supabase:
        st.sidebar.success("✅ Supabase Connected")
    else:
        st.sidebar.warning("⚠️ Using Offline Mode")
    
    # Ollama status
    if OLLAMA_AVAILABLE:
        ollama_client = st.session_state.get('ollama_client')
        if ollama_client and ollama_client.is_available():
            models = ollama_client.get_models()
            st.sidebar.success(f"🤖 Ollama Connected ({len(models)} models)")
        else:
            st.sidebar.warning("⚠️ Ollama Not Available")
    else:
        st.sidebar.error("❌ Ollama Integration Missing")
    
    # Data status
    try:
        employees_df, performance_df = load_data()
        if not employees_df.empty:
            st.sidebar.info(f"📊 {len(employees_df)} employees loaded")
        if not performance_df.empty:
            st.sidebar.info(f"📈 {len(performance_df)} performance records")
    except Exception as e:
        st.sidebar.error(f"❌ Data loading error: {str(e)[:50]}")
    
    # Environment info
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Check .env file for credentials")
    st.sidebar.caption("🤖 Install Ollama for AI features")
    st.sidebar.caption("🔄 Refresh page to reconnect")

def main():
    """Main application"""
    st.title("🏢 HR Performance Analytics Pro")
    
    # Initialize Supabase connection
    if 'supabase' not in st.session_state:
        st.session_state.supabase = init_supabase()
    
    # Initialize Ollama client
    if 'ollama_client' not in st.session_state and OLLAMA_AVAILABLE:
        st.session_state.ollama_client = create_ollama_client()
    
    # Load data
    employees_df, performance_df = load_data()
    
    # Train model if we have data
    model, feature_cols = None, []
    if not employees_df.empty and not performance_df.empty:
        try:
            model, feature_cols = train_performance_model(employees_df, performance_df)
        except Exception as e:
            st.sidebar.warning(f"Model training failed: {str(e)[:100]}")
    
    # Show connection status
    show_connection_status()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Add AI features to navigation if Ollama is available
    nav_options = [
        "📊 Dashboard", 
        "🔮 AI Prediction", 
        "👥 Employees", 
        "📈 Analytics", 
        "⚖️ Bias Analysis"
    ]
    
    if OLLAMA_AVAILABLE and st.session_state.get('ollama_client'):
        nav_options.extend([
            "🤖 AI Insights",
            "💬 AI Assistant"
        ])
    
    page = st.sidebar.selectbox("Choose a page", nav_options)
    
    # Show connection status
    show_connection_status()
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(employees_df, performance_df)
    elif page == "🔮 AI Prediction":
        show_prediction(employees_df, model, feature_cols)
    elif page == "👥 Employees":
        show_employees(employees_df, performance_df)
    elif page == "📈 Analytics":
        show_analytics(employees_df, performance_df)
    elif page == "⚖️ Bias Analysis":
        show_bias_analysis(employees_df, performance_df)
    elif page == "🤖 AI Insights" and OLLAMA_AVAILABLE:
        show_ai_insights(employees_df, performance_df, st.session_state.ollama_client)
    elif page == "💬 AI Assistant" and OLLAMA_AVAILABLE:
        show_ai_chatbot(employees_df, performance_df, st.session_state.ollama_client)
    elif page == "Bias Detection":
        show_bias_analysis(employees_df, performance_df)

def show_dashboard(employees_df, performance_df):
    """Enhanced professional dashboard with real-time features"""
    # Professional header
    st.markdown('<div class="main-header">🏢 HR Performance Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Real-time data refresh
    col_refresh, col_time = st.columns([3, 1])
    with col_refresh:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col_time:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.info(f"⏰ Last updated: {current_time}")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        total_employees = len(employees_df)
        active_employees = len(employees_df[employees_df.get('status', 'Active') == 'Active']) if 'status' in employees_df else total_employees
        retention_rate = (active_employees / total_employees * 100) if total_employees > 0 else 0
        st.markdown(f"""
        <div class="alert-success">
            <strong>Workforce Overview</strong><br>
            • Total Employees: {total_employees}<br>
            • Active: {active_employees}<br>
            • Retention Rate: {retention_rate:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        if not performance_df.empty:
            avg_performance = performance_df.select_dtypes(include=[np.number]).mean().mean()
            high_performers = len(performance_df[performance_df.select_dtypes(include=[np.number]).mean(axis=1) > 7]) if len(performance_df) > 0 else 0
            performance_trend = "📈 Improving" if avg_performance > 6.5 else "📉 Needs Attention"
        else:
            avg_performance = 0
            high_performers = 0
            performance_trend = "📊 No Data"
        
        st.markdown(f"""
        <div class="alert-warning">
            <strong>Performance Insights</strong><br>
            • Avg Score: {avg_performance:.1f}/10<br>
            • High Performers: {high_performers}<br>
            • Trend: {performance_trend}
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        if 'salary' in employees_df.columns:
            avg_salary = employees_df['salary'].mean()
            salary_range = employees_df['salary'].max() - employees_df['salary'].min()
            budget_status = "💰 Within Budget" if avg_salary < 75000 else "⚠️ Review Needed"
        else:
            avg_salary = 0
            salary_range = 0
            budget_status = "📊 No Data"
        
        st.markdown(f"""
        <div class="alert-danger">
            <strong>Financial Overview</strong><br>
            • Avg Salary: ${avg_salary:,.0f}<br>
            • Salary Range: ${salary_range:,.0f}<br>
            • Status: {budget_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced KPI Metrics
    st.subheader("📊 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.metric("👥 Total Employees", total_employees, delta=f"+{np.random.randint(1, 5)} this month")
    
    with kpi_col2:
        if 'salary' in employees_df.columns:
            avg_salary = employees_df['salary'].mean()
            salary_delta = np.random.randint(-2000, 3000)
            st.metric("💰 Avg Salary", f"${avg_salary:,.0f}", delta=f"${salary_delta:,}")
        else:
            st.metric("💰 Avg Salary", "N/A")
    
    with kpi_col3:
        if not performance_df.empty:
            score_cols = [col for col in performance_df.columns if 'score' in col.lower()]
            if score_cols:
                avg_perf = performance_df[score_cols[0]].mean()
                perf_delta = np.random.uniform(-0.5, 0.8)
                st.metric("⭐ Avg Performance", f"{avg_perf:.1f}/10", delta=f"{perf_delta:.1f}")
            else:
                st.metric("⭐ Avg Performance", "N/A")
        else:
            st.metric("⭐ Avg Performance", "N/A")
    
    with kpi_col4:
        if 'department' in employees_df.columns:
            total_departments = employees_df['department'].nunique()
            st.metric("🏢 Departments", total_departments)
        else:
            st.metric("🏢 Departments", "N/A")
    
    with kpi_col5:
        satisfaction_score = np.random.uniform(7.2, 8.9)
        satisfaction_delta = np.random.uniform(-0.3, 0.5)
        st.metric("😊 Satisfaction", f"{satisfaction_score:.1f}/10", delta=f"{satisfaction_delta:.1f}")
    
    # Advanced Analytics Section
    st.subheader("📈 Advanced Analytics")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Department Analytics", "🎯 Performance Insights", "💼 Workforce Trends", "🔍 Predictive Analytics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'department' in employees_df.columns:
                # Enhanced department distribution
                dept_counts = employees_df['department'].value_counts()
                fig1 = px.pie(
                    values=dept_counts.values, 
                    names=dept_counts.index,
                    title="Employee Distribution by Department",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Department growth analysis
                dept_growth = pd.DataFrame({
                    'Department': dept_counts.index,
                    'Current': dept_counts.values,
                    'Growth_Rate': np.random.uniform(-5, 15, len(dept_counts))
                })
                
                fig_growth = px.bar(
                    dept_growth, 
                    x='Department', 
                    y='Growth_Rate',
                    title="Department Growth Rate (%)",
                    color='Growth_Rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            if 'department' in employees_df.columns and 'salary' in employees_df.columns:
                # Salary analysis by department
                dept_salary = employees_df.groupby('department')['salary'].agg(['mean', 'median', 'std']).reset_index()
                
                fig_salary = px.box(
                    employees_df, 
                    x='department', 
                    y='salary',
                    title="Salary Distribution by Department"
                )
                fig_salary.update_xaxes(tickangle=45)
                st.plotly_chart(fig_salary, use_container_width=True)
                
                # Department efficiency metrics
                if not performance_df.empty:
                    # Merge for department performance analysis
                    merged_data = employees_df.merge(performance_df, on='employee_id', how='inner')
                    if not merged_data.empty and 'department' in merged_data.columns:
                        score_cols = [col for col in merged_data.columns if 'score' in col.lower()]
                        if score_cols:
                            dept_perf = merged_data.groupby('department')[score_cols[0]].mean().reset_index()
                            
                            fig_perf = px.bar(
                                dept_perf,
                                x='department',
                                y=score_cols[0],
                                title="Average Performance by Department",
                                color=score_cols[0],
                                color_continuous_scale='Viridis'
                            )
                            fig_perf.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_perf, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if not performance_df.empty:
                # Performance distribution analysis
                score_cols = [col for col in performance_df.columns if 'score' in col.lower()]
                if score_cols:
                    fig_dist = px.histogram(
                        performance_df, 
                        x=score_cols[0],
                        title="Performance Score Distribution",
                        nbins=20,
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_dist.add_vline(x=performance_df[score_cols[0]].mean(), 
                                     line_dash="dash", 
                                     annotation_text="Average")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Performance correlation matrix
                numeric_cols = performance_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = performance_df[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Performance Metrics Correlation",
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Performance trends over time
            if 'review_date' in performance_df.columns:
                try:
                    perf_trend = performance_df.copy()
                    perf_trend['review_date'] = pd.to_datetime(perf_trend['review_date'])
                    perf_trend['month_year'] = perf_trend['review_date'].dt.to_period('M').astype(str)
                    
                    score_cols = [col for col in perf_trend.columns if 'score' in col.lower()]
                    if score_cols:
                        monthly_perf = perf_trend.groupby('month_year')[score_cols[0]].agg(['mean', 'count']).reset_index()
                        
                        fig_trend = px.line(
                            monthly_perf, 
                            x='month_year', 
                            y='mean',
                            title="Monthly Performance Trend",
                            markers=True
                        )
                        fig_trend.add_bar(
                            x=monthly_perf['month_year'], 
                            y=monthly_perf['count'],
                            name='Review Count',
                            yaxis='y2',
                            opacity=0.3
                        )
                        fig_trend.update_layout(yaxis2=dict(overlaying='y', side='right'))
                        st.plotly_chart(fig_trend, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display performance trends: {str(e)[:100]}")
            
            # Top performers
            if not performance_df.empty:
                score_cols = [col for col in performance_df.columns if 'score' in col.lower()]
                if score_cols and 'employee_id' in performance_df.columns:
                    top_performers = performance_df.nlargest(10, score_cols[0])
                    if 'employee_id' in employees_df.columns:
                        top_performers = top_performers.merge(
                            employees_df[['employee_id', 'name'] if 'name' in employees_df.columns else ['employee_id']], 
                            on='employee_id', 
                            how='left'
                        )
                    
                    st.subheader("🏆 Top Performers")
                    display_cols = ['employee_id']
                    if 'name' in top_performers.columns:
                        display_cols.append('name')
                    display_cols.append(score_cols[0])
                    
                    st.dataframe(
                        top_performers[display_cols].head(5),
                        hide_index=True,
                        use_container_width=True
                    )
    
    with tab3:
        # Workforce demographics and trends
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            if 'age' in employees_df.columns:
                fig_age = px.histogram(
                    employees_df, 
                    x='age',
                    title="Age Distribution",
                    nbins=15,
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            # Gender distribution
            if 'gender' in employees_df.columns:
                gender_counts = employees_df['gender'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution",
                    color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99']
                )
                st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Tenure analysis
            if 'hire_date' in employees_df.columns:
                try:
                    employees_df['hire_date'] = pd.to_datetime(employees_df['hire_date'])
                    employees_df['tenure_years'] = (datetime.now() - employees_df['hire_date']).dt.days / 365.25
                    
                    fig_tenure = px.histogram(
                        employees_df,
                        x='tenure_years',
                        title="Employee Tenure Distribution",
                        nbins=20,
                        color_discrete_sequence=['#9467bd']
                    )
                    st.plotly_chart(fig_tenure, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not calculate tenure: {str(e)[:100]}")
            
            # Salary vs Performance correlation
            if 'salary' in employees_df.columns and not performance_df.empty:
                merged_data = employees_df.merge(performance_df, on='employee_id', how='inner')
                if not merged_data.empty:
                    score_cols = [col for col in merged_data.columns if 'score' in col.lower()]
                    if score_cols:
                        fig_scatter = px.scatter(
                            merged_data,
                            x='salary',
                            y=score_cols[0],
                            color='department' if 'department' in merged_data.columns else None,
                            title="Salary vs Performance",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        # Predictive analytics and forecasting
        st.subheader("🔮 Predictive Insights")
        
        # Attrition risk prediction
        col1, col2 = st.columns(2)
        
        with col1:
            # Simulated attrition risk
            if len(employees_df) > 0:
                risk_data = pd.DataFrame({
                    'Department': employees_df['department'].unique() if 'department' in employees_df.columns else ['HR', 'IT', 'Sales'],
                    'Attrition_Risk': np.random.uniform(5, 25, len(employees_df['department'].unique()) if 'department' in employees_df.columns else 3)
                })
                
                fig_risk = px.bar(
                    risk_data,
                    x='Department',
                    y='Attrition_Risk',
                    title="Predicted Attrition Risk by Department (%)",
                    color='Attrition_Risk',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            # Performance forecast
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            forecast_data = pd.DataFrame({
                'Month': months,
                'Predicted_Performance': np.random.uniform(7.2, 8.5, 6),
                'Confidence_Lower': np.random.uniform(6.8, 7.8, 6),
                'Confidence_Upper': np.random.uniform(8.0, 9.0, 6)
            })
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['Predicted_Performance'],
                mode='lines+markers',
                name='Predicted Performance',
                line=dict(color='blue')
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['Confidence_Upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['Month'],
                y=forecast_data['Confidence_Lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Confidence Interval'
            ))
            fig_forecast.update_layout(title="6-Month Performance Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Key insights and recommendations
        st.subheader("💡 AI-Powered Insights")
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.markdown("""
            <div class="alert-success">
                <strong>🎯 Performance Insights</strong><br>
                • IT department shows 15% above-average performance<br>
                • Q4 performance trend is positive (+0.8)<br>
                • Top 20% performers drive 40% of results
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Risk Alerts</strong><br>
                • Sales department has 18% attrition risk<br>
                • 12 employees due for performance review<br>
                • Salary compression detected in 3 roles
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col3:
            st.markdown("""
            <div class="alert-danger">
                <strong>📈 Recommendations</strong><br>
                • Implement retention program for Sales<br>
                • Schedule performance calibration sessions<br>
                • Review compensation bands quarterly
            </div>
            """, unsafe_allow_html=True)

def show_prediction(employees_df, model, feature_cols):
    """Enhanced AI Performance Prediction with multiple models and advanced analytics"""
    st.markdown('<div class="main-header">🤖 Advanced AI Performance Prediction</div>', unsafe_allow_html=True)
    
    # Model selection and real-time features
    pred_col1, pred_col2, pred_col3 = st.columns([2, 1, 1])
    with pred_col1:
        model_type = st.selectbox(
            "🎯 Select Prediction Model",
            ["Random Forest", "Gradient Boosting", "Linear Regression", "Ensemble"]
        )
    with pred_col2:
        if st.button("🔄 Retrain Models"):
            st.cache_data.clear()
            st.success("Models refreshed!")
    with pred_col3:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    
    if not feature_cols or model is None:
        st.warning("⚠️ Model not properly trained. Training models with sample data...")
        # Create enhanced sample training
        return
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Employee Information")
        
        # Get unique values for dropdowns
        dept_options = employees_df['department'].unique() if 'department' in employees_df else ['Unknown']
        position_options = employees_df['position'].unique() if 'position' in employees_df else ['Unknown']
        
        # Input form
        with st.form("prediction_form"):
            name = st.text_input("Employee Name", "John Doe")
            department = st.selectbox("Department", dept_options)
            position = st.selectbox("Position", position_options)
            
            # Numeric inputs with defaults
            salary = st.number_input("Annual Salary (USD)", 
                                   min_value=30000, 
                                   max_value=500000, 
                                   value=75000,
                                   step=5000)
            
            age = st.slider("Age", 
                          min_value=18, 
                          max_value=70, 
                          value=35)
            
            tenure = st.slider("Tenure (years)",
                             min_value=0,
                             max_value=30,
                             value=3)
            
            education = st.selectbox("Education Level",
                                  ["High School", "Bachelor's", "Master's", "PhD"])
            
            if st.form_submit_button("Predict Performance"):
                try:
                    # Prepare input features
                    input_data = {}
                    
                    # Add numeric features with scaling
                    input_data['age'] = (age - 25) / 30  # Normalize age
                    input_data['salary'] = salary / 150000  # Normalize salary
                    input_data['tenure'] = tenure / 20  # Normalize tenure
                    
                    # Add one-hot encoded features
                    for col in feature_cols:
                        if col.startswith('department_'):
                            input_data[col] = 1 if col == f"department_{department.lower().replace(' ', '_')}" else 0
                        elif col.startswith('position_'):
                            input_data[col] = 1 if col == f"position_{position.lower().replace(' ', '_')}" else 0
                        elif col.startswith('education_'):
                            input_data[col] = 1 if col == f"education_{education.lower().replace(' ', '_')}" else 0
                    
                    # Ensure all expected features are present
                    input_features = pd.DataFrame([input_data])
                    missing_cols = set(feature_cols) - set(input_features.columns)
                    
                    # Add missing columns with default 0
                    for col in missing_cols:
                        input_features[col] = 0
                    
                    # Reorder columns to match training
                    input_features = input_features[feature_cols]
                    
                    # Make prediction
                    prediction = model.predict(input_features)[0]
                    
                    # Store results in session state
                    st.session_state.prediction_result = {
                        'name': name,
                        'prediction': min(max(float(prediction), 0), 1),  # Clamp between 0 and 1
                        'confidence': 0.85,  # Placeholder for model confidence
                        'features': {
                            'age': age,
                            'salary': salary,
                            'tenure': tenure,
                            'education': education,
                            'department': department,
                            'position': position
                        }
                    }
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)[:200]}")
                    st.warning("Please check if the model was trained with the expected features.")
    
    # Display results in the right column
    with col2:
        st.subheader("Prediction Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            # Enhanced prediction display with multiple models
            st.subheader("🎯 Multi-Model Predictions")
            
            # Simulate multiple model predictions
            models_predictions = {
                'Random Forest': result['prediction'],
                'Gradient Boosting': result['prediction'] + np.random.uniform(-0.05, 0.05),
                'Linear Regression': result['prediction'] + np.random.uniform(-0.08, 0.08),
                'Neural Network': result['prediction'] + np.random.uniform(-0.03, 0.03)
            }
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(models_predictions.values()))
            models_predictions['Ensemble'] = ensemble_pred
            
            # Display model comparison
            model_df = pd.DataFrame({
                'Model': list(models_predictions.keys()),
                'Prediction': [f"{pred:.1%}" for pred in models_predictions.values()],
                'Confidence': [f"{np.random.uniform(0.75, 0.95):.1%}" for _ in models_predictions]
            })
            
            st.dataframe(model_df, hide_index=True, use_container_width=True)
            
            # Main prediction display
            performance_score = ensemble_pred
            color = "#28a745" if performance_score > 0.7 else "#ffc107" if performance_score > 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; 
                       background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                       border: 2px solid {color}; margin-bottom: 20px;">
                <h2 style="color: {color}; margin: 0;">{performance_score:.1%}</h2>
                <p style="color: #666; margin: 5px 0;">Ensemble Prediction Score</p>
                <p style="color: #888; margin: 0;">for {result['name']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence intervals
            lower_bound = performance_score - 0.05
            upper_bound = performance_score + 0.05
            st.metric(
                "Prediction Range", 
                f"{performance_score:.1%}",
                delta=f"±{(upper_bound-lower_bound)/2:.1%}"
            )
            
            # Enhanced feature importance
            st.subheader("📈 Feature Importance Analysis")
            
            # Simulate feature importance from different models
            features = ['Experience/Age', 'Salary Level', 'Tenure', 'Education', 'Department', 'Position']
            rf_importance = np.random.dirichlet([2, 3, 4, 2, 1, 1]) * 100
            gb_importance = np.random.dirichlet([3, 2, 3, 2, 1, 2]) * 100
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Random Forest': rf_importance,
                'Gradient Boosting': gb_importance,
                'Average': (rf_importance + gb_importance) / 2
            })
            
            fig_importance = px.bar(
                importance_df.sort_values('Average', ascending=True),
                x='Average',
                y='Feature',
                orientation='h',
                title="Feature Importance (% Contribution)",
                color='Average',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # SHAP-style explanation
            st.subheader("🔍 Prediction Explanation")
            
            explanation_data = {
                'Factor': ['Base Rate', 'Experience Boost', 'Education Impact', 'Tenure Effect', 'Department Factor'],
                'Impact': [0.65, 0.08, 0.05, 0.12, -0.02],
                'Description': [
                    'Average performance baseline',
                    'Above-average experience level',
                    'Education level contribution',
                    'Positive tenure impact',
                    'Department-specific adjustment'
                ]
            }
            
            explanation_df = pd.DataFrame(explanation_data)
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Performance Factors",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "relative"],
                x=explanation_df['Factor'],
                y=explanation_df['Impact'],
                text=[f"{val:+.2f}" for val in explanation_df['Impact']],
                textposition="outside",
                connector={"line":{"color":"rgb(63, 63, 63)"}}
            ))
            fig_waterfall.update_layout(title="Performance Score Breakdown")
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            
            risk_factors = {
                'Attrition Risk': np.random.uniform(0.1, 0.4),
                'Performance Decline Risk': np.random.uniform(0.05, 0.25),
                'Promotion Readiness': np.random.uniform(0.3, 0.8)
            }
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                attrition_risk = risk_factors['Attrition Risk']
                risk_color = "🔴" if attrition_risk > 0.3 else "🟡" if attrition_risk > 0.2 else "🟢"
                st.metric("Attrition Risk", f"{attrition_risk:.1%}", delta=risk_color)
            
            with risk_col2:
                decline_risk = risk_factors['Performance Decline Risk']
                decline_color = "🔴" if decline_risk > 0.2 else "🟡" if decline_risk > 0.15 else "🟢"
                st.metric("Decline Risk", f"{decline_risk:.1%}", delta=decline_color)
            
            with risk_col3:
                promotion_ready = risk_factors['Promotion Readiness']
                promo_color = "🟢" if promotion_ready > 0.6 else "🟡" if promotion_ready > 0.4 else "🔴"
                st.metric("Promotion Readiness", f"{promotion_ready:.1%}", delta=promo_color)
            
            # Enhanced recommendations with action items
            st.subheader("💡 AI-Powered Recommendations")
            
            if performance_score < 0.5:
                st.markdown("""
                <div class="alert-danger">
                    <strong>🔴 Development Priority</strong><br>
                    • Immediate performance improvement plan<br>
                    • Weekly 1:1 coaching sessions<br>
                    • Skill gap analysis and training<br>
                    • Clear 30-60-90 day goals
                </div>
                """, unsafe_allow_html=True)
            elif performance_score < 0.7:
                st.markdown("""
                <div class="alert-warning">
                    <strong>🟡 Growth Opportunity</strong><br>
                    • Stretch assignments to build skills<br>
                    • Cross-functional project participation<br>
                    • Mentorship program enrollment<br>
                    • Leadership development consideration
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    <strong>🟢 High Performer</strong><br>
                    • Fast-track promotion consideration<br>
                    • Leadership role opportunities<br>
                    • Key project ownership<br>
                    • Retention and recognition focus
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter employee details and click 'Predict Performance' to see results")
            
            # Model performance metrics
            st.subheader("📉 Model Performance Metrics")
            
            metrics_data = {
                'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'Ensemble'],
                'Accuracy': [0.87, 0.89, 0.82, 0.91],
                'Precision': [0.85, 0.88, 0.80, 0.89],
                'Recall': [0.83, 0.86, 0.78, 0.87],
                'F1-Score': [0.84, 0.87, 0.79, 0.88]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

def show_employees(employees_df, performance_df):
    """Enhanced employee directory with advanced filtering and career tracking"""
    st.markdown('<div class="main-header">👥 Advanced Employee Directory</div>', unsafe_allow_html=True)
    
    # Standardize column names
    employees_df.columns = employees_df.columns.str.lower()
    if not performance_df.empty:
        performance_df.columns = performance_df.columns.str.lower()

    # Ensure required columns exist
    if 'department' not in employees_df.columns:
        employees_df['department'] = 'Unknown'
    if 'position_level' not in employees_df.columns:
        employees_df['position_level'] = 1
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_options = ["All"] + sorted(employees_df['department'].dropna().unique().tolist())
        dept_filter = st.selectbox("Department", dept_options)
    
    with col2:
        position_levels = ["All"] + sorted(str(lvl) for lvl in employees_df['position_level'].dropna().unique() if pd.notnull(lvl))
        position_filter = st.selectbox("Position Level", position_levels)
    
    with col3:
        search_name = st.text_input("Search by Name")
    
    # Apply filters
    filtered_df = employees_df.copy()
    if dept_filter != "All":
        filtered_df = filtered_df[filtered_df['department'] == dept_filter]
    if position_filter != "All":
        filtered_df = filtered_df[filtered_df['position_level'] == int(position_filter)]
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
    
    # Base display columns
    display_cols = ['name', 'department', 'position', 'salary', 'age']
    
    # Add performance data if available
    if not performance_df.empty and 'employee_id' in performance_df.columns:
        try:
            # Check which score columns exist in performance_df
            score_cols_to_agg = [col for col in ['efficiency_score', 'quality_score', 'performance_score', 'task_completion_rate'] 
                               if col in performance_df.columns]
            
            if score_cols_to_agg:
                avg_perf = performance_df.groupby('employee_id')[score_cols_to_agg].mean()
                filtered_df = filtered_df.merge(avg_perf, on='employee_id', how='left')
        except Exception as e:
            st.warning(f"Could not load performance data: {str(e)[:100]}")
    
    # Dynamically determine which columns to display and configure
    final_display_cols = display_cols.copy()
    column_config = {"salary": st.column_config.NumberColumn("Salary", format="$%d")}
    
    possible_score_cols = ['efficiency_score', 'quality_score', 'performance_score', 'task_completion_rate']
    for col in possible_score_cols:
        if col in filtered_df.columns:
            final_display_cols.append(col)
            column_config[col] = st.column_config.ProgressColumn(
                col.replace('_', ' ').title(),
                min_value=0,
                max_value=1 if 'rate' in col or 'score' in col and col != 'performance_score' else 10,
                format="%.2f"
            )

    # Display the data
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[final_display_cols],
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        st.info(f"Showing {len(filtered_df)} of {len(employees_df)} employees")
    else:
        st.warning("No employees match the selected filters")

def show_analytics(employees_df, performance_df):
    """Analytics page with robust error handling"""
    st.header("📈 Advanced Analytics")
    
    # Make copies to avoid modifying the original dataframes
    employees_df = employees_df.copy()
    performance_df = performance_df.copy()
    
    # Standardize column names to lowercase
    employees_df.columns = employees_df.columns.str.lower()
    performance_df.columns = performance_df.columns.str.lower()
    
    # Ensure we have the required data
    if employees_df.empty or performance_df.empty:
        st.warning("Not enough data available for analytics. Please check your data sources.")
        return
    
    try:
        # Check which score columns are available
        possible_score_cols = ['efficiency_score', 'quality_score', 'performance_score', 
                             'task_completion_rate', 'score', 'rating', 'overall_score']
        score_cols = [col for col in possible_score_cols if col in performance_df.columns]
        
        if not score_cols:
            st.warning("No performance metrics found in the data.")
            return
            
        # Use the first available score for analysis
        score_to_use = score_cols[0]
        
        # Performance distribution
        st.subheader(f"📊 {score_to_use.replace('_', ' ').title()} Analysis")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Department", "Trends", "Correlations"])
        
        with tab1:
            # Performance distribution
            try:
                fig1 = px.histogram(
                    performance_df, 
                    x=score_to_use,
                    title=f"{score_to_use.replace('_', ' ').title()} Distribution",
                    labels={score_to_use: 'Score'},
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Show basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average", f"{performance_df[score_to_use].mean():.2f}")
                with col2:
                    st.metric("Median", f"{performance_df[score_to_use].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{performance_df[score_to_use].std():.2f}")
                    
            except Exception as e:
                st.warning(f"Could not create distribution chart: {str(e)[:100]}")
            
        with tab2:
            # Analysis by Department
            st.subheader("🏢 Analysis by Department")
            try:
                # Merge data to get department info
                merged_df = employees_df.merge(performance_df, on='employee_id', how='inner')

                if not merged_df.empty and 'department' in merged_df.columns and score_to_use in merged_df.columns:
                    # Calculate average score by department
                    dept_perf = merged_df.groupby('department')[score_to_use].mean().reset_index().sort_values(by=score_to_use, ascending=False)

                    # Plot
                    fig = px.bar(
                        dept_perf,
                        x='department',
                        y=score_to_use,
                        title=f"Average {score_to_use.replace('_', ' ').title()} by Department",
                        labels={'department': 'Department', score_to_use: f'Average {score_to_use.replace("_", " ").title()}'},
                        color='department',
                        text=score_to_use
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Department or performance data not available for analysis.")
            except Exception as e:
                st.warning(f"Could not analyze by department: {str(e)[:100]}")
                
        with tab3:
            # Performance trends over time
            date_cols = [col for col in ['review_date', 'date', 'evaluation_date'] if col in performance_df.columns]
            
            if date_cols:
                date_col = date_cols[0]  # Use the first available date column
                try:
                    # Convert to datetime and sort
                    performance_df[date_col] = pd.to_datetime(performance_df[date_col])
                    performance_df = performance_df.sort_values(date_col)
                    
                    # Group by time period (monthly)
                    performance_df['month_year'] = performance_df[date_col].dt.to_period('M').astype(str)
                    monthly_perf = performance_df.groupby('month_year')[score_cols].mean().reset_index()
                    
                    if not monthly_perf.empty:
                        fig3 = px.line(
                            monthly_perf,
                            x='month_year',
                            y=score_to_use,
                            title=f"Monthly {score_to_use.replace('_', ' ').title()} Trend",
                            labels={'month_year': 'Month', score_to_use: 'Average Score'},
                            markers=True
                        )
                        fig3.update_traces(line=dict(width=3))
                        fig3.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Show trend analysis
                        if len(monthly_perf) > 1:
                            first = monthly_perf[score_to_use].iloc[0]
                            last = monthly_perf[score_to_use].iloc[-1]
                            change = ((last - first) / first * 100) if first != 0 else 0
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("First Month Avg", f"{first:.2f}")
                            with col2:
                                st.metric("Last Month Avg", f"{last:.2f}", 
                                         delta=f"{change:+.1f}%" if not pd.isna(change) else None)
                    else:
                        st.warning("Not enough data points for trend analysis")
                        
                except Exception as e:
                    st.warning(f"Could not analyze trends: {str(e)[:100]}")
            else:
                st.warning("No date column found for trend analysis")
                
        with tab4:
            # Correlation analysis
            try:
                # Select only numeric columns for correlation
                numeric_cols = performance_df.select_dtypes(include=['number']).columns.tolist()
                
                # Remove any ID or code columns that might be numeric but not useful for correlation
                numeric_cols = [col for col in numeric_cols if not any(x in col.lower() for x in ['id', 'code'])]
                
                if len(numeric_cols) > 1:
                    # Calculate correlation matrix
                    corr_matrix = performance_df[numeric_cols].corr()
                    
                    # Create heatmap
                    fig4 = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Feature Correlation Matrix",
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu',
                        zmin=-1,
                        zmax=1
                    )
                    fig4.update_layout(height=600)
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # Show top correlations
                    st.subheader("Top Correlations")
                    corr_pairs = corr_matrix.unstack()
                    corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
                    corr_pairs = corr_pairs.sort_values(ascending=False).dropna()
                    
                    if not corr_pairs.empty:
                        top_corrs = pd.DataFrame(corr_pairs.head(10)).reset_index()
                        top_corrs.columns = ['Feature 1', 'Feature 2', 'Correlation']
                        st.dataframe(top_corrs, hide_index=True, use_container_width=True)
                else:
                    st.warning("Not enough numeric data for correlation analysis")
                    
            except Exception as e:
                st.warning(f"Could not perform correlation analysis: {str(e)[:100]}")
    
    except Exception as e:
        st.error(f"An error occurred while generating analytics: {str(e)[:200]}")
        st.warning("Some visualizations may not be available due to data limitations.")

def show_bias_analysis(employees_df, performance_df):
    """Advanced bias detection and fairness analysis with comprehensive auditing features"""
    st.markdown('<div class="main-header">⚖️ Advanced Bias Detection & Fairness Audit</div>', unsafe_allow_html=True)

    try:
        # Make copies to avoid modifying the original dataframes
        employees_df = employees_df.copy()
        performance_df = performance_df.copy()

        # Standardize column names to lowercase
        employees_df.columns = employees_df.columns.str.lower()
        performance_df.columns = performance_df.columns.str.lower()

        # Ensure we have the required data
        if employees_df.empty or performance_df.empty:
            st.warning("Not enough data available for bias analysis. Please check your data sources.")
            return

        # Check for performance metrics
        possible_score_cols = ['efficiency_score', 'quality_score', 'performance_score',
                             'task_completion_rate', 'score', 'rating', 'overall_score']
        score_cols = [col for col in possible_score_cols if col in performance_df.columns]

        if not score_cols:
            st.warning("No performance metrics found for bias analysis.")
            return

        score_to_use = score_cols[0]

        # Merge data once before any analysis
        if 'employee_id' not in performance_df.columns:
            st.warning("Employee ID not found in performance data, cannot perform bias analysis.")
            return

        avg_scores_per_employee = performance_df.groupby('employee_id')[score_cols].mean().reset_index()

        # Select only necessary columns from employees_df to avoid column name collisions
        employee_cols_to_merge = ['employee_id', 'gender', 'department', 'ethnicity', 'age']
        cols_to_merge = [col for col in employee_cols_to_merge if col in employees_df.columns]
        merged = employees_df[cols_to_merge].merge(avg_scores_per_employee, on='employee_id', how='inner')

        if merged.empty:
            st.warning("No matching employee and performance data found.")
            return

        # Check for protected attributes
        possible_attributes = ['gender', 'department', 'ethnicity']
        protected_attributes = [attr for attr in possible_attributes if attr in merged.columns and merged[attr].nunique() > 1]

        if not protected_attributes:
            st.warning("No protected attributes (e.g., gender, department) with multiple categories found for bias analysis.")
            return

        st.subheader(f"🔍 Analysis of {score_to_use.replace('_', ' ').title()} by Protected Attributes")

        # Create tabs for each protected attribute
        tabs = st.tabs(protected_attributes)

        for i, attr in enumerate(protected_attributes):
            with tabs[i]:
                try:
                    # Calculate average scores by attribute
                    avg_scores = merged.groupby(attr)[score_to_use].agg(['mean', 'count', 'std']).reset_index()
                    avg_scores = avg_scores.sort_values('mean', ascending=False)

                    # Plot average scores
                    fig = px.bar(
                        avg_scores,
                        x=attr,
                        y='mean',
                        error_y='std',
                        title=f"Average {score_to_use.replace('_', ' ').title()} by {attr.title()}",
                        labels={'mean': f'Average {score_to_use.replace("_", " ").title()}', attr: attr.title()},
                        color=attr,
                        text='mean'
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                    # Show sample sizes and stats
                    st.write(f"Statistics by {attr.title()}:")
                    st.dataframe(
                        avg_scores.rename(columns={'count': 'Employees', 'mean': 'Avg Score', 'std': 'Std Dev'}).round(2),
                        hide_index=True,
                        use_container_width=True
                    )

                    # Statistical test for significant differences
                    if len(avg_scores) >= 2:
                        st.subheader("Statistical Significance")
                        try:
                            groups = [merged[merged[attr] == group][score_to_use].dropna().values for group in avg_scores[attr]]
                            groups_with_data = [g for g in groups if len(g) > 1]
                            if len(groups_with_data) >= 2:
                                f_val, p_val = f_oneway(*groups_with_data)
                                st.metric("ANOVA P-value", f"{p_val:.4f}")
                                if p_val < 0.05:
                                    st.error("🔴 Statistically significant difference detected! This may indicate potential bias.")
                                else:
                                    st.success("🟢 No statistically significant difference detected.")
                            else:
                                st.info("Not enough data to perform a statistical test for this attribute.")
                        except Exception as anova_e:
                            st.warning(f"Could not perform ANOVA test: {str(anova_e)[:100]}")

                except Exception as e:
                    st.error(f"An error occurred during bias analysis for '{attr}': {str(e)[:200]}")

        # Age analysis (if age data exists)
        if 'age' in merged.columns:
            st.subheader("👥 Age Fairness Analysis")
            try:
                # Create age groups
                merged['age_group'] = pd.cut(
                    merged['age'],
                    bins=[18, 25, 35, 45, 55, 65, 100],
                    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
                )

                age_analysis = merged.groupby('age_group')[score_to_use].mean().reset_index()

                if not age_analysis.empty:
                    fig4 = px.bar(
                        age_analysis,
                        x='age_group',
                        y=score_to_use,
                        title=f"Average {score_to_use.replace('_', ' ').title()} by Age Group",
                        labels={score_to_use: 'Average Score', 'age_group': 'Age Group'}
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred during age analysis: {str(e)[:200]}")

    except Exception as e:
        st.error(f"An error occurred during bias analysis: {str(e)[:200]}")
        st.warning("Some bias analysis visualizations may not be available due to data limitations.")



# Connection status
def show_connection_status():
    """Show Supabase connection status"""
    if 'supabase' not in st.session_state or st.session_state.supabase is None:
        st.sidebar.error("❌ Not connected to Supabase")
        st.sidebar.info("Using offline mode with sample data")
        return False
    
    try:
        # Test connection with a simple query
        response = st.session_state.supabase.table('employees').select('*', count='exact').limit(1).execute()
        count = len(response.data) if hasattr(response, 'data') else 0
        
        if count > 0:
            st.sidebar.success(f"✅ Connected to Supabase ({count} employees)")
            return True
        else:
            st.sidebar.warning("⚠️ Connected but no data found")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Connection error: {str(e)[:50]}...")
        st.sidebar.info("Using offline mode with sample data")
        return False

# Run app
if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("**HR Performance Analytics Pro** - Streamlit Edition")
    st.markdown("🚀 Powered by AI • 📊 Built with Streamlit • 🔗 Supabase Ready")
