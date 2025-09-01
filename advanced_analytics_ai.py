import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from ollama_ai_service import ai_service
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def render_advanced_analytics_dashboard():
    """Main Advanced Analytics & AI Dashboard"""
    st.title("🧠 Advanced Analytics & AI")
    
    # Initialize session state
    if 'ai_models' not in st.session_state:
        st.session_state.ai_models = {}
    if 'analytics_data' not in st.session_state:
        st.session_state.analytics_data = {}
    
    # Sidebar navigation
    st.sidebar.markdown("### 🧠 AI Analytics")
    analytics_page = st.sidebar.radio("Select Module", [
        "📊 AI Dashboard",
        "📉 Attrition Prediction",
        "🎯 Skill Gap Analysis",
        "⭐ Performance Recommendations",
        "🔄 Succession Planning",
        "🔍 Explainable AI",
        "⚙️ Model Management"
    ])
    
    if analytics_page == "📊 AI Dashboard":
        render_ai_dashboard_overview()
    elif analytics_page == "📉 Attrition Prediction":
        render_attrition_prediction()
    elif analytics_page == "🎯 Skill Gap Analysis":
        render_skill_gap_analysis()
    elif analytics_page == "⭐ Performance Recommendations":
        render_performance_recommendations()
    elif analytics_page == "🔄 Succession Planning":
        render_succession_forecasts()
    elif analytics_page == "🔍 Explainable AI":
        render_explainable_ai()
    elif analytics_page == "⚙️ Model Management":
        render_model_management()

def render_ai_dashboard_overview():
    """AI Analytics Dashboard Overview"""
    st.header("📊 AI Analytics Overview")
    
    # AI Model Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Attrition Risk", "12%", "↓ 3%")
    with col2:
        st.metric("Skill Gaps", "45", "↓ 8")
    with col3:
        st.metric("Model Accuracy", "94.2%", "↑ 1.5%")
    with col4:
        st.metric("Predictions Made", "1,247", "↑ 156")
    
    # AI Insights Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top AI Insights")
        insights = [
            {"insight": "Engineering team shows 18% attrition risk", "confidence": 0.87, "action": "Retention program recommended"},
            {"insight": "Python skills gap identified in 23 employees", "confidence": 0.92, "action": "Training courses suggested"},
            {"insight": "Sarah Johnson ready for promotion", "confidence": 0.89, "action": "Succession plan activated"},
            {"insight": "Q4 performance dip predicted", "confidence": 0.78, "action": "Intervention recommended"}
        ]
        
        for insight in insights:
            with st.expander(f"💡 {insight['insight']} (Confidence: {insight['confidence']:.2f})"):
                st.write(f"**Recommended Action:** {insight['action']}")
                if st.button("Take Action", key=f"action_{insight['insight'][:10]}"):
                    st.success("Action initiated!")
    
    with col2:
        st.subheader("📈 Model Performance")
        
        models = ['Attrition', 'Performance', 'Succession', 'Skill Gap']
        accuracies = [94.2, 91.8, 88.5, 96.1]
        
        fig = px.bar(x=models, y=accuracies,
                    title="AI Model Accuracy Scores",
                    color=accuracies, color_continuous_scale="viridis")
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Target: 85%")
        st.plotly_chart(fig, use_container_width=True)

def render_attrition_prediction():
    """Attrition Risk Prediction"""
    st.header("📉 Attrition Risk Prediction")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Risk Assessment", "📊 Model Training", "📈 Predictions"])
    
    with tab1:
        st.subheader("Employee Attrition Risk Assessment")
        
        # Generate sample employee risk data
        employees = [f"Employee {i}" for i in range(1, 21)]
        risk_scores = np.random.uniform(0.1, 0.9, len(employees))
        departments = np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], len(employees))
        
        risk_df = pd.DataFrame({
            'Employee': employees,
            'Department': departments,
            'Risk Score': risk_scores,
            'Risk Level': ['High' if r > 0.7 else 'Medium' if r > 0.4 else 'Low' for r in risk_scores]
        })
        
        # High-risk employees
        high_risk = risk_df[risk_df['Risk Level'] == 'High']
        if not high_risk.empty:
            st.subheader("🚨 High-Risk Employees")
            st.dataframe(high_risk, use_container_width=True)
        
        # Risk distribution
        fig = px.histogram(risk_df, x='Risk Score', color='Department',
                          title="Attrition Risk Distribution by Department")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Attrition Model Training")
        
        if st.button("Train Attrition Model"):
            with st.spinner("Training XGBoost attrition model..."):
                # Simulate model training
                accuracy, feature_importance = train_attrition_model()
                
                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}%")
                
                # Feature importance
                st.write("**Top Risk Factors:**")
                importance_df = pd.DataFrame({
                    'Feature': ['Salary Satisfaction', 'Manager Rating', 'Work-Life Balance', 'Career Growth', 'Job Satisfaction'],
                    'Importance': feature_importance
                })
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title="Feature Importance for Attrition Prediction")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Attrition Predictions")
        
        # Department-wise predictions
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        predicted_attrition = np.random.uniform(8, 25, len(departments))
        
        fig = px.bar(x=departments, y=predicted_attrition,
                    title="Predicted Attrition Rate by Department (%)",
                    color=predicted_attrition, color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
        
        # Retention recommendations
        st.write("**AI Retention Recommendations:**")
        recommendations = [
            "Implement flexible work arrangements for Engineering team",
            "Increase professional development budget for Sales team",
            "Conduct stay interviews with high-risk employees",
            "Review compensation packages for top performers"
        ]
        
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")

def render_skill_gap_analysis():
    """Skill Gap Detection & Analysis"""
    st.header("🎯 Skill Gap Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Gap Overview", "🔍 Individual Analysis", "📚 Training Suggestions"])
    
    with tab1:
        st.subheader("Organization-wide Skill Gaps")
        
        # Skill gap heatmap
        skills = ['Python', 'Machine Learning', 'Project Management', 'Data Analysis', 'Leadership']
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        
        gap_matrix = np.random.uniform(0.2, 0.8, (len(departments), len(skills)))
        
        fig = px.imshow(gap_matrix, x=skills, y=departments,
                       title="Skill Gap Intensity by Department",
                       color_continuous_scale="Reds",
                       labels={'color': 'Gap Severity'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Critical skill gaps
        st.write("**Critical Skill Gaps:**")
        critical_gaps = [
            {"skill": "Machine Learning", "gap_size": 23, "priority": "High", "impact": "Product Development"},
            {"skill": "Data Analysis", "gap_size": 18, "priority": "High", "impact": "Decision Making"},
            {"skill": "Leadership", "gap_size": 15, "priority": "Medium", "impact": "Team Management"}
        ]
        
        gap_df = pd.DataFrame(critical_gaps)
        st.dataframe(gap_df, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Individual Skill Analysis")
        
        employee_select = st.selectbox("Select Employee", 
                                     [f"Employee {i}" for i in range(1, 21)], 
                                     key="skill_employee")
        
        # Current skills vs required
        current_skills = ['Python', 'SQL', 'Communication']
        required_skills = ['Python', 'SQL', 'Machine Learning', 'Leadership', 'Communication']
        
        skill_levels = {
            'Python': {'current': 8, 'required': 9},
            'SQL': {'current': 7, 'required': 8},
            'Machine Learning': {'current': 3, 'required': 7},
            'Leadership': {'current': 5, 'required': 8},
            'Communication': {'current': 9, 'required': 8}
        }
        
        # Skill gap visualization
        skills_list = list(skill_levels.keys())
        current_levels = [skill_levels[s]['current'] for s in skills_list]
        required_levels = [skill_levels[s]['required'] for s in skills_list]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current Level', x=skills_list, y=current_levels))
        fig.add_trace(go.Bar(name='Required Level', x=skills_list, y=required_levels))
        fig.update_layout(title=f"Skill Analysis for {employee_select}", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📚 AI-Powered Training Suggestions")
        
        if st.button("Generate Training Recommendations"):
            with st.spinner("Generating AI-powered training recommendations..."):
                # Use Ollama for skill gap analysis
                context = f"Employee skills: {', '.join(['Python', 'SQL', 'Communication'])}. Required skills: Python, Machine Learning, Leadership"
                recommendations = ai_service.generate_recommendations(context, "training")
                
                st.write("**AI-Powered Training Recommendations:**")
                for i, rec in enumerate(recommendations[:3]):
                    with st.expander(f"📚 Recommendation {i+1}"):
                        st.write(rec)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Priority:** High")
                            st.write(f"**Duration:** 4-6 weeks")
                        with col2:
                            st.write(f"**Match Score:** {np.random.uniform(0.8, 0.95):.2f}")
                            if st.button("Enroll", key=f"enroll_ai_{i}"):
                                st.success("Enrolled in course!")

def render_performance_recommendations():
    """AI-Driven Performance Recommendations"""
    st.header("⭐ Performance Recommendations")
    
    tab1, tab2 = st.tabs(["🎯 Individual Recommendations", "📊 Team Insights"])
    
    with tab1:
        employee_select = st.selectbox("Select Employee", 
                                     [f"Employee {i}" for i in range(1, 21)], 
                                     key="perf_employee")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Performance", "87%", "↑ 5%")
        with col2:
            st.metric("Goal Achievement", "92%", "↑ 8%")
        with col3:
            st.metric("Improvement Potential", "15%", "")
        
        # AI recommendations using Ollama
        st.subheader("🤖 AI Performance Recommendations")
        
        if st.button("Generate AI Recommendations"):
            with st.spinner("Generating personalized recommendations..."):
                context = f"Employee performance: 87%, Goal achievement: 92%, Areas for improvement needed"
                recommendations = ai_service.generate_recommendations(context, "performance improvement")
                
                for i, rec in enumerate(recommendations[:3]):
                    with st.expander(f"💡 AI Recommendation {i+1}"):
                        st.write(rec)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Impact:** {np.random.choice(['High', 'Medium', 'Low'])}")
                        with col2:
                            st.write(f"**Effort:** {np.random.choice(['Low', 'Medium', 'High'])}")
                        if st.button("Implement", key=f"impl_ai_{i}"):
                            st.success("Recommendation added to development plan!")
    
    with tab2:
        st.subheader("📊 Team Performance Insights")
        
        # Team performance distribution
        teams = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        avg_performance = np.random.uniform(75, 95, len(teams))
        
        fig = px.bar(x=teams, y=avg_performance,
                    title="Average Team Performance Scores",
                    color=avg_performance, color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance improvement opportunities
        st.write("**Team Improvement Opportunities:**")
        opportunities = [
            {"team": "Engineering", "opportunity": "Code review process optimization", "potential_gain": "8%"},
            {"team": "Sales", "opportunity": "CRM training and automation", "potential_gain": "12%"},
            {"team": "Marketing", "opportunity": "Data analytics upskilling", "potential_gain": "15%"}
        ]
        
        opp_df = pd.DataFrame(opportunities)
        st.dataframe(opp_df, use_container_width=True)

def render_succession_forecasts():
    """AI-Powered Succession Planning"""
    st.header("🔄 Succession Planning Forecasts")
    
    tab1, tab2 = st.tabs(["👑 Leadership Pipeline", "📈 Readiness Prediction"])
    
    with tab1:
        st.subheader("Leadership Pipeline Analysis")
        
        # Succession readiness matrix
        positions = ['VP Engineering', 'Sales Director', 'Marketing Manager', 'HR Director']
        candidates = ['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Wang']
        
        readiness_scores = np.random.uniform(0.4, 0.95, (len(positions), len(candidates)))
        
        fig = px.imshow(readiness_scores, x=candidates, y=positions,
                       title="Succession Readiness Matrix",
                       color_continuous_scale="RdYlGn",
                       labels={'color': 'Readiness Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top succession candidates
        st.write("**Top Succession Candidates:**")
        succession_data = [
            {"candidate": "Sarah Johnson", "target_role": "VP Engineering", "readiness": "92%", "timeline": "6 months"},
            {"candidate": "Mike Chen", "target_role": "Sales Director", "readiness": "87%", "timeline": "9 months"},
            {"candidate": "Lisa Wang", "target_role": "HR Director", "readiness": "89%", "timeline": "12 months"}
        ]
        
        succession_df = pd.DataFrame(succession_data)
        st.dataframe(succession_df, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Promotion Readiness Prediction")
        
        if st.button("Run Succession Analysis"):
            with st.spinner("Analyzing succession readiness..."):
                # Simulate succession analysis
                analysis_results = analyze_succession_readiness()
                
                st.write("**Succession Analysis Results:**")
                for result in analysis_results:
                    st.write(f"• **{result['employee']}**: {result['prediction']} (Confidence: {result['confidence']:.2f})")

def render_explainable_ai():
    """Explainable AI Dashboard"""
    st.header("🔍 Explainable AI")
    
    tab1, tab2 = st.tabs(["📊 SHAP Analysis", "🔍 LIME Explanations"])
    
    with tab1:
        st.subheader("📊 SHAP Feature Importance")
        
        model_select = st.selectbox("Select Model", 
                                  ["Attrition Prediction", "Performance Prediction", "Succession Planning"], 
                                  key="shap_model")
        
        if st.button("Generate AI Analysis"):
            with st.spinner("Generating AI explanations..."):
                # Use Ollama for feature importance analysis
                context = f"Analyzing {model_select} model for feature importance"
                recommendations = ai_service.generate_recommendations(context, "feature importance")
                
                # Simulate feature importance visualization
                features = ['Salary', 'Manager Rating', 'Work-Life Balance', 'Career Growth', 'Job Satisfaction']
                importance_values = np.random.uniform(-0.3, 0.3, len(features))
                
                fig = px.bar(x=importance_values, y=features, orientation='h',
                            title=f"AI Feature Importance - {model_select}",
                            color=importance_values, color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**AI Insights:**")
                for rec in recommendations[:3]:
                    st.write(f"• {rec}")
    
    with tab2:
        st.subheader("🔍 LIME Local Explanations")
        
        employee_select = st.selectbox("Select Employee for Explanation", 
                                     [f"Employee {i}" for i in range(1, 11)], 
                                     key="lime_employee")
        
        if st.button("Generate LIME Explanation"):
            st.write(f"**Local Explanation for {employee_select}:**")
            
            explanations = [
                {"feature": "Low salary compared to market", "impact": "+0.25", "direction": "Increases attrition risk"},
                {"feature": "High manager rating", "impact": "-0.18", "direction": "Decreases attrition risk"},
                {"feature": "Limited career growth", "impact": "+0.15", "direction": "Increases attrition risk"}
            ]
            
            for exp in explanations:
                direction_color = "🔴" if "Increases" in exp['direction'] else "🟢"
                st.write(f"{direction_color} **{exp['feature']}**: {exp['impact']} - {exp['direction']}")

def render_model_management():
    """AI Model Management"""
    st.header("⚙️ AI Model Management")
    
    tab1, tab2 = st.tabs(["📊 Model Status", "🔄 Model Training"])
    
    with tab1:
        st.subheader("AI Model Status")
        
        models = [
            {"name": "Attrition Prediction", "version": "v2.1", "accuracy": "94.2%", "last_trained": "2024-08-15", "status": "Active"},
            {"name": "Performance Prediction", "version": "v1.8", "accuracy": "91.8%", "last_trained": "2024-08-10", "status": "Active"},
            {"name": "Succession Planning", "version": "v1.5", "accuracy": "88.5%", "last_trained": "2024-08-05", "status": "Training"},
            {"name": "Skill Gap Detection", "version": "v2.0", "accuracy": "96.1%", "last_trained": "2024-08-12", "status": "Active"}
        ]
        
        model_df = pd.DataFrame(models)
        st.dataframe(model_df, use_container_width=True)
        
        # Model performance trends
        dates = pd.date_range(start='2024-01-01', periods=8, freq='M')
        accuracy_trends = {
            'Attrition': np.random.uniform(90, 95, 8),
            'Performance': np.random.uniform(88, 93, 8),
            'Succession': np.random.uniform(85, 90, 8)
        }
        
        fig = go.Figure()
        for model, accuracies in accuracy_trends.items():
            fig.add_trace(go.Scatter(x=dates, y=accuracies, mode='lines+markers', name=model))
        
        fig.update_layout(title="Model Accuracy Trends", yaxis_title="Accuracy %")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔄 Model Training & Deployment")
        
        col1, col2 = st.columns(2)
        with col1:
            model_to_train = st.selectbox("Select Model", 
                                        ["Attrition Prediction", "Performance Prediction", "Succession Planning"], 
                                        key="train_model")
            training_data_size = st.slider("Training Data Size", 1000, 10000, 5000)
        
        with col2:
            algorithm = st.selectbox("Algorithm", ["XGBoost", "Random Forest", "Gradient Boosting"], key="algorithm")
            cross_validation = st.checkbox("Cross Validation", value=True)
        
        if st.button("Start Training"):
            with st.spinner(f"Training {model_to_train} model..."):
                # Simulate training
                st.success(f"✅ {model_to_train} model trained successfully!")
                st.info(f"New accuracy: {np.random.uniform(92, 97):.1f}%")

def train_attrition_model():
    """Train attrition prediction model"""
    # Simulate model training
    accuracy = np.random.uniform(92, 96)
    feature_importance = np.random.uniform(0.1, 0.4, 5)
    feature_importance = feature_importance / feature_importance.sum()
    
    return accuracy, feature_importance

def generate_skill_recommendations():
    """Generate AI-powered skill recommendations"""
    recommendations = [
        {"course": "Advanced Python Programming", "provider": "Coursera", "skill": "Python", 
         "level": "Advanced", "duration": "6 weeks", "match_score": 0.92, "priority": "High"},
        {"course": "Machine Learning Fundamentals", "provider": "edX", "skill": "ML", 
         "level": "Beginner", "duration": "8 weeks", "match_score": 0.88, "priority": "High"},
        {"course": "Leadership Essentials", "provider": "LinkedIn Learning", "skill": "Leadership", 
         "level": "Intermediate", "duration": "4 weeks", "match_score": 0.85, "priority": "Medium"}
    ]
    
    return recommendations

def analyze_succession_readiness():
    """Analyze succession readiness using AI"""
    employees = ['Sarah Johnson', 'Mike Chen', 'Lisa Wang', 'Alex Brown']
    
    results = []
    for employee in employees:
        confidence = np.random.uniform(0.7, 0.95)
        prediction = "Ready for promotion" if confidence > 0.8 else "Needs development"
        
        results.append({
            'employee': employee,
            'prediction': prediction,
            'confidence': confidence
        })
    
    return results

if __name__ == "__main__":
    render_advanced_analytics_dashboard()
