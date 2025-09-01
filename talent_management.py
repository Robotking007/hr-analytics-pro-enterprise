"""
Talent Management System
AI-powered talent management with succession planning, OKR tracking, career path recommendations, and promotion prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class TalentManager:
    def __init__(self):
        self.career_paths = {
            'Engineering': ['Junior Developer', 'Developer', 'Senior Developer', 'Lead Developer', 'Engineering Manager', 'Director of Engineering'],
            'Sales': ['Sales Associate', 'Sales Representative', 'Senior Sales Rep', 'Sales Manager', 'Regional Manager', 'VP Sales'],
            'Marketing': ['Marketing Coordinator', 'Marketing Specialist', 'Senior Specialist', 'Marketing Manager', 'Director Marketing', 'CMO'],
            'HR': ['HR Coordinator', 'HR Specialist', 'HR Business Partner', 'HR Manager', 'HR Director', 'CHRO'],
            'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'Senior Manager', 'Finance Director', 'CFO']
        }
        
        self.skills_database = {
            'technical': ['Python', 'JavaScript', 'SQL', 'Machine Learning', 'Cloud Computing', 'DevOps'],
            'leadership': ['Team Management', 'Strategic Planning', 'Decision Making', 'Communication', 'Mentoring'],
            'business': ['Project Management', 'Business Analysis', 'Process Improvement', 'Customer Relations'],
            'soft_skills': ['Problem Solving', 'Creativity', 'Adaptability', 'Collaboration', 'Time Management']
        }
    
    def generate_employee_profiles(self):
        """Generate sample employee profiles for talent management"""
        np.random.seed(42)
        profiles = []
        
        for i in range(50):
            dept = np.random.choice(list(self.career_paths.keys()))
            current_level = np.random.randint(0, len(self.career_paths[dept]) - 2)
            
            profile = {
                'employee_id': f'EMP{i+1:03d}',
                'name': f'Employee {i+1}',
                'department': dept,
                'current_role': self.career_paths[dept][current_level],
                'level': current_level,
                'years_experience': np.random.randint(1, 15),
                'performance_rating': np.random.uniform(3.0, 5.0),
                'potential_rating': np.random.uniform(2.5, 5.0),
                'skills': np.random.choice(
                    [skill for category in self.skills_database.values() for skill in category], 
                    size=np.random.randint(3, 8), replace=False
                ).tolist(),
                'last_promotion': np.random.randint(0, 36),  # months ago
                'ready_for_promotion': np.random.choice([True, False], p=[0.3, 0.7])
            }
            profiles.append(profile)
        
        return profiles
    
    def predict_promotion_readiness(self, profiles):
        """Predict promotion readiness using Random Forest"""
        df = pd.DataFrame(profiles)
        
        # Prepare features
        features = []
        labels = []
        
        for _, row in df.iterrows():
            feature_vector = [
                row['years_experience'],
                row['performance_rating'],
                row['potential_rating'],
                row['last_promotion'],
                len(row['skills']),
                row['level']
            ]
            features.append(feature_vector)
            labels.append(row['ready_for_promotion'])
        
        # Train model
        X = np.array(features)
        y = np.array(labels)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict probabilities
        probabilities = model.predict_proba(X)[:, 1]
        
        # Add predictions to profiles
        for i, profile in enumerate(profiles):
            profile['promotion_probability'] = probabilities[i]
            profile['promotion_recommendation'] = 'High' if probabilities[i] > 0.7 else 'Medium' if probabilities[i] > 0.4 else 'Low'
        
        return profiles
    
    def recommend_career_paths(self, employee_profile, all_profiles):
        """Recommend career paths using similarity analysis"""
        current_skills = set(employee_profile['skills'])
        department = employee_profile['department']
        current_level = employee_profile['level']
        
        recommendations = []
        
        # Next level in current department
        if current_level < len(self.career_paths[department]) - 1:
            next_role = self.career_paths[department][current_level + 1]
            recommendations.append({
                'path_type': 'Vertical',
                'target_role': next_role,
                'department': department,
                'probability': 0.8,
                'timeline': '12-18 months',
                'required_skills': ['Leadership', 'Strategic Planning']
            })
        
        # Lateral moves
        for dept, roles in self.career_paths.items():
            if dept != department and current_level < len(roles):
                lateral_role = roles[current_level]
                recommendations.append({
                    'path_type': 'Lateral',
                    'target_role': lateral_role,
                    'department': dept,
                    'probability': 0.6,
                    'timeline': '6-12 months',
                    'required_skills': ['Cross-functional', 'Adaptability']
                })
        
        return recommendations[:5]  # Top 5 recommendations

def render_talent_dashboard():
    """Render talent management dashboard"""
    st.markdown('<div class="main-header">🌟 Talent Management</div>', unsafe_allow_html=True)
    
    talent_manager = TalentManager()
    
    # Talent Overview KPIs
    st.markdown("### 📊 Talent Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Performers", "156", "+12")
    with col2:
        st.metric("Promotion Ready", "34", "+5")
    with col3:
        st.metric("Succession Coverage", "78%", "+3%")
    with col4:
        st.metric("Internal Mobility", "23%", "+2%")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👑 Succession Planning", "🎯 OKR Management", "🚀 Career Development", 
        "📊 Talent Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        render_succession_planning(talent_manager)
    
    with tab2:
        render_okr_management()
    
    with tab3:
        render_career_development(talent_manager)
    
    with tab4:
        render_talent_analytics(talent_manager)
    
    with tab5:
        render_talent_settings()

def render_succession_planning(talent_manager):
    """Render succession planning interface"""
    st.markdown("### 👑 Succession Planning")
    
    # Generate employee profiles
    profiles = talent_manager.generate_employee_profiles()
    profiles = talent_manager.predict_promotion_readiness(profiles)
    
    # Key positions
    st.markdown("#### 🏢 Key Positions")
    
    key_positions = [
        {'role': 'Engineering Manager', 'current': 'John Smith', 'successors': 3, 'risk': 'Low'},
        {'role': 'Sales Director', 'current': 'Sarah Johnson', 'successors': 2, 'risk': 'Medium'},
        {'role': 'Marketing Manager', 'current': 'Mike Wilson', 'successors': 1, 'risk': 'High'},
        {'role': 'HR Director', 'current': 'Lisa Brown', 'successors': 2, 'risk': 'Low'},
    ]
    
    for position in key_positions:
        with st.expander(f"🎯 {position['role']} - {position['current']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Current:** {position['current']}")
                st.write(f"**Successors:** {position['successors']}")
                st.write(f"**Risk Level:** {position['risk']}")
            
            with col2:
                st.markdown("**Potential Successors:**")
                ready_candidates = [p for p in profiles if p['promotion_recommendation'] == 'High'][:3]
                for candidate in ready_candidates:
                    st.write(f"• {candidate['name']} ({candidate['promotion_probability']:.1%})")
            
            with col3:
                if st.button(f"📋 View Succession Plan", key=f"plan_{position['role']}"):
                    st.info(f"Detailed succession plan for {position['role']}")
    
    # Succession matrix
    st.markdown("#### 📊 Succession Matrix")
    
    # Performance vs Potential matrix
    matrix_data = []
    for profile in profiles[:20]:  # Top 20 for visualization
        matrix_data.append({
            'Employee': profile['name'],
            'Performance': profile['performance_rating'],
            'Potential': profile['potential_rating'],
            'Department': profile['department'],
            'Promotion_Ready': profile['promotion_recommendation']
        })
    
    matrix_df = pd.DataFrame(matrix_data)
    
    fig = px.scatter(matrix_df, x='Performance', y='Potential', 
                    color='Promotion_Ready', size='Performance',
                    hover_data=['Employee', 'Department'],
                    title="9-Box Talent Matrix (Performance vs Potential)")
    
    # Add quadrant lines
    fig.add_hline(y=3.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=3.5, line_dash="dash", line_color="gray")
    
    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def render_okr_management():
    """Render OKR management interface"""
    st.markdown("### 🎯 OKR Management")
    
    # OKR overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active OKRs", "247", "+15")
    with col2:
        st.metric("On Track", "189", "+8")
    with col3:
        st.metric("At Risk", "34", "+3")
    with col4:
        st.metric("Avg Progress", "67%", "+5%")
    
    # Create new OKR
    st.markdown("#### ➕ Create New OKR")
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)], key="okr_employee")
        objective_title = st.text_input("Objective Title", "Increase team productivity")
        objective_description = st.text_area("Objective Description", height=100)
        
        quarter = st.selectbox("Quarter", ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"], key="okr_quarter")
        owner = st.selectbox("Owner", [f"EMP{i:03d}" for i in range(1, 21)], key="okr_owner")
    
    with col2:
        st.markdown("**Key Results**")
        kr1 = st.text_input("Key Result 1", "Reduce average task completion time by 20%")
        kr1_target = st.number_input("Target Value 1", value=20.0)
        
        kr2 = st.text_input("Key Result 2", "Implement 3 process improvements")
        kr2_target = st.number_input("Target Value 2", value=3.0)
        
        kr3 = st.text_input("Key Result 3", "Achieve 95% team satisfaction score")
        kr3_target = st.number_input("Target Value 3", value=95.0)
    
    if st.button("🎯 Create OKR", type="primary"):
        st.success(f"✅ OKR created for {employee_id}")
    
    # Active OKRs
    st.markdown("#### 📋 Active OKRs")
    
    sample_okrs = [
        {
            'Employee': 'EMP001',
            'Objective': 'Increase Sales Performance',
            'Progress': 75,
            'Key Results': 3,
            'Status': 'On Track',
            'Quarter': 'Q1 2024'
        },
        {
            'Employee': 'EMP002', 
            'Objective': 'Improve Customer Satisfaction',
            'Progress': 45,
            'Key Results': 4,
            'Status': 'At Risk',
            'Quarter': 'Q1 2024'
        },
        {
            'Employee': 'EMP003',
            'Objective': 'Launch New Product Feature',
            'Progress': 90,
            'Key Results': 2,
            'Status': 'Ahead',
            'Quarter': 'Q1 2024'
        }
    ]
    
    for i, okr in enumerate(sample_okrs):
        with st.expander(f"🎯 {okr['Employee']}: {okr['Objective']} ({okr['Progress']}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Employee:** {okr['Employee']}")
                st.write(f"**Quarter:** {okr['Quarter']}")
                st.write(f"**Status:** {okr['Status']}")
            
            with col2:
                st.write(f"**Progress:** {okr['Progress']}%")
                st.progress(okr['Progress'] / 100)
                st.write(f"**Key Results:** {okr['Key Results']}")
            
            with col3:
                if st.button("📝 Update Progress", key=f"update_okr_{i}"):
                    st.info("Progress update form opened")
                if st.button("📊 View Details", key=f"details_okr_{i}"):
                    st.info("Detailed OKR view opened")

def render_career_development(talent_manager):
    """Render career development interface"""
    st.markdown("### 🚀 Career Development")
    
    # Employee selection
    employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)], key="career_dev_employee")
    
    # Generate profiles for recommendations
    profiles = talent_manager.generate_employee_profiles()
    selected_profile = profiles[0]  # Use first profile as example
    
    # Current profile
    st.markdown("#### 👤 Employee Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Role", selected_profile['current_role'])
    with col2:
        st.metric("Experience", f"{selected_profile['years_experience']} years")
    with col3:
        st.metric("Performance", f"{selected_profile['performance_rating']:.1f}/5.0")
    with col4:
        st.metric("Potential", f"{selected_profile['potential_rating']:.1f}/5.0")
    
    # Skills assessment
    st.markdown("#### 🛠️ Skills Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Skills**")
        for skill in selected_profile['skills']:
            st.write(f"• {skill}")
    
    with col2:
        st.markdown("**Skill Gaps**")
        recommended_skills = ['Leadership', 'Strategic Planning', 'Data Analysis', 'Project Management']
        for skill in recommended_skills:
            if skill not in selected_profile['skills']:
                st.write(f"• {skill} ⚠️")
    
    # Career path recommendations
    st.markdown("#### 🛤️ Career Path Recommendations")
    
    if st.button("🤖 Generate AI Recommendations", type="primary"):
        recommendations = talent_manager.recommend_career_paths(selected_profile, profiles)
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"🎯 {rec['path_type']} Move: {rec['target_role']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Target Role:** {rec['target_role']}")
                    st.write(f"**Department:** {rec['department']}")
                    st.write(f"**Path Type:** {rec['path_type']}")
                
                with col2:
                    st.write(f"**Success Probability:** {rec['probability']:.1%}")
                    st.write(f"**Timeline:** {rec['timeline']}")
                
                with col3:
                    st.markdown("**Required Skills:**")
                    for skill in rec['required_skills']:
                        st.write(f"• {skill}")
                
                if st.button(f"📋 Create Development Plan", key=f"plan_{i}"):
                    st.success(f"Development plan created for {rec['target_role']}")
    
    # Development activities
    st.markdown("#### 📚 Development Activities")
    
    activities = [
        {'type': 'Training', 'title': 'Leadership Fundamentals', 'duration': '2 weeks', 'status': 'Available'},
        {'type': 'Mentoring', 'title': 'Senior Manager Mentorship', 'duration': '3 months', 'status': 'In Progress'},
        {'type': 'Project', 'title': 'Cross-functional Initiative', 'duration': '6 months', 'status': 'Planned'},
        {'type': 'Certification', 'title': 'Project Management (PMP)', 'duration': '4 months', 'status': 'Available'}
    ]
    
    for activity in activities:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**{activity['type']}**")
        with col2:
            st.write(activity['title'])
        with col3:
            st.write(activity['duration'])
        with col4:
            status_color = {'Available': '🟢', 'In Progress': '🟡', 'Planned': '🔵'}
            st.write(f"{status_color.get(activity['status'], '⚪')} {activity['status']}")

def render_talent_analytics(talent_manager):
    """Render talent analytics dashboard"""
    st.markdown("### 📊 Talent Analytics")
    
    # Generate data for analytics
    profiles = talent_manager.generate_employee_profiles()
    profiles = talent_manager.predict_promotion_readiness(profiles)
    
    # Talent distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        performance_data = [p['performance_rating'] for p in profiles]
        fig1 = px.histogram(x=performance_data, nbins=10, 
                           title="Performance Rating Distribution")
        fig1.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Department talent distribution
        dept_counts = pd.Series([p['department'] for p in profiles]).value_counts()
        fig2 = px.pie(values=dept_counts.values, names=dept_counts.index,
                     title="Talent Distribution by Department")
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Promotion readiness
    st.markdown("#### 🚀 Promotion Readiness Analysis")
    
    promotion_data = pd.DataFrame(profiles)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Promotion readiness by department
        promo_by_dept = promotion_data.groupby(['department', 'promotion_recommendation']).size().unstack(fill_value=0)
        fig3 = px.bar(promo_by_dept, title="Promotion Readiness by Department")
        fig3.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # High performers
        high_performers = promotion_data[promotion_data['promotion_recommendation'] == 'High']
        st.markdown("**High Potential Employees**")
        
        for _, emp in high_performers.head(5).iterrows():
            st.write(f"• **{emp['name']}** - {emp['current_role']} ({emp['promotion_probability']:.1%})")
    
    # Skills analysis
    st.markdown("#### 🛠️ Skills Analysis")
    
    all_skills = [skill for profile in profiles for skill in profile['skills']]
    skill_counts = pd.Series(all_skills).value_counts().head(10)
    
    fig4 = px.bar(x=skill_counts.values, y=skill_counts.index, orientation='h',
                 title="Top 10 Skills in Organization")
    fig4.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

def render_talent_settings():
    """Render talent management settings"""
    st.markdown("### ⚙️ Talent Management Settings")
    
    # Performance review settings
    st.markdown("#### 📊 Performance Review Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        review_frequency = st.selectbox("Review Frequency", ["Quarterly", "Semi-Annual", "Annual"], key="review_freq")
        rating_scale = st.selectbox("Rating Scale", ["1-5", "1-10", "A-F"], key="rating_scale")
        self_review = st.checkbox("Enable Self Reviews", value=True)
        peer_review = st.checkbox("Enable Peer Reviews", value=True)
    
    with col2:
        manager_review = st.checkbox("Manager Reviews", value=True)
        skip_level_review = st.checkbox("Skip Level Reviews", value=False)
        calibration_sessions = st.checkbox("Calibration Sessions", value=True)
        review_templates = st.checkbox("Standardized Templates", value=True)
    
    # OKR settings
    st.markdown("#### 🎯 OKR Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        okr_cycle = st.selectbox("OKR Cycle", ["Quarterly", "Semi-Annual", "Annual"], key="okr_cycle_setting")
        max_objectives = st.number_input("Max Objectives per Employee", value=3, min_value=1, max_value=10)
        max_key_results = st.number_input("Max Key Results per Objective", value=5, min_value=1, max_value=10)
    
    with col2:
        okr_visibility = st.selectbox("OKR Visibility", ["Public", "Team Only", "Manager Only"], key="okr_visibility")
        progress_updates = st.selectbox("Progress Update Frequency", ["Weekly", "Bi-weekly", "Monthly"], key="progress_updates")
        automated_reminders = st.checkbox("Automated Progress Reminders", value=True)
    
    # Career development settings
    st.markdown("#### 🚀 Career Development Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        career_conversations = st.checkbox("Mandatory Career Conversations", value=True)
        development_budget = st.number_input("Annual Development Budget per Employee", value=2000)
        mentorship_program = st.checkbox("Formal Mentorship Program", value=True)
    
    with col2:
        succession_planning = st.checkbox("Succession Planning", value=True)
        internal_mobility = st.checkbox("Internal Mobility Program", value=True)
        skills_tracking = st.checkbox("Skills Tracking", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Talent management settings saved successfully!")

if __name__ == "__main__":
    render_talent_dashboard()
