"""
Learning & Development System
AI-powered learning management with training catalog, course enrollment, certification tracking, and personalized learning paths
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
from lightfm.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class LearningManager:
    def __init__(self):
        self.course_catalog = {
            'technical': [
                {'id': 'TECH001', 'title': 'Python Programming Fundamentals', 'duration': 40, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.5},
                {'id': 'TECH002', 'title': 'Machine Learning Basics', 'duration': 60, 'level': 'Intermediate', 'provider': 'Coursera', 'rating': 4.7},
                {'id': 'TECH003', 'title': 'Cloud Computing with AWS', 'duration': 80, 'level': 'Advanced', 'provider': 'AWS', 'rating': 4.6},
                {'id': 'TECH004', 'title': 'Data Analysis with SQL', 'duration': 30, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.3},
                {'id': 'TECH005', 'title': 'DevOps Fundamentals', 'duration': 50, 'level': 'Intermediate', 'provider': 'Udemy', 'rating': 4.4}
            ],
            'leadership': [
                {'id': 'LEAD001', 'title': 'Leadership Essentials', 'duration': 20, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.2},
                {'id': 'LEAD002', 'title': 'Strategic Management', 'duration': 35, 'level': 'Advanced', 'provider': 'Harvard', 'rating': 4.8},
                {'id': 'LEAD003', 'title': 'Team Building & Communication', 'duration': 25, 'level': 'Intermediate', 'provider': 'Internal', 'rating': 4.1},
                {'id': 'LEAD004', 'title': 'Change Management', 'duration': 30, 'level': 'Intermediate', 'provider': 'LinkedIn', 'rating': 4.3},
                {'id': 'LEAD005', 'title': 'Executive Coaching', 'duration': 45, 'level': 'Advanced', 'provider': 'External', 'rating': 4.6}
            ],
            'business': [
                {'id': 'BUS001', 'title': 'Project Management (PMP)', 'duration': 120, 'level': 'Advanced', 'provider': 'PMI', 'rating': 4.7},
                {'id': 'BUS002', 'title': 'Business Analysis Fundamentals', 'duration': 40, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.0},
                {'id': 'BUS003', 'title': 'Financial Planning & Analysis', 'duration': 60, 'level': 'Intermediate', 'provider': 'Coursera', 'rating': 4.4},
                {'id': 'BUS004', 'title': 'Digital Marketing Strategy', 'duration': 35, 'level': 'Intermediate', 'provider': 'Google', 'rating': 4.5},
                {'id': 'BUS005', 'title': 'Customer Relations Management', 'duration': 25, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.2}
            ],
            'compliance': [
                {'id': 'COMP001', 'title': 'Data Privacy & GDPR', 'duration': 15, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.1},
                {'id': 'COMP002', 'title': 'Workplace Safety Training', 'duration': 10, 'level': 'Beginner', 'provider': 'OSHA', 'rating': 4.0},
                {'id': 'COMP003', 'title': 'Anti-Harassment Training', 'duration': 8, 'level': 'Beginner', 'provider': 'Internal', 'rating': 3.9},
                {'id': 'COMP004', 'title': 'Information Security Awareness', 'duration': 12, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.2},
                {'id': 'COMP005', 'title': 'Ethics & Code of Conduct', 'duration': 6, 'level': 'Beginner', 'provider': 'Internal', 'rating': 4.0}
            ]
        }
        
        self.certifications = [
            {'name': 'AWS Certified Solutions Architect', 'provider': 'AWS', 'validity_months': 36, 'renewal_required': True},
            {'name': 'Project Management Professional (PMP)', 'provider': 'PMI', 'validity_months': 36, 'renewal_required': True},
            {'name': 'Certified ScrumMaster (CSM)', 'provider': 'Scrum Alliance', 'validity_months': 24, 'renewal_required': True},
            {'name': 'Google Analytics Certified', 'provider': 'Google', 'validity_months': 12, 'renewal_required': True},
            {'name': 'Microsoft Azure Fundamentals', 'provider': 'Microsoft', 'validity_months': 24, 'renewal_required': False}
        ]
    
    def get_all_courses(self):
        """Get all courses from catalog"""
        all_courses = []
        for category, courses in self.course_catalog.items():
            for course in courses:
                course['category'] = category
                all_courses.append(course)
        return all_courses
    
    def recommend_courses(self, employee_profile, performance_gaps=None):
        """AI-powered course recommendations based on employee profile and performance gaps"""
        all_courses = self.get_all_courses()
        recommendations = []
        
        # Get employee's current skills and role
        current_skills = employee_profile.get('skills', [])
        current_role = employee_profile.get('role', '').lower()
        department = employee_profile.get('department', '').lower()
        
        # Skill gap analysis
        if performance_gaps:
            for gap in performance_gaps:
                gap_skill = gap.lower()
                # Find courses that match the skill gap
                for course in all_courses:
                    course_title = course['title'].lower()
                    if gap_skill in course_title or any(skill.lower() in course_title for skill in current_skills):
                        recommendations.append({
                            'course': course,
                            'reason': f'Addresses skill gap: {gap}',
                            'priority': 'High',
                            'match_score': 0.9
                        })
        
        # Role-based recommendations
        role_keywords = {
            'engineer': ['technical', 'programming', 'cloud', 'devops'],
            'manager': ['leadership', 'management', 'strategic'],
            'analyst': ['data', 'analysis', 'sql', 'business'],
            'sales': ['customer', 'marketing', 'business'],
            'hr': ['compliance', 'leadership', 'communication']
        }
        
        relevant_categories = []
        for role_key, keywords in role_keywords.items():
            if role_key in current_role:
                relevant_categories.extend(keywords)
        
        # Add role-based course recommendations
        for course in all_courses:
            course_title = course['title'].lower()
            category = course['category']
            
            if category in relevant_categories or any(keyword in course_title for keyword in relevant_categories):
                if not any(rec['course']['id'] == course['id'] for rec in recommendations):
                    recommendations.append({
                        'course': course,
                        'reason': f'Relevant for {current_role} role',
                        'priority': 'Medium',
                        'match_score': 0.7
                    })
        
        # Sort by priority and match score
        recommendations.sort(key=lambda x: (x['priority'] == 'High', x['match_score']), reverse=True)
        return recommendations[:10]  # Top 10 recommendations
    
    def generate_learning_path(self, target_role, current_skills):
        """Generate personalized learning path for career progression"""
        all_courses = self.get_all_courses()
        
        # Define skill requirements for different roles
        role_requirements = {
            'senior_developer': ['Python', 'Machine Learning', 'Cloud Computing', 'DevOps'],
            'team_lead': ['Leadership', 'Project Management', 'Communication', 'Strategic Planning'],
            'data_scientist': ['Machine Learning', 'Data Analysis', 'Python', 'Statistics'],
            'product_manager': ['Business Analysis', 'Project Management', 'Strategic Management', 'Customer Relations']
        }
        
        target_skills = role_requirements.get(target_role.lower().replace(' ', '_'), [])
        skill_gaps = [skill for skill in target_skills if skill not in current_skills]
        
        learning_path = []
        for gap in skill_gaps:
            # Find relevant courses for each skill gap
            relevant_courses = []
            for course in all_courses:
                if gap.lower() in course['title'].lower():
                    relevant_courses.append(course)
            
            if relevant_courses:
                # Sort by level (beginner first) and rating
                relevant_courses.sort(key=lambda x: (x['level'] == 'Beginner', x['rating']), reverse=True)
                learning_path.append({
                    'skill': gap,
                    'courses': relevant_courses[:2],  # Top 2 courses per skill
                    'estimated_duration': sum(c['duration'] for c in relevant_courses[:2])
                })
        
        return learning_path

def render_learning_dashboard():
    """Render learning and development dashboard"""
    st.markdown('<div class="main-header">📚 Learning & Development</div>', unsafe_allow_html=True)
    
    learning_manager = LearningManager()
    
    # Learning Overview KPIs
    st.markdown("### 📊 Learning Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Learners", "892", "+67")
    with col2:
        st.metric("Courses Completed", "2,340", "+156")
    with col3:
        st.metric("Avg Completion Rate", "78.5%", "+3.2%")
    with col4:
        st.metric("Certifications Earned", "145", "+23")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Course Catalog", "🎯 My Learning", "🏆 Certifications", 
        "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        render_course_catalog(learning_manager)
    
    with tab2:
        render_my_learning(learning_manager)
    
    with tab3:
        render_certifications(learning_manager)
    
    with tab4:
        render_learning_analytics(learning_manager)
    
    with tab5:
        render_learning_settings()

def render_course_catalog(learning_manager):
    """Render course catalog interface"""
    st.markdown("### 📖 Course Catalog")
    
    # Search and filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search Courses", placeholder="Enter course title or keyword")
    with col2:
        category_filter = st.selectbox("Category", ["All"] + list(learning_manager.course_catalog.keys()))
    with col3:
        level_filter = st.selectbox("Level", ["All", "Beginner", "Intermediate", "Advanced"])
    
    # Get all courses
    all_courses = learning_manager.get_all_courses()
    
    # Apply filters
    filtered_courses = all_courses
    
    if search_term:
        filtered_courses = [c for c in filtered_courses if search_term.lower() in c['title'].lower()]
    
    if category_filter != "All":
        filtered_courses = [c for c in filtered_courses if c['category'] == category_filter]
    
    if level_filter != "All":
        filtered_courses = [c for c in filtered_courses if c['level'] == level_filter]
    
    # Display courses
    st.markdown(f"#### 📚 Available Courses ({len(filtered_courses)} found)")
    
    # Course grid
    cols = st.columns(2)
    for i, course in enumerate(filtered_courses[:10]):  # Show first 10 courses
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="glass-card" style="padding: 20px; margin: 10px 0;">
                    <h4>{course['title']}</h4>
                    <p><strong>Category:</strong> {course['category'].title()}</p>
                    <p><strong>Level:</strong> {course['level']}</p>
                    <p><strong>Duration:</strong> {course['duration']} hours</p>
                    <p><strong>Provider:</strong> {course['provider']}</p>
                    <p><strong>Rating:</strong> ⭐ {course['rating']}/5.0</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_enroll, col_info = st.columns(2)
                with col_enroll:
                    if st.button("📝 Enroll", key=f"enroll_{course['id']}"):
                        st.success(f"✅ Enrolled in {course['title']}")
                with col_info:
                    if st.button("ℹ️ Details", key=f"details_{course['id']}"):
                        st.info(f"Course details for {course['title']}")

def render_my_learning(learning_manager):
    """Render personalized learning interface"""
    st.markdown("### 🎯 My Learning")
    
    # Employee selection for demo
    employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)])
    
    # Mock employee profile
    employee_profile = {
        'employee_id': employee_id,
        'role': 'Software Developer',
        'department': 'Engineering',
        'skills': ['Python', 'JavaScript', 'SQL'],
        'performance_gaps': ['Machine Learning', 'Cloud Computing', 'Leadership']
    }
    
    # Current enrollments
    st.markdown("#### 📚 Current Enrollments")
    
    current_enrollments = [
        {'course': 'Python Programming Fundamentals', 'progress': 75, 'due_date': '2024-04-15', 'status': 'In Progress'},
        {'course': 'Machine Learning Basics', 'progress': 30, 'due_date': '2024-05-01', 'status': 'In Progress'},
        {'course': 'Leadership Essentials', 'progress': 100, 'due_date': '2024-03-20', 'status': 'Completed'}
    ]
    
    for enrollment in current_enrollments:
        with st.expander(f"📖 {enrollment['course']} ({enrollment['progress']}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Progress:** {enrollment['progress']}%")
                st.progress(enrollment['progress'] / 100)
            with col2:
                st.write(f"**Due Date:** {enrollment['due_date']}")
                st.write(f"**Status:** {enrollment['status']}")
            with col3:
                if enrollment['status'] == 'In Progress':
                    if st.button("▶️ Continue", key=f"continue_{enrollment['course']}"):
                        st.info("Redirecting to course...")
                else:
                    st.success("✅ Completed")
    
    # AI Recommendations
    st.markdown("#### 🤖 AI-Powered Recommendations")
    
    if st.button("🔮 Generate Personalized Recommendations", type="primary"):
        recommendations = learning_manager.recommend_courses(
            employee_profile, 
            employee_profile['performance_gaps']
        )
        
        st.success(f"✅ Generated {len(recommendations)} personalized recommendations")
        
        for i, rec in enumerate(recommendations[:5]):  # Show top 5
            course = rec['course']
            with st.expander(f"🎯 {course['title']} ({rec['priority']} Priority)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Course:** {course['title']}")
                    st.write(f"**Duration:** {course['duration']} hours")
                    st.write(f"**Level:** {course['level']}")
                
                with col2:
                    st.write(f"**Provider:** {course['provider']}")
                    st.write(f"**Rating:** ⭐ {course['rating']}/5.0")
                    st.write(f"**Priority:** {rec['priority']}")
                
                with col3:
                    st.write(f"**Reason:** {rec['reason']}")
                    st.write(f"**Match Score:** {rec['match_score']:.1%}")
                
                if st.button("📝 Enroll Now", key=f"enroll_rec_{i}"):
                    st.success(f"✅ Enrolled in {course['title']}")
    
    # Learning Path
    st.markdown("#### 🛤️ Personalized Learning Path")
    
    target_role = st.selectbox("Target Role", [
        "Senior Developer", "Team Lead", "Data Scientist", "Product Manager"
    ])
    
    if st.button("🗺️ Generate Learning Path"):
        learning_path = learning_manager.generate_learning_path(
            target_role, 
            employee_profile['skills']
        )
        
        if learning_path:
            st.success(f"✅ Learning path generated for {target_role}")
            
            total_duration = sum(step['estimated_duration'] for step in learning_path)
            st.write(f"**Total Estimated Duration:** {total_duration} hours")
            
            for i, step in enumerate(learning_path):
                with st.expander(f"Step {i+1}: {step['skill']} ({step['estimated_duration']} hours)"):
                    st.write(f"**Skill to Develop:** {step['skill']}")
                    st.write("**Recommended Courses:**")
                    
                    for course in step['courses']:
                        st.write(f"• {course['title']} ({course['duration']} hours, {course['level']})")
        else:
            st.info("You already have all required skills for this role!")

def render_certifications(learning_manager):
    """Render certifications management"""
    st.markdown("### 🏆 Certifications")
    
    # Certification overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Certifications", "145", "+23")
    with col2:
        st.metric("Expiring Soon", "12", "+3")
    with col3:
        st.metric("Renewal Required", "8", "+2")
    with col4:
        st.metric("Completion Rate", "89%", "+5%")
    
    # Available certifications
    st.markdown("#### 📜 Available Certifications")
    
    for cert in learning_manager.certifications:
        with st.expander(f"🏆 {cert['name']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Provider:** {cert['provider']}")
                st.write(f"**Validity:** {cert['validity_months']} months")
            with col2:
                st.write(f"**Renewal Required:** {'Yes' if cert['renewal_required'] else 'No'}")
                st.write("**Prerequisites:** Course completion")
            with col3:
                if st.button("📝 Start Certification", key=f"cert_{cert['name']}"):
                    st.success(f"✅ Started certification process for {cert['name']}")
    
    # Employee certifications
    st.markdown("#### 👤 Employee Certifications")
    
    employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)], key="cert_employee")
    
    # Mock employee certifications
    employee_certs = [
        {'name': 'AWS Certified Solutions Architect', 'earned_date': '2023-06-15', 'expiry_date': '2026-06-15', 'status': 'Active'},
        {'name': 'Project Management Professional (PMP)', 'earned_date': '2022-03-20', 'expiry_date': '2025-03-20', 'status': 'Expiring Soon'},
        {'name': 'Google Analytics Certified', 'earned_date': '2023-01-10', 'expiry_date': '2024-01-10', 'status': 'Expired'}
    ]
    
    cert_df = pd.DataFrame(employee_certs)
    st.dataframe(cert_df, use_container_width=True, hide_index=True)
    
    # Certification reminders
    st.markdown("#### 🔔 Renewal Reminders")
    
    expiring_certs = [cert for cert in employee_certs if cert['status'] in ['Expiring Soon', 'Expired']]
    
    if expiring_certs:
        for cert in expiring_certs:
            status_color = '🟡' if cert['status'] == 'Expiring Soon' else '🔴'
            st.warning(f"{status_color} {cert['name']} - {cert['status']} (Expires: {cert['expiry_date']})")
    else:
        st.success("✅ All certifications are up to date!")

def render_learning_analytics(learning_manager):
    """Render learning analytics dashboard"""
    st.markdown("### 📊 Learning Analytics")
    
    # Generate sample analytics data
    all_courses = learning_manager.get_all_courses()
    
    # Course popularity
    st.markdown("#### 📈 Course Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most popular courses
        course_enrollments = {course['title']: np.random.randint(20, 150) for course in all_courses[:10]}
        
        fig1 = px.bar(x=list(course_enrollments.values()), 
                     y=list(course_enrollments.keys()),
                     orientation='h',
                     title="Most Popular Courses (Enrollments)")
        fig1.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Completion rates by category
        categories = list(learning_manager.course_catalog.keys())
        completion_rates = [np.random.uniform(70, 95) for _ in categories]
        
        fig2 = px.bar(x=categories, y=completion_rates,
                     title="Completion Rates by Category (%)")
        fig2.update_layout(height=500, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Learning trends
    st.markdown("#### 📅 Learning Trends")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    enrollments = [120, 135, 150, 180, 165, 190]
    completions = [95, 108, 125, 145, 132, 155]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=months, y=enrollments, name='Enrollments', line=dict(color='#1f77b4', width=3)))
    fig3.add_trace(go.Scatter(x=months, y=completions, name='Completions', line=dict(color='#ff7f0e', width=3)))
    
    fig3.update_layout(
        title="Monthly Learning Trends",
        xaxis_title="Month",
        yaxis_title="Count",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Skills development
    st.markdown("#### 🛠️ Skills Development")
    
    skills_data = pd.DataFrame({
        'Skill': ['Python', 'Leadership', 'Project Management', 'Machine Learning', 'Cloud Computing'],
        'Learners': [145, 89, 67, 123, 98],
        'Avg Progress': [78, 85, 72, 65, 81],
        'Completion Rate': [82, 91, 76, 58, 74]
    })
    
    st.dataframe(skills_data, use_container_width=True, hide_index=True)

def render_learning_settings():
    """Render learning system settings"""
    st.markdown("### ⚙️ Learning Settings")
    
    # Learning policies
    st.markdown("#### 📋 Learning Policies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mandatory_training = st.checkbox("Mandatory Compliance Training", value=True)
        learning_budget = st.number_input("Annual Learning Budget per Employee", value=1500)
        max_concurrent_courses = st.number_input("Max Concurrent Courses", value=3, min_value=1, max_value=10)
    
    with col2:
        auto_enrollment = st.checkbox("Auto-enroll in Required Courses", value=True)
        completion_deadline = st.number_input("Course Completion Deadline (days)", value=90, min_value=30, max_value=365)
        certification_reminders = st.checkbox("Certification Renewal Reminders", value=True)
    
    # Content management
    st.markdown("#### 📚 Content Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        internal_courses = st.checkbox("Enable Internal Course Creation", value=True)
        external_providers = st.multiselect("Approved External Providers", 
                                          ["Coursera", "Udemy", "LinkedIn Learning", "Pluralsight", "AWS Training"])
        content_approval = st.checkbox("Require Content Approval", value=True)
    
    with col2:
        course_ratings = st.checkbox("Enable Course Ratings", value=True)
        discussion_forums = st.checkbox("Enable Discussion Forums", value=True)
        progress_tracking = st.checkbox("Detailed Progress Tracking", value=True)
    
    # Notifications
    st.markdown("#### 🔔 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enrollment_notifications = st.checkbox("Course Enrollment Notifications", value=True)
        completion_notifications = st.checkbox("Course Completion Notifications", value=True)
        deadline_reminders = st.checkbox("Deadline Reminders", value=True)
    
    with col2:
        manager_notifications = st.checkbox("Manager Progress Notifications", value=True)
        achievement_notifications = st.checkbox("Achievement Notifications", value=True)
        recommendation_notifications = st.checkbox("New Recommendation Notifications", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Learning settings saved successfully!")

if __name__ == "__main__":
    render_learning_dashboard()
