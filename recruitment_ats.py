"""
Recruitment & Applicant Tracking System (ATS)
AI-powered recruitment with job postings, candidate pipelines, resume parsing, and intelligent candidate matching
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

class RecruitmentManager:
    def __init__(self):
        self.job_postings = [
            {
                'id': 'JOB001',
                'title': 'Senior Software Engineer',
                'department': 'Engineering',
                'location': 'New York, NY',
                'type': 'Full-time',
                'posted_date': '2024-03-01',
                'status': 'Active',
                'applications': 45,
                'required_skills': ['Python', 'JavaScript', 'React', 'SQL', 'AWS'],
                'experience_level': 'Senior',
                'salary_range': '$120,000 - $150,000'
            },
            {
                'id': 'JOB002',
                'title': 'Marketing Manager',
                'department': 'Marketing',
                'location': 'San Francisco, CA',
                'type': 'Full-time',
                'posted_date': '2024-03-05',
                'status': 'Active',
                'applications': 32,
                'required_skills': ['Digital Marketing', 'SEO', 'Analytics', 'Content Strategy'],
                'experience_level': 'Mid-level',
                'salary_range': '$80,000 - $100,000'
            },
            {
                'id': 'JOB003',
                'title': 'Data Scientist',
                'department': 'Analytics',
                'location': 'Remote',
                'type': 'Full-time',
                'posted_date': '2024-02-28',
                'status': 'Active',
                'applications': 67,
                'required_skills': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Tableau'],
                'experience_level': 'Senior',
                'salary_range': '$130,000 - $160,000'
            }
        ]
        
        self.candidates = self._generate_candidates()
        
    def _generate_candidates(self):
        """Generate sample candidate data"""
        np.random.seed(42)
        candidates = []
        
        skills_pool = ['Python', 'JavaScript', 'React', 'SQL', 'AWS', 'Machine Learning', 
                      'Digital Marketing', 'SEO', 'Analytics', 'Project Management',
                      'Leadership', 'Communication', 'Problem Solving']
        
        for i in range(100):
            candidate = {
                'id': f'CAND{i+1:03d}',
                'name': f'Candidate {i+1}',
                'email': f'candidate{i+1}@email.com',
                'phone': f'555-{np.random.randint(1000, 9999)}',
                'location': np.random.choice(['New York, NY', 'San Francisco, CA', 'Remote', 'Chicago, IL']),
                'experience_years': np.random.randint(1, 15),
                'current_role': np.random.choice(['Software Engineer', 'Marketing Specialist', 'Data Analyst', 'Product Manager']),
                'skills': np.random.choice(skills_pool, size=np.random.randint(3, 8), replace=False).tolist(),
                'education': np.random.choice(['Bachelor\'s', 'Master\'s', 'PhD']),
                'applied_jobs': np.random.choice([job['id'] for job in self.job_postings], 
                                               size=np.random.randint(1, 3), replace=False).tolist(),
                'stage': np.random.choice(['Applied', 'Screening', 'Interview', 'Final Review', 'Offer', 'Rejected']),
                'score': np.random.uniform(0.3, 0.95),
                'resume_text': f"Experienced {np.random.choice(['software engineer', 'marketing professional', 'data analyst'])} with {np.random.randint(1, 15)} years of experience.",
                'application_date': (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d')
            }
            candidates.append(candidate)
        
        return candidates
    
    def parse_resume(self, resume_text):
        """Simple resume parsing simulation"""
        # Extract skills (simplified)
        skills_keywords = ['python', 'javascript', 'react', 'sql', 'aws', 'machine learning', 
                          'marketing', 'seo', 'analytics', 'project management']
        
        found_skills = []
        resume_lower = resume_text.lower()
        
        for skill in skills_keywords:
            if skill in resume_lower:
                found_skills.append(skill.title())
        
        # Extract experience years (simplified regex)
        experience_match = re.search(r'(\d+)\s*years?\s*(?:of\s*)?experience', resume_lower)
        experience_years = int(experience_match.group(1)) if experience_match else 0
        
        # Extract education level
        education_keywords = ['phd', 'doctorate', 'master', 'bachelor', 'degree']
        education = 'Not specified'
        for edu in education_keywords:
            if edu in resume_lower:
                education = edu.title()
                break
        
        return {
            'skills': found_skills,
            'experience_years': experience_years,
            'education': education
        }
    
    def calculate_candidate_job_match(self, candidate, job):
        """Calculate match score between candidate and job"""
        candidate_skills = set([skill.lower() for skill in candidate['skills']])
        required_skills = set([skill.lower() for skill in job['required_skills']])
        
        # Skills match score
        skills_match = len(candidate_skills.intersection(required_skills)) / len(required_skills) if required_skills else 0
        
        # Experience level match
        experience_match = 0.5  # Default
        if job['experience_level'] == 'Senior' and candidate['experience_years'] >= 5:
            experience_match = 1.0
        elif job['experience_level'] == 'Mid-level' and 2 <= candidate['experience_years'] <= 7:
            experience_match = 1.0
        elif job['experience_level'] == 'Junior' and candidate['experience_years'] <= 3:
            experience_match = 1.0
        
        # Location match (simplified)
        location_match = 1.0 if candidate['location'] == job['location'] or job['location'] == 'Remote' else 0.7
        
        # Overall match score
        overall_score = (skills_match * 0.5 + experience_match * 0.3 + location_match * 0.2)
        
        return {
            'overall_score': overall_score,
            'skills_match': skills_match,
            'experience_match': experience_match,
            'location_match': location_match
        }
    
    def predict_candidate_success(self, candidates_data):
        """Predict candidate success using logistic regression"""
        if len(candidates_data) < 10:
            return []
        
        # Prepare features
        features = []
        labels = []
        
        for candidate in candidates_data:
            feature_vector = [
                candidate['experience_years'],
                len(candidate['skills']),
                candidate['score'],
                1 if candidate['education'] in ['Master\'s', 'PhD'] else 0,
                1 if candidate['stage'] in ['Interview', 'Final Review', 'Offer'] else 0
            ]
            features.append(feature_vector)
            labels.append(1 if candidate['stage'] in ['Offer', 'Interview', 'Final Review'] else 0)
        
        # Train model
        X = np.array(features)
        y = np.array(labels)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, y)
        
        # Predict success probability
        success_probs = model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to candidates
        predictions = []
        for i, candidate in enumerate(candidates_data):
            if success_probs[i] > 0.7:
                predictions.append({
                    'candidate': candidate,
                    'success_probability': success_probs[i],
                    'recommendation': 'High potential - prioritize for interview'
                })
        
        return sorted(predictions, key=lambda x: x['success_probability'], reverse=True)

def render_recruitment_dashboard():
    """Render recruitment and ATS dashboard"""
    st.markdown('<div class="main-header">🎯 Recruitment & ATS</div>', unsafe_allow_html=True)
    
    recruitment_manager = RecruitmentManager()
    
    # Recruitment Overview KPIs
    st.markdown("### 📊 Recruitment Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Jobs", "23", "+3")
    with col2:
        st.metric("Total Applications", "1,247", "+89")
    with col3:
        st.metric("Interviews Scheduled", "45", "+12")
    with col4:
        st.metric("Offers Extended", "8", "+2")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💼 Job Postings", "👥 Candidate Pipeline", "🔍 Resume Screening", 
        "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        render_job_postings(recruitment_manager)
    
    with tab2:
        render_candidate_pipeline(recruitment_manager)
    
    with tab3:
        render_resume_screening(recruitment_manager)
    
    with tab4:
        render_recruitment_analytics(recruitment_manager)
    
    with tab5:
        render_recruitment_settings()

def render_job_postings(recruitment_manager):
    """Render job postings management"""
    st.markdown("### 💼 Job Postings Management")
    
    # Create new job posting
    with st.expander("➕ Create New Job Posting"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title", "Senior Software Engineer")
            department = st.selectbox("Department", ["Engineering", "Marketing", "Sales", "HR", "Finance"])
            location = st.text_input("Location", "New York, NY")
            job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Internship"])
        
        with col2:
            experience_level = st.selectbox("Experience Level", ["Junior", "Mid-level", "Senior", "Executive"])
            salary_range = st.text_input("Salary Range", "$120,000 - $150,000")
            required_skills = st.text_area("Required Skills (comma-separated)", "Python, JavaScript, React, SQL")
            
        job_description = st.text_area("Job Description", height=150)
        
        if st.button("📝 Post Job", type="primary"):
            st.success(f"✅ Job posting created: {job_title}")
    
    # Active job postings
    st.markdown("#### 📋 Active Job Postings")
    
    for job in recruitment_manager.job_postings:
        with st.expander(f"💼 {job['title']} - {job['department']} ({job['applications']} applications)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Title:** {job['title']}")
                st.write(f"**Department:** {job['department']}")
                st.write(f"**Location:** {job['location']}")
                st.write(f"**Type:** {job['type']}")
            
            with col2:
                st.write(f"**Posted:** {job['posted_date']}")
                st.write(f"**Status:** {job['status']}")
                st.write(f"**Applications:** {job['applications']}")
                st.write(f"**Experience:** {job['experience_level']}")
            
            with col3:
                st.write(f"**Salary:** {job['salary_range']}")
                st.write("**Required Skills:**")
                for skill in job['required_skills']:
                    st.write(f"• {skill}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("👥 View Candidates", key=f"candidates_{job['id']}"):
                    st.info(f"Viewing candidates for {job['title']}")
            with col2:
                if st.button("📝 Edit Job", key=f"edit_{job['id']}"):
                    st.info(f"Editing {job['title']}")
            with col3:
                if st.button("📤 Promote", key=f"promote_{job['id']}"):
                    st.success("Job promoted on job boards")
            with col4:
                if st.button("⏸️ Pause", key=f"pause_{job['id']}"):
                    st.warning("Job posting paused")

def render_candidate_pipeline(recruitment_manager):
    """Render candidate pipeline management"""
    st.markdown("### 👥 Candidate Pipeline")
    
    # Pipeline overview
    pipeline_stages = ['Applied', 'Screening', 'Interview', 'Final Review', 'Offer', 'Rejected']
    stage_counts = {}
    
    for stage in pipeline_stages:
        stage_counts[stage] = len([c for c in recruitment_manager.candidates if c['stage'] == stage])
    
    # Pipeline visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.funnel(
            x=list(stage_counts.values()),
            y=list(stage_counts.keys()),
            title="Recruitment Funnel"
        )
        fig1.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.pie(
            values=list(stage_counts.values()),
            names=list(stage_counts.keys()),
            title="Candidates by Stage"
        )
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Job selection for pipeline view
    selected_job = st.selectbox("Select Job", [f"{job['id']} - {job['title']}" for job in recruitment_manager.job_postings])
    job_id = selected_job.split(' - ')[0]
    
    # Filter candidates for selected job
    job_candidates = [c for c in recruitment_manager.candidates if job_id in c['applied_jobs']]
    
    st.markdown(f"#### 👥 Candidates for {selected_job} ({len(job_candidates)} total)")
    
    # Stage-wise candidate view
    for stage in pipeline_stages:
        stage_candidates = [c for c in job_candidates if c['stage'] == stage]
        
        if stage_candidates:
            with st.expander(f"{stage} ({len(stage_candidates)} candidates)"):
                for candidate in stage_candidates[:5]:  # Show first 5 per stage
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**{candidate['name']}**")
                        st.write(f"📧 {candidate['email']}")
                    
                    with col2:
                        st.write(f"📍 {candidate['location']}")
                        st.write(f"🎓 {candidate['education']}")
                    
                    with col3:
                        st.write(f"⭐ Score: {candidate['score']:.2f}")
                        st.write(f"📅 Applied: {candidate['application_date']}")
                    
                    with col4:
                        if st.button("➡️ Advance", key=f"advance_{candidate['id']}_{stage}"):
                            st.success(f"Advanced {candidate['name']} to next stage")
                        if st.button("❌ Reject", key=f"reject_{candidate['id']}_{stage}"):
                            st.error(f"Rejected {candidate['name']}")

def render_resume_screening(recruitment_manager):
    """Render AI-powered resume screening"""
    st.markdown("### 🔍 AI-Powered Resume Screening")
    
    # Job selection for screening
    selected_job = st.selectbox("Select Job for Screening", 
                               [f"{job['id']} - {job['title']}" for job in recruitment_manager.job_postings],
                               key="screening_job")
    job_id = selected_job.split(' - ')[0]
    job = next(job for job in recruitment_manager.job_postings if job['id'] == job_id)
    
    # Resume upload simulation
    st.markdown("#### 📄 Resume Upload & Parsing")
    
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file is not None:
        # Simulate resume parsing
        sample_resume_text = """
        John Doe
        Senior Software Engineer
        
        Experience: 7 years of experience in software development
        Skills: Python, JavaScript, React, SQL, AWS, Machine Learning
        Education: Master's degree in Computer Science
        
        Previous roles:
        - Software Engineer at Tech Corp (5 years)
        - Junior Developer at StartupXYZ (2 years)
        """
        
        st.text_area("Parsed Resume Content", sample_resume_text, height=200)
        
        if st.button("🤖 Analyze Resume", type="primary"):
            parsed_data = recruitment_manager.parse_resume(sample_resume_text)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Extracted Skills:**")
                for skill in parsed_data['skills']:
                    st.write(f"• {skill}")
            
            with col2:
                st.write(f"**Experience:** {parsed_data['experience_years']} years")
                st.write(f"**Education:** {parsed_data['education']}")
            
            with col3:
                # Calculate match score
                mock_candidate = {
                    'skills': parsed_data['skills'],
                    'experience_years': parsed_data['experience_years'],
                    'location': 'New York, NY'
                }
                
                match_score = recruitment_manager.calculate_candidate_job_match(mock_candidate, job)
                
                st.write("**Match Analysis:**")
                st.write(f"Overall Score: {match_score['overall_score']:.1%}")
                st.write(f"Skills Match: {match_score['skills_match']:.1%}")
                st.write(f"Experience Match: {match_score['experience_match']:.1%}")
                
                if match_score['overall_score'] > 0.7:
                    st.success("🟢 Strong Match - Recommend for Interview")
                elif match_score['overall_score'] > 0.5:
                    st.warning("🟡 Moderate Match - Consider for Screening")
                else:
                    st.error("🔴 Weak Match - Not Recommended")
    
    # AI Candidate Recommendations
    st.markdown("#### 🤖 AI Candidate Recommendations")
    
    if st.button("🔮 Generate AI Recommendations", type="primary"):
        # Filter candidates for selected job
        job_candidates = [c for c in recruitment_manager.candidates if job_id in c['applied_jobs']]
        
        # Get AI predictions
        predictions = recruitment_manager.predict_candidate_success(job_candidates)
        
        if predictions:
            st.success(f"✅ Identified {len(predictions)} high-potential candidates")
            
            for i, pred in enumerate(predictions[:5]):  # Show top 5
                candidate = pred['candidate']
                
                with st.expander(f"🌟 {candidate['name']} - {pred['success_probability']:.1%} success probability"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Name:** {candidate['name']}")
                        st.write(f"**Current Role:** {candidate['current_role']}")
                        st.write(f"**Experience:** {candidate['experience_years']} years")
                    
                    with col2:
                        st.write(f"**Location:** {candidate['location']}")
                        st.write(f"**Education:** {candidate['education']}")
                        st.write(f"**Current Stage:** {candidate['stage']}")
                    
                    with col3:
                        st.write(f"**Success Probability:** {pred['success_probability']:.1%}")
                        st.write(f"**Recommendation:** {pred['recommendation']}")
                        
                        # Calculate job match
                        match_score = recruitment_manager.calculate_candidate_job_match(candidate, job)
                        st.write(f"**Job Match:** {match_score['overall_score']:.1%}")
                    
                    if st.button(f"📞 Schedule Interview", key=f"interview_{candidate['id']}"):
                        st.success(f"Interview scheduled with {candidate['name']}")
        else:
            st.info("No high-potential candidates identified for this job.")

def render_recruitment_analytics(recruitment_manager):
    """Render recruitment analytics dashboard"""
    st.markdown("### 📊 Recruitment Analytics")
    
    # Time to hire metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Time to Hire", "28 days", "-3 days")
    with col2:
        st.metric("Cost per Hire", "$4,250", "-$200")
    with col3:
        st.metric("Offer Acceptance Rate", "85%", "+5%")
    
    # Hiring trends
    st.markdown("#### 📈 Hiring Trends")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    applications = [120, 135, 180, 165, 190, 210]
    hires = [8, 12, 15, 11, 14, 16]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=months, y=applications, name='Applications', 
                             line=dict(color='#1f77b4', width=3), yaxis='y'))
    fig1.add_trace(go.Scatter(x=months, y=hires, name='Hires', 
                             line=dict(color='#ff7f0e', width=3), yaxis='y2'))
    
    fig1.update_layout(
        title="Monthly Hiring Trends",
        xaxis_title="Month",
        yaxis=dict(title="Applications", side='left'),
        yaxis2=dict(title="Hires", side='right', overlaying='y'),
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Source effectiveness
    st.markdown("#### 🎯 Source Effectiveness")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sources = ['LinkedIn', 'Company Website', 'Indeed', 'Referrals', 'Recruiters']
        applications_by_source = [45, 35, 25, 20, 15]
        
        fig2 = px.bar(x=sources, y=applications_by_source,
                     title="Applications by Source")
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        hires_by_source = [12, 8, 5, 8, 3]
        conversion_rates = [h/a*100 for h, a in zip(hires_by_source, applications_by_source)]
        
        fig3 = px.bar(x=sources, y=conversion_rates,
                     title="Conversion Rate by Source (%)")
        fig3.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Department hiring
    st.markdown("#### 🏢 Hiring by Department")
    
    dept_data = pd.DataFrame({
        'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
        'Open Positions': [8, 5, 3, 2, 1],
        'Applications': [180, 85, 45, 25, 15],
        'Interviews': [24, 12, 8, 4, 3],
        'Offers': [6, 3, 2, 1, 1],
        'Hires': [5, 2, 2, 1, 1]
    })
    
    st.dataframe(dept_data, use_container_width=True, hide_index=True)

def render_recruitment_settings():
    """Render recruitment system settings"""
    st.markdown("### ⚙️ Recruitment Settings")
    
    # Job posting settings
    st.markdown("#### 💼 Job Posting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_post_boards = st.multiselect("Auto-post to Job Boards", 
                                        ["LinkedIn", "Indeed", "Glassdoor", "Monster", "ZipRecruiter"])
        posting_duration = st.number_input("Default Posting Duration (days)", value=30, min_value=7, max_value=90)
        approval_required = st.checkbox("Require Approval for Job Postings", value=True)
    
    with col2:
        salary_disclosure = st.checkbox("Require Salary Range Disclosure", value=True)
        equal_opportunity = st.checkbox("Include EEO Statement", value=True)
        remote_options = st.checkbox("Enable Remote Work Options", value=True)
    
    # Screening settings
    st.markdown("#### 🔍 Screening Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_screening = st.checkbox("Enable AI Auto-Screening", value=True)
        minimum_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.6, 0.1)
        keyword_filtering = st.checkbox("Enable Keyword Filtering", value=True)
    
    with col2:
        resume_parsing = st.checkbox("Enable Resume Parsing", value=True)
        duplicate_detection = st.checkbox("Duplicate Application Detection", value=True)
        bias_detection = st.checkbox("Enable Bias Detection", value=True)
    
    # Interview settings
    st.markdown("#### 🎤 Interview Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interview_types = st.multiselect("Available Interview Types", 
                                       ["Phone Screen", "Video Interview", "In-Person", "Technical Assessment"])
        default_duration = st.number_input("Default Interview Duration (minutes)", value=60, min_value=15, max_value=180)
        calendar_integration = st.checkbox("Calendar Integration", value=True)
    
    with col2:
        automated_scheduling = st.checkbox("Automated Interview Scheduling", value=True)
        feedback_required = st.checkbox("Require Interview Feedback", value=True)
        panel_interviews = st.checkbox("Enable Panel Interviews", value=True)
    
    # Notification settings
    st.markdown("#### 🔔 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        candidate_notifications = st.checkbox("Candidate Status Notifications", value=True)
        hiring_manager_alerts = st.checkbox("Hiring Manager Alerts", value=True)
        application_confirmations = st.checkbox("Application Confirmations", value=True)
    
    with col2:
        interview_reminders = st.checkbox("Interview Reminders", value=True)
        offer_notifications = st.checkbox("Offer Notifications", value=True)
        rejection_notifications = st.checkbox("Rejection Notifications", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Recruitment settings saved successfully!")

if __name__ == "__main__":
    render_recruitment_dashboard()
