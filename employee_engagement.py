import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from ollama_ai_service import ai_service
import warnings
warnings.filterwarnings('ignore')

def render_engagement_dashboard():
    """Main Employee Engagement & Surveys Dashboard"""
    st.title("💝 Employee Engagement & Surveys")
    
    # Initialize session state
    if 'surveys' not in st.session_state:
        st.session_state.surveys = []
    if 'survey_responses' not in st.session_state:
        st.session_state.survey_responses = []
    if 'recognition_data' not in st.session_state:
        st.session_state.recognition_data = []
    
    # Sidebar navigation
    st.sidebar.markdown("### 🎯 Engagement Tools")
    engagement_page = st.sidebar.radio("Select Module", [
        "📊 Engagement Dashboard",
        "📝 Pulse Surveys",
        "💬 Anonymous Feedback",
        "🏆 Peer Recognition",
        "📈 Sentiment Analysis",
        "⚙️ Survey Settings"
    ])
    
    if engagement_page == "📊 Engagement Dashboard":
        render_engagement_overview()
    elif engagement_page == "📝 Pulse Surveys":
        render_pulse_surveys()
    elif engagement_page == "💬 Anonymous Feedback":
        render_anonymous_feedback()
    elif engagement_page == "🏆 Peer Recognition":
        render_peer_recognition()
    elif engagement_page == "📈 Sentiment Analysis":
        render_sentiment_analysis()
    elif engagement_page == "⚙️ Survey Settings":
        render_survey_settings()

def render_engagement_overview():
    """Engagement Dashboard Overview"""
    st.header("📊 Employee Engagement Overview")
    
    # Generate sample data if empty
    if not st.session_state.survey_responses:
        generate_sample_engagement_data()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Engagement", "78%", "↑ 5%")
    with col2:
        st.metric("Survey Participation", "85%", "↑ 12%")
    with col3:
        st.metric("Recognition Points", "2,450", "↑ 18%")
    with col4:
        st.metric("Sentiment Score", "0.72", "↑ 0.08")
    
    # Engagement Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Engagement Trends")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        engagement_scores = np.random.normal(75, 10, len(dates))
        engagement_scores = np.clip(engagement_scores, 0, 100)
        
        fig = px.line(x=dates, y=engagement_scores, 
                     title="Weekly Engagement Scores",
                     labels={'x': 'Date', 'y': 'Engagement %'})
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Target: 70%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Department Engagement")
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        scores = [82, 75, 88, 79, 73]
        
        fig = px.bar(x=departments, y=scores,
                    title="Engagement by Department",
                    color=scores, color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("🔄 Recent Activity")
    activities = [
        {"time": "2 hours ago", "activity": "New pulse survey launched", "type": "survey"},
        {"time": "4 hours ago", "activity": "John received recognition from Sarah", "type": "recognition"},
        {"time": "6 hours ago", "activity": "Anonymous feedback submitted", "type": "feedback"},
        {"time": "1 day ago", "activity": "Monthly engagement report generated", "type": "report"}
    ]
    
    for activity in activities:
        icon = "📝" if activity["type"] == "survey" else "🏆" if activity["type"] == "recognition" else "💬" if activity["type"] == "feedback" else "📊"
        st.write(f"{icon} **{activity['time']}** - {activity['activity']}")

def render_pulse_surveys():
    """Pulse Surveys Management"""
    st.header("📝 Pulse Surveys")
    
    tab1, tab2, tab3 = st.tabs(["📋 Active Surveys", "➕ Create Survey", "📊 Survey Analytics"])
    
    with tab1:
        st.subheader("Active Surveys")
        
        # Sample active surveys
        active_surveys = [
            {"id": 1, "title": "Q4 Engagement Check", "responses": 45, "target": 60, "status": "Active"},
            {"id": 2, "title": "Remote Work Satisfaction", "responses": 32, "target": 50, "status": "Active"},
            {"id": 3, "title": "Manager Feedback", "responses": 28, "target": 40, "status": "Draft"}
        ]
        
        for survey in active_surveys:
            with st.expander(f"📋 {survey['title']} - {survey['responses']}/{survey['target']} responses"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Rate", f"{(survey['responses']/survey['target']*100):.1f}%")
                with col2:
                    st.metric("Status", survey['status'])
                with col3:
                    if st.button(f"View Results", key=f"view_{survey['id']}"):
                        st.success("Opening survey results...")
    
    with tab2:
        st.subheader("Create New Survey")
        
        survey_title = st.text_input("Survey Title")
        survey_description = st.text_area("Description")
        
        st.write("**Survey Questions:**")
        question_type = st.selectbox("Question Type", 
                                   ["Rating Scale", "Multiple Choice", "Text Response", "Yes/No"])
        
        if question_type == "Rating Scale":
            question = st.text_input("Question")
            scale_min = st.number_input("Min Value", value=1)
            scale_max = st.number_input("Max Value", value=5)
        elif question_type == "Multiple Choice":
            question = st.text_input("Question")
            options = st.text_area("Options (one per line)")
        else:
            question = st.text_input("Question")
        
        col1, col2 = st.columns(2)
        with col1:
            anonymous = st.checkbox("Anonymous Survey", value=True)
        with col2:
            target_groups = st.multiselect("Target Groups", 
                                         ["All Employees", "Engineering", "Sales", "Marketing"])
        
        if st.button("Create Survey"):
            st.success("Survey created successfully!")
    
    with tab3:
        st.subheader("📊 Survey Analytics")
        
        # Survey completion trends
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        completion_rates = np.random.uniform(70, 95, 12)
        
        fig = px.line(x=dates, y=completion_rates,
                     title="Survey Completion Rates Over Time",
                     labels={'x': 'Month', 'y': 'Completion Rate %'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Question effectiveness
        st.write("**Most Effective Questions:**")
        questions = [
            {"question": "How satisfied are you with your role?", "response_rate": 95, "engagement": 0.85},
            {"question": "Do you feel valued by your manager?", "response_rate": 92, "engagement": 0.78},
            {"question": "Would you recommend this company?", "response_rate": 88, "engagement": 0.82}
        ]
        
        df = pd.DataFrame(questions)
        st.dataframe(df, use_container_width=True)

def render_anonymous_feedback():
    """Anonymous Feedback System"""
    st.header("💬 Anonymous Feedback")
    
    tab1, tab2 = st.tabs(["📝 Submit Feedback", "📊 Feedback Analytics"])
    
    with tab1:
        st.subheader("Submit Anonymous Feedback")
        
        feedback_category = st.selectbox("Category", 
                                       ["General", "Management", "Work Environment", 
                                        "Career Development", "Compensation", "Other"])
        
        feedback_text = st.text_area("Your Feedback", 
                                   placeholder="Share your thoughts anonymously...")
        
        urgency = st.select_slider("Urgency Level", 
                                 options=["Low", "Medium", "High", "Critical"])
        
        if st.button("Submit Feedback"):
            # Perform sentiment analysis
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(feedback_text)
            
            feedback_data = {
                "timestamp": datetime.now(),
                "category": feedback_category,
                "text": feedback_text,
                "urgency": urgency,
                "sentiment": sentiment
            }
            
            st.session_state.survey_responses.append(feedback_data)
            st.success("Thank you for your feedback! It has been submitted anonymously.")
    
    with tab2:
        st.subheader("📊 Feedback Analytics")
        
        if st.session_state.survey_responses:
            # Sentiment distribution
            sentiments = [resp.get('sentiment', {}) for resp in st.session_state.survey_responses]
            if sentiments:
                sentiment_scores = [s.get('compound', 0) for s in sentiments if s]
                
                fig = px.histogram(x=sentiment_scores, nbins=20,
                                 title="Feedback Sentiment Distribution",
                                 labels={'x': 'Sentiment Score', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        categories = ["Management", "Work Environment", "Career Development", "Compensation", "Other"]
        feedback_counts = np.random.randint(5, 25, len(categories))
        
        fig = px.pie(values=feedback_counts, names=categories,
                    title="Feedback by Category")
        st.plotly_chart(fig, use_container_width=True)

def render_peer_recognition():
    """Peer-to-Peer Recognition System"""
    st.header("🏆 Peer Recognition")
    
    tab1, tab2, tab3 = st.tabs(["🎖️ Give Recognition", "🏅 Leaderboard", "📊 Recognition Analytics"])
    
    with tab1:
        st.subheader("Give Recognition")
        
        col1, col2 = st.columns(2)
        with col1:
            recipient = st.selectbox("Recognize Employee", 
                                   ["John Smith", "Sarah Johnson", "Mike Chen", "Lisa Wang"])
            recognition_type = st.selectbox("Recognition Type",
                                          ["Great Work", "Team Player", "Innovation", 
                                           "Leadership", "Customer Focus", "Problem Solving"])
        
        with col2:
            points = st.slider("Points to Award", 1, 100, 25)
            badge = st.selectbox("Badge", 
                               ["⭐ Star Performer", "🚀 Innovator", "🤝 Team Player", 
                                "💡 Problem Solver", "🎯 Goal Crusher"])
        
        message = st.text_area("Recognition Message")
        
        if st.button("Send Recognition"):
            recognition_data = {
                "timestamp": datetime.now(),
                "recipient": recipient,
                "type": recognition_type,
                "points": points,
                "badge": badge,
                "message": message,
                "sender": "Current User"
            }
            
            st.session_state.recognition_data.append(recognition_data)
            st.success(f"Recognition sent to {recipient}! 🎉")
    
    with tab2:
        st.subheader("🏅 Recognition Leaderboard")
        
        # Generate sample leaderboard data
        employees = ["Sarah Johnson", "Mike Chen", "John Smith", "Lisa Wang", "Alex Brown"]
        points = [450, 380, 320, 290, 250]
        badges = [12, 9, 8, 7, 6]
        
        leaderboard_df = pd.DataFrame({
            "Rank": range(1, 6),
            "Employee": employees,
            "Total Points": points,
            "Badges Earned": badges,
            "Recent Recognition": ["Innovation Award", "Team Player", "Great Work", 
                                 "Customer Focus", "Problem Solver"]
        })
        
        st.dataframe(leaderboard_df, use_container_width=True)
        
        # Points distribution
        fig = px.bar(x=employees, y=points,
                    title="Recognition Points by Employee",
                    color=points, color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Recognition Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recognition trends
            dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
            recognition_counts = np.random.poisson(15, 12)
            
            fig = px.line(x=dates, y=recognition_counts,
                         title="Monthly Recognition Activity",
                         labels={'x': 'Month', 'y': 'Recognitions Given'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recognition types
            types = ["Great Work", "Team Player", "Innovation", "Leadership", "Customer Focus"]
            type_counts = np.random.randint(10, 30, len(types))
            
            fig = px.pie(values=type_counts, names=types,
                        title="Recognition Types Distribution")
            st.plotly_chart(fig, use_container_width=True)

def render_sentiment_analysis():
    """Advanced Sentiment Analysis"""
    st.header("📈 Sentiment Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Deep Analysis", "📈 Predictions"])
    
    with tab1:
        st.subheader("Sentiment Overview")
        
        # Overall sentiment metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Sentiment", "0.72", "↑ 0.08")
        with col2:
            st.metric("Positive Feedback", "68%", "↑ 5%")
        with col3:
            st.metric("Neutral Feedback", "22%", "↓ 2%")
        with col4:
            st.metric("Negative Feedback", "10%", "↓ 3%")
        
        # Sentiment trends
        dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
        sentiment_scores = np.random.normal(0.6, 0.2, 52)
        
        fig = px.line(x=dates, y=sentiment_scores,
                     title="Weekly Sentiment Trends",
                     labels={'x': 'Week', 'y': 'Sentiment Score'})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Deep Sentiment Analysis")
        
        # Word cloud
        st.write("**Most Common Positive Words:**")
        positive_words = ["excellent", "great", "amazing", "helpful", "supportive", 
                         "collaborative", "innovative", "efficient", "productive"]
        
        # Simulate word cloud data
        if st.button("Generate Word Cloud"):
            wordcloud_text = " ".join(positive_words * np.random.randint(1, 10, len(positive_words)))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        
        # Sentiment by department
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        dept_sentiments = np.random.uniform(0.4, 0.8, len(departments))
        
        fig = px.bar(x=departments, y=dept_sentiments,
                    title="Average Sentiment by Department",
                    color=dept_sentiments, color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Sentiment Predictions")
        
        # Generate prediction data using Prophet
        if st.button("Generate Sentiment Forecast"):
            # Create sample historical data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
            sentiment_data = pd.DataFrame({
                'ds': dates,
                'y': np.random.normal(0.65, 0.15, len(dates))
            })
            
            # Fit Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(sentiment_data)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=12, freq='W')
            forecast = model.predict(future)
            
            # Plot forecast
            fig = px.line(forecast, x='ds', y='yhat',
                         title="Sentiment Score Forecast (Next 12 Weeks)")
            fig.add_scatter(x=sentiment_data['ds'], y=sentiment_data['y'], 
                           mode='markers', name='Historical Data')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📊 Predicted sentiment trend shows stable engagement with seasonal variations.")

def render_survey_settings():
    """Survey and Engagement Settings"""
    st.header("⚙️ Survey Settings")
    
    tab1, tab2, tab3 = st.tabs(["📋 Survey Config", "🔔 Notifications", "📊 Analytics Config"])
    
    with tab1:
        st.subheader("Survey Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default Settings:**")
            default_anonymous = st.checkbox("Anonymous by Default", value=True)
            auto_reminders = st.checkbox("Automatic Reminders", value=True)
            response_limit = st.number_input("Response Time Limit (days)", value=7)
        
        with col2:
            st.write("**Survey Templates:**")
            template_type = st.selectbox("Template Category",
                                       ["Engagement", "Satisfaction", "360 Feedback", "Exit Interview"])
            
            if st.button("Load Template"):
                st.success(f"Loaded {template_type} template")
    
    with tab2:
        st.subheader("Notification Settings")
        
        st.write("**Email Notifications:**")
        notify_new_survey = st.checkbox("New Survey Launch", value=True)
        notify_responses = st.checkbox("Response Milestones", value=True)
        notify_completion = st.checkbox("Survey Completion", value=True)
        
        st.write("**Slack Integration:**")
        slack_webhook = st.text_input("Slack Webhook URL")
        slack_channel = st.text_input("Default Channel", value="#hr-updates")
    
    with tab3:
        st.subheader("Analytics Configuration")
        
        st.write("**Sentiment Analysis:**")
        sentiment_model = st.selectbox("Sentiment Model", ["VADER", "DistilBERT", "Custom"])
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        
        st.write("**Reporting:**")
        auto_reports = st.checkbox("Automatic Monthly Reports", value=True)
        report_recipients = st.text_area("Report Recipients (emails)")

def generate_sample_engagement_data():
    """Generate sample engagement data for demonstration"""
    # Sample survey responses
    sample_responses = []
    for i in range(50):
        response = {
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 30)),
            "employee_id": f"EMP{i+1:03d}",
            "survey_id": np.random.choice([1, 2, 3]),
            "responses": {
                "satisfaction": np.random.randint(1, 6),
                "engagement": np.random.randint(1, 6),
                "recommendation": np.random.randint(1, 6)
            },
            "sentiment": {
                "compound": np.random.uniform(-0.5, 0.8),
                "positive": np.random.uniform(0.1, 0.9),
                "negative": np.random.uniform(0.0, 0.3),
                "neutral": np.random.uniform(0.1, 0.5)
            }
        }
        sample_responses.append(response)
    
    st.session_state.survey_responses = sample_responses
    
    # Sample recognition data
    sample_recognition = []
    for i in range(20):
        recognition = {
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 30)),
            "recipient": f"Employee {i+1}",
            "sender": f"Employee {np.random.randint(1, 50)}",
            "type": np.random.choice(["Great Work", "Team Player", "Innovation", "Leadership"]),
            "points": np.random.randint(10, 50),
            "message": "Great job on the project!"
        }
        sample_recognition.append(recognition)
    
    st.session_state.recognition_data = sample_recognition

if __name__ == "__main__":
    render_engagement_dashboard()
