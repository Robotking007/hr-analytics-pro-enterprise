import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def render_compliance_dashboard():
    """Main Compliance & Global HR Support Dashboard"""
    st.title("⚖️ Compliance & Global HR Support")
    
    # Initialize session state
    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = []
    if 'compliance_rules' not in st.session_state:
        st.session_state.compliance_rules = []
    
    # Sidebar navigation
    st.sidebar.markdown("### 🌍 Compliance Tools")
    compliance_page = st.sidebar.radio("Select Module", [
        "📊 Compliance Dashboard",
        "📋 Labor Law Updates",
        "🔍 Audit Logs",
        "📊 Compliance Reports",
        "🌍 Multi-Country Support",
        "⚙️ Compliance Settings"
    ])
    
    if compliance_page == "📊 Compliance Dashboard":
        render_compliance_overview()
    elif compliance_page == "📋 Labor Law Updates":
        render_labor_law_updates()
    elif compliance_page == "🔍 Audit Logs":
        render_audit_logs()
    elif compliance_page == "📊 Compliance Reports":
        render_compliance_reports()
    elif compliance_page == "🌍 Multi-Country Support":
        render_multi_country_support()
    elif compliance_page == "⚙️ Compliance Settings":
        render_compliance_settings()

def render_compliance_overview():
    """Compliance Dashboard Overview"""
    st.header("📊 Compliance Overview")
    
    # Generate sample data if empty
    if not st.session_state.audit_logs:
        generate_sample_compliance_data()
    
    # Key Compliance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Compliance Score", "94%", "↑ 2%")
    with col2:
        st.metric("Active Violations", "3", "↓ 5")
    with col3:
        st.metric("Audit Actions", "127", "↑ 12")
    with col4:
        st.metric("Risk Level", "Low", "↓ Medium")
    
    # Compliance Status by Country
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Global Compliance Status")
        countries = ['United States', 'United Kingdom', 'Canada', 'Germany', 'India']
        compliance_scores = [96, 94, 98, 92, 89]
        
        fig = px.bar(x=countries, y=compliance_scores,
                    title="Compliance Scores by Country",
                    color=compliance_scores, color_continuous_scale="RdYlGn")
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Minimum: 90%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Assessment")
        risk_categories = ['Payroll', 'Benefits', 'Time Tracking', 'Data Privacy', 'Safety']
        risk_scores = [15, 25, 10, 30, 20]
        
        fig = px.pie(values=risk_scores, names=risk_categories,
                    title="Risk Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Compliance Alerts
    st.subheader("🚨 Recent Compliance Alerts")
    alerts = [
        {"time": "2 hours ago", "type": "Warning", "message": "Overtime limit exceeded for 3 employees in UK", "severity": "Medium"},
        {"time": "1 day ago", "type": "Info", "message": "New GDPR update available for review", "severity": "Low"},
        {"time": "2 days ago", "type": "Critical", "message": "Missing break records for CA employees", "severity": "High"}
    ]
    
    for alert in alerts:
        severity_color = "🔴" if alert["severity"] == "High" else "🟡" if alert["severity"] == "Medium" else "🟢"
        st.write(f"{severity_color} **{alert['time']}** - {alert['message']}")

def render_labor_law_updates():
    """Labor Law Updates and Monitoring"""
    st.header("📋 Labor Law Updates")
    
    tab1, tab2, tab3 = st.tabs(["📰 Recent Updates", "🔍 Law Search", "📊 Impact Analysis"])
    
    with tab1:
        st.subheader("Recent Labor Law Updates")
        
        # Sample law updates
        updates = [
            {
                "date": "2024-08-15",
                "country": "United States",
                "title": "Federal Minimum Wage Update",
                "summary": "New federal minimum wage requirements effective January 2025",
                "impact": "High",
                "status": "Action Required"
            },
            {
                "date": "2024-08-10",
                "country": "United Kingdom",
                "title": "Working Time Regulations Amendment",
                "summary": "Updated break requirements for shift workers",
                "impact": "Medium",
                "status": "Under Review"
            },
            {
                "date": "2024-08-05",
                "country": "Canada",
                "title": "Employment Standards Act Update",
                "summary": "New parental leave provisions",
                "impact": "Medium",
                "status": "Implemented"
            }
        ]
        
        for update in updates:
            with st.expander(f"🏛️ {update['title']} - {update['country']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Date:** {update['date']}")
                    st.write(f"**Impact:** {update['impact']}")
                with col2:
                    st.write(f"**Status:** {update['status']}")
                    st.write(f"**Country:** {update['country']}")
                with col3:
                    if st.button(f"View Details", key=f"law_{update['date']}"):
                        st.info("Opening detailed law analysis...")
                
                st.write(f"**Summary:** {update['summary']}")
    
    with tab2:
        st.subheader("🔍 Labor Law Search")
        
        col1, col2 = st.columns(2)
        with col1:
            search_country = st.selectbox("Country", 
                                        ["All Countries", "United States", "United Kingdom", 
                                         "Canada", "Germany", "India"])
            search_category = st.selectbox("Category",
                                         ["All Categories", "Minimum Wage", "Working Hours", 
                                          "Leave Policies", "Safety", "Discrimination"])
        
        with col2:
            search_query = st.text_input("Search Keywords")
            date_range = st.date_input("Date Range", value=[datetime.now() - timedelta(days=90), datetime.now()])
        
        if st.button("Search Laws"):
            st.success("Search completed - 12 relevant laws found")
    
    with tab3:
        st.subheader("📊 Law Impact Analysis")
        
        # Impact by category
        categories = ['Payroll', 'Benefits', 'Time Tracking', 'Safety', 'Privacy']
        impact_scores = np.random.randint(1, 10, len(categories))
        
        fig = px.bar(x=categories, y=impact_scores,
                    title="Law Change Impact by HR Category",
                    color=impact_scores, color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

def render_audit_logs():
    """Comprehensive Audit Logging"""
    st.header("🔍 Audit Logs")
    
    tab1, tab2, tab3 = st.tabs(["📋 Recent Actions", "🔍 Search Logs", "📊 Audit Analytics"])
    
    with tab1:
        st.subheader("Recent HR Actions")
        
        # Generate sample audit logs
        if not st.session_state.audit_logs:
            generate_sample_audit_logs()
        
        # Display recent logs
        recent_logs = sorted(st.session_state.audit_logs, 
                           key=lambda x: x['timestamp'], reverse=True)[:20]
        
        for log in recent_logs:
            severity_icon = "🔴" if log['severity'] == "High" else "🟡" if log['severity'] == "Medium" else "🟢"
            st.write(f"{severity_icon} **{log['timestamp'].strftime('%Y-%m-%d %H:%M')}** - "
                    f"{log['user']} performed {log['action']} on {log['resource']}")
    
    with tab2:
        st.subheader("🔍 Search Audit Logs")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_user = st.text_input("User")
            search_action = st.selectbox("Action Type", 
                                       ["All", "Create", "Update", "Delete", "View", "Export"])
        with col2:
            search_resource = st.selectbox("Resource", 
                                         ["All", "Employee", "Payroll", "Benefits", "Attendance"])
            date_filter = st.date_input("Date Range", 
                                      value=[datetime.now() - timedelta(days=30), datetime.now()])
        with col3:
            severity_filter = st.multiselect("Severity", ["Low", "Medium", "High"])
        
        if st.button("Search Logs"):
            st.success("Search completed - 45 matching audit entries found")
    
    with tab3:
        st.subheader("📊 Audit Analytics")
        
        # Action frequency
        actions = ['View', 'Update', 'Create', 'Delete', 'Export']
        action_counts = np.random.poisson(20, len(actions))
        
        fig = px.bar(x=actions, y=action_counts,
                    title="Most Common Actions (Last 30 Days)",
                    color=action_counts, color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        # User activity heatmap
        st.write("**User Activity Heatmap:**")
        users = [f"User{i}" for i in range(1, 8)]
        hours = list(range(24))
        activity_data = np.random.poisson(2, (len(users), len(hours)))
        
        fig = px.imshow(activity_data, x=hours, y=users,
                       title="User Activity by Hour",
                       labels={'x': 'Hour of Day', 'y': 'User', 'color': 'Actions'})
        st.plotly_chart(fig, use_container_width=True)

def render_compliance_reports():
    """Export-Ready Compliance Reports"""
    st.header("📊 Compliance Reports")
    
    tab1, tab2, tab3 = st.tabs(["📋 Generate Reports", "📈 Compliance Trends", "🔍 Violation Analysis"])
    
    with tab1:
        st.subheader("Generate Compliance Reports")
        
        col1, col2 = st.columns(2)
        with col1:
            report_type = st.selectbox("Report Type", [
                "Full Compliance Audit",
                "Payroll Compliance",
                "Time & Attendance Compliance",
                "Data Privacy Compliance",
                "Safety Compliance"
            ])
            
            report_period = st.selectbox("Period", 
                                       ["Last Month", "Last Quarter", "Last Year", "Custom"])
            
            if report_period == "Custom":
                custom_dates = st.date_input("Custom Date Range", 
                                           value=[datetime.now() - timedelta(days=90), datetime.now()])
        
        with col2:
            report_format = st.selectbox("Format", ["PDF", "Excel", "CSV"])
            include_recommendations = st.checkbox("Include AI Recommendations", value=True)
            include_charts = st.checkbox("Include Visualizations", value=True)
        
        if st.button("Generate Report"):
            with st.spinner("Generating compliance report..."):
                # Simulate report generation
                st.success("✅ Compliance report generated successfully!")
                st.download_button(
                    label="📥 Download Report",
                    data="Sample compliance report data",
                    file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
    
    with tab2:
        st.subheader("📈 Compliance Trends")
        
        # Compliance score trends
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        compliance_scores = np.random.uniform(85, 98, 12)
        
        fig = px.line(x=dates, y=compliance_scores,
                     title="Monthly Compliance Scores",
                     labels={'x': 'Month', 'y': 'Compliance Score %'})
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="Minimum Required: 90%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Violation trends by category
        categories = ['Payroll', 'Time Tracking', 'Benefits', 'Safety', 'Privacy']
        violation_data = []
        for month in range(12):
            for cat in categories:
                violation_data.append({
                    'Month': dates[month],
                    'Category': cat,
                    'Violations': np.random.poisson(2)
                })
        
        violation_df = pd.DataFrame(violation_data)
        fig = px.line(violation_df, x='Month', y='Violations', color='Category',
                     title="Violations by Category Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Violation Analysis")
        
        # AI-powered anomaly detection
        if st.button("Run Anomaly Detection"):
            with st.spinner("Analyzing compliance data for anomalies..."):
                # Simulate anomaly detection
                anomalies = detect_compliance_anomalies()
                
                st.write("**Detected Anomalies:**")
                for anomaly in anomalies:
                    severity_icon = "🔴" if anomaly['severity'] == "High" else "🟡"
                    st.write(f"{severity_icon} {anomaly['description']} - Risk Score: {anomaly['risk_score']:.2f}")

def render_multi_country_support():
    """Multi-Country HR Compliance Support"""
    st.header("🌍 Multi-Country HR Support")
    
    tab1, tab2, tab3 = st.tabs(["🌐 Country Overview", "📋 Local Requirements", "🔄 Sync Status"])
    
    with tab1:
        st.subheader("Global Operations Overview")
        
        # Country compliance matrix
        countries = ['United States', 'United Kingdom', 'Canada', 'Germany', 'India', 'Australia']
        compliance_areas = ['Payroll', 'Benefits', 'Time Tracking', 'Privacy', 'Safety']
        
        # Generate compliance matrix
        compliance_matrix = np.random.choice([0, 1], size=(len(countries), len(compliance_areas)), p=[0.1, 0.9])
        
        fig = px.imshow(compliance_matrix, 
                       x=compliance_areas, y=countries,
                       title="Compliance Status Matrix",
                       color_continuous_scale="RdYlGn",
                       labels={'color': 'Compliant'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Country-specific metrics
        st.write("**Country-Specific Metrics:**")
        country_data = []
        for country in countries:
            country_data.append({
                'Country': country,
                'Employees': np.random.randint(50, 500),
                'Compliance Score': np.random.uniform(85, 98),
                'Last Audit': (datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d'),
                'Risk Level': np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.25, 0.05])
            })
        
        country_df = pd.DataFrame(country_data)
        st.dataframe(country_df, use_container_width=True)
    
    with tab2:
        st.subheader("📋 Local Requirements by Country")
        
        selected_country = st.selectbox("Select Country", countries)
        
        # Country-specific requirements
        requirements = {
            'United States': {
                'Minimum Wage': '$7.25/hour (Federal)',
                'Overtime': '1.5x after 40 hours/week',
                'Vacation': 'No federal requirement',
                'Sick Leave': 'Varies by state',
                'Maternity': '12 weeks unpaid (FMLA)'
            },
            'United Kingdom': {
                'Minimum Wage': '£10.42/hour',
                'Overtime': 'No statutory requirement',
                'Vacation': '28 days minimum',
                'Sick Leave': 'SSP after 4 days',
                'Maternity': '52 weeks (39 paid)'
            },
            'Canada': {
                'Minimum Wage': 'Varies by province',
                'Overtime': '1.5x after 44 hours/week',
                'Vacation': '2 weeks minimum',
                'Sick Leave': 'Varies by province',
                'Maternity': '17-18 weeks'
            }
        }
        
        if selected_country in requirements:
            req_data = requirements[selected_country]
            for key, value in req_data.items():
                st.write(f"**{key}:** {value}")
        
        # Compliance checklist
        st.write("**Compliance Checklist:**")
        checklist_items = [
            "Payroll calculations comply with local minimum wage",
            "Overtime rules properly implemented",
            "Leave policies meet local requirements",
            "Tax withholdings are accurate",
            "Employee records are complete"
        ]
        
        for item in checklist_items:
            checked = st.checkbox(item, value=np.random.choice([True, False], p=[0.8, 0.2]))
    
    with tab3:
        st.subheader("🔄 System Sync Status")
        
        # Sync status for different systems
        systems = ['Payroll System', 'Benefits Platform', 'Time Tracking', 'HRIS', 'Tax Engine']
        sync_status = []
        
        for system in systems:
            status = np.random.choice(['Synced', 'Pending', 'Error'], p=[0.8, 0.15, 0.05])
            last_sync = datetime.now() - timedelta(minutes=np.random.randint(1, 1440))
            
            sync_status.append({
                'System': system,
                'Status': status,
                'Last Sync': last_sync.strftime('%Y-%m-%d %H:%M'),
                'Records': np.random.randint(100, 5000)
            })
        
        sync_df = pd.DataFrame(sync_status)
        st.dataframe(sync_df, use_container_width=True)
        
        if st.button("Force Sync All"):
            st.success("Synchronization initiated for all systems")

def render_compliance_settings():
    """Compliance System Settings"""
    st.header("⚙️ Compliance Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔔 Alerts", "🤖 AI Config", "📊 Reporting"])
    
    with tab1:
        st.subheader("Compliance Alerts Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Alert Thresholds:**")
            compliance_threshold = st.slider("Minimum Compliance Score", 0, 100, 90)
            risk_threshold = st.slider("Risk Score Alert Level", 0.0, 1.0, 0.7)
            violation_threshold = st.number_input("Max Violations per Month", value=5)
        
        with col2:
            st.write("**Notification Settings:**")
            email_alerts = st.checkbox("Email Alerts", value=True)
            slack_alerts = st.checkbox("Slack Notifications", value=True)
            dashboard_alerts = st.checkbox("Dashboard Notifications", value=True)
    
    with tab2:
        st.subheader("🤖 AI Configuration")
        
        st.write("**Anomaly Detection:**")
        anomaly_sensitivity = st.slider("Anomaly Detection Sensitivity", 0.0, 1.0, 0.1)
        
        st.write("**Risk Scoring:**")
        risk_model = st.selectbox("Risk Model", ["Random Forest", "Gradient Boosting", "Neural Network"])
        risk_features = st.multiselect("Risk Factors", 
                                     ["Payroll Variance", "Attendance Patterns", "Policy Violations", 
                                      "Audit History", "Country Risk"])
    
    with tab3:
        st.subheader("📊 Reporting Configuration")
        
        st.write("**Automatic Reports:**")
        monthly_reports = st.checkbox("Monthly Compliance Reports", value=True)
        quarterly_audits = st.checkbox("Quarterly Audit Reports", value=True)
        
        report_recipients = st.text_area("Report Recipients (emails)")
        
        st.write("**Report Customization:**")
        include_trends = st.checkbox("Include Trend Analysis", value=True)
        include_predictions = st.checkbox("Include Risk Predictions", value=True)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)

def detect_compliance_anomalies():
    """AI-powered compliance anomaly detection"""
    # Simulate anomaly detection using Isolation Forest
    
    # Generate sample compliance data
    n_samples = 1000
    features = np.random.normal(0, 1, (n_samples, 5))  # 5 compliance features
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    features[anomaly_indices] += np.random.normal(3, 1, (len(anomaly_indices), 5))
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features)
    
    # Generate anomaly descriptions
    anomalies = []
    anomaly_descriptions = [
        "Unusual payroll variance detected in Engineering department",
        "Abnormal overtime patterns in Sales team",
        "Irregular benefit enrollment activity",
        "Suspicious data access patterns detected",
        "Unusual leave request patterns in Q4"
    ]
    
    for i, desc in enumerate(anomaly_descriptions):
        anomalies.append({
            'description': desc,
            'risk_score': np.random.uniform(0.6, 0.95),
            'severity': np.random.choice(['Medium', 'High'], p=[0.7, 0.3])
        })
    
    return anomalies[:3]  # Return top 3 anomalies

def generate_sample_compliance_data():
    """Generate sample compliance and audit data"""
    # Sample audit logs
    actions = ['Create', 'Update', 'Delete', 'View', 'Export']
    resources = ['Employee', 'Payroll', 'Benefits', 'Attendance', 'Performance']
    users = ['admin@company.com', 'hr@company.com', 'manager@company.com']
    
    sample_logs = []
    for i in range(100):
        log = {
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 30)),
            "user": np.random.choice(users),
            "action": np.random.choice(actions),
            "resource": np.random.choice(resources),
            "severity": np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]),
            "ip_address": f"192.168.1.{np.random.randint(1, 255)}",
            "details": "Sample audit log entry"
        }
        sample_logs.append(log)
    
    st.session_state.audit_logs = sample_logs

if __name__ == "__main__":
    render_compliance_dashboard()
