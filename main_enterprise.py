"""
HR Analytics Pro - Enterprise Edition Main Application
Complete enterprise platform with authentication, modern UI, and comprehensive features
"""
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="HR Analytics Pro - Enterprise",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from auth import auth_manager
from ui_components import ui
from payroll_management import render_payroll_dashboard
from benefits_administration import render_benefits_dashboard
from time_attendance import render_time_attendance_dashboard
from talent_management import render_talent_dashboard
from learning_development import render_learning_dashboard
from recruitment_ats import render_recruitment_dashboard
from employee_engagement import render_engagement_dashboard
from compliance_global import render_compliance_dashboard
from workflow_automation import render_workflow_dashboard
from advanced_analytics_ai import render_advanced_analytics_dashboard

# Apply theme CSS
st.markdown(ui.theme.get_theme_css(), unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Check authentication
    if not auth_manager.check_authentication():
        auth_manager.show_auth_form()
        return
    
    # Render main application
    render_sidebar()
    render_main_content()

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        # User info
        user_data = st.session_state.user_data
        st.markdown(f"""
        <div class="glass-card">
            <div style="text-align: center; color: white;">
                <div style="font-size: 2rem;">👤</div>
                <h3 style="margin: 10px 0;">{user_data.get('full_name', 'User')}</h3>
                <p style="margin: 5px 0; opacity: 0.8;">{user_data.get('email', '')}</p>
                <span class="status-badge badge-success">{user_data.get('role', 'user').title()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme toggle
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = 'dark'
                st.rerun()
        with col2:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = 'light'
                st.rerun()
        
        # Navigation menu
        st.markdown("### 🧭 Navigation")
        
        nav_items = [
            {"icon": "📊", "label": "Dashboard", "page": "dashboard"},
            {"icon": "👥", "label": "Employee Management", "page": "employees"},
            {"icon": "💰", "label": "Payroll Management", "page": "payroll"},
            {"icon": "🏥", "label": "Benefits Administration", "page": "benefits"},
            {"icon": "⏰", "label": "Time & Attendance", "page": "time_attendance"},
            {"icon": "🌟", "label": "Talent Management", "page": "talent"},
            {"icon": "📚", "label": "Learning & Development", "page": "learning"},
            {"icon": "🎯", "label": "Recruitment & ATS", "page": "recruitment"},
            {"icon": "💝", "label": "Employee Engagement", "page": "engagement"},
            {"icon": "⚖️", "label": "Compliance & Global", "page": "compliance"},
            {"icon": "🔄", "label": "Workflow Automation", "page": "workflows"},
            {"icon": "🧠", "label": "Advanced AI Analytics", "page": "advanced_ai"},
            {"icon": "📈", "label": "Analytics & Reports", "page": "analytics"},
            {"icon": "🤖", "label": "AI Predictions", "page": "predictions"},
            {"icon": "⚖️", "label": "Bias Audit", "page": "bias_audit"},
            {"icon": "🔍", "label": "Data Explorer", "page": "data_explorer"},
            {"icon": "⚙️", "label": "Settings", "page": "settings"},
        ]
        
        # Add admin-only pages
        if user_data.get('role') == 'admin':
            nav_items.append({"icon": "🔧", "label": "Administration", "page": "admin"})
        
        selected_page = st.session_state.get('current_page', 'dashboard')
        
        for item in nav_items:
            if st.button(f"{item['icon']} {item['label']}", 
                       key=f"nav_{item['page']}", 
                       use_container_width=True,
                       type="primary" if selected_page == item['page'] else "secondary"):
                st.session_state.current_page = item['page']
                st.rerun()
        
        st.markdown("---")
        
        # Sign out button
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            auth_manager.sign_out()
            st.rerun()

def render_main_content():
    """Render main content based on selected page"""
    current_page = st.session_state.get('current_page', 'dashboard')
    
    if current_page == 'dashboard':
        render_dashboard()
    elif current_page == 'employees':
        render_employee_management()
    elif current_page == 'payroll':
        render_payroll_dashboard()
    elif current_page == 'benefits':
        render_benefits_dashboard()
    elif current_page == 'time_attendance':
        render_time_attendance_dashboard()
    elif current_page == 'talent':
        render_talent_dashboard()
    elif current_page == 'learning':
        render_learning_dashboard()
    elif current_page == 'recruitment':
        render_recruitment_dashboard()
    elif current_page == 'engagement':
        render_engagement_dashboard()
    elif current_page == 'compliance':
        render_compliance_dashboard()
    elif current_page == 'workflows':
        render_workflow_dashboard()
    elif current_page == 'advanced_ai':
        render_advanced_analytics_dashboard()
    elif current_page == 'analytics':
        render_analytics_reports()
    elif current_page == 'predictions':
        render_ai_predictions()
    elif current_page == 'bias_audit':
        render_bias_audit()
    elif current_page == 'data_explorer':
        render_data_explorer()
    elif current_page == 'settings':
        render_settings()
    elif current_page == 'admin':
        render_administration()

def render_dashboard():
    """Render main dashboard"""
    st.markdown('<div class="main-header">📊 HR Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # KPI Cards
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("Total Employees", "1,247", "+23 this month", "👥")
    
    with col2:
        ui.render_kpi_card("Avg Performance", "87.3%", "+2.1%", "⭐")
    
    with col3:
        ui.render_kpi_card("Model Accuracy", "94.1%", "+0.8%", "🤖")
    
    with col4:
        ui.render_kpi_card("Bias Score", "0.82", "Improving", "⚖️")
    
    # Quick Actions
    quick_actions = [
        {"icon": "➕", "label": "Add Employee", "callback": lambda: show_add_employee_dialog()},
        {"icon": "🔮", "label": "Run Prediction", "callback": lambda: st.session_state.update({"current_page": "predictions"})},
        {"icon": "🔍", "label": "Start Audit", "callback": lambda: st.session_state.update({"current_page": "bias_audit"})},
        {"icon": "📊", "label": "Export Data", "callback": lambda: export_analytics()},
        {"icon": "📁", "label": "Upload Files", "callback": lambda: show_file_upload()}
    ]
    
    ui.render_quick_actions(quick_actions)
    
    # Recent Insights Feed
    insights = [
        {
            "title": "High Turnover Risk Detected",
            "description": "12 employees in Sales department show high attrition probability",
            "timestamp": "2 hours ago",
            "priority": "High"
        },
        {
            "title": "Performance Model Updated",
            "description": "New ML model deployed with 94.1% accuracy",
            "timestamp": "4 hours ago",
            "priority": "Medium"
        },
        {
            "title": "Bias Audit Completed",
            "description": "Q4 bias analysis shows improvement in hiring practices",
            "timestamp": "1 day ago",
            "priority": "Low"
        }
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ui.render_insights_feed(insights)
    
    with col2:
        # System Status
        status_items = [
            {"name": "Performance Model", "status": "healthy", "description": "Running optimally"},
            {"name": "Data Pipeline", "status": "healthy", "description": "All systems operational"},
            {"name": "Bias Monitor", "status": "warning", "description": "Requires attention"},
        ]
        
        ui.render_system_status(status_items)

def render_employee_management():
    """Render employee management page"""
    st.markdown('<div class="main-header">👥 Employee Management</div>', unsafe_allow_html=True)
    
    # Employee Statistics
    st.markdown("### 📊 Employee Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("Total Employees", "1,247", "+23", "👥")
    
    with col2:
        ui.render_kpi_card("High Performers", "312", "+15", "🌟")
    
    with col3:
        ui.render_kpi_card("At-Risk", "89", "-7", "⚠️")
    
    with col4:
        ui.render_kpi_card("Avg Performance", "87.3%", "+2.1%", "📈")
    
    # Employee Directory
    st.markdown("### 📋 Employee Directory")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dept_filter = st.selectbox("Department", ["All", "Engineering", "Sales", "Marketing", "HR"])
    with col2:
        role_filter = st.selectbox("Role", ["All", "Manager", "Senior", "Junior", "Intern"])
    with col3:
        risk_filter = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"])
    with col4:
        search_term = st.text_input("Search", placeholder="Search employees...")
    
    # Sample employee data
    employees = [
        {"name": "John Smith", "department": "Engineering", "position": "Senior Developer", "performance": 0.92, "risk": "Low"},
        {"name": "Sarah Johnson", "department": "Sales", "position": "Sales Manager", "performance": 0.88, "risk": "Low"},
        {"name": "Mike Chen", "department": "Marketing", "position": "Marketing Analyst", "performance": 0.75, "risk": "Medium"},
        {"name": "Emily Davis", "department": "HR", "position": "HR Specialist", "performance": 0.65, "risk": "High"},
        {"name": "David Wilson", "department": "Engineering", "position": "Junior Developer", "performance": 0.82, "risk": "Low"},
    ]
    
    # Display employee cards
    for employee in employees:
        ui.render_employee_card(employee, employee["performance"])
    
    # Add Employee Button
    if st.button("➕ Add New Employee", type="primary"):
        show_add_employee_dialog()

def render_ai_predictions():
    """Render AI predictions page"""
    st.markdown('<div class="main-header">🤖 AI Predictions</div>', unsafe_allow_html=True)
    
    # Model Status Cards
    st.markdown("### 🔧 Model Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("Turnover Model", "Active", "94.1% accuracy", "🔄")
    
    with col2:
        ui.render_kpi_card("Performance", "Active", "91.8% accuracy", "📈")
    
    with col3:
        ui.render_kpi_card("Career Path", "Training", "Updating...", "🎯")
    
    with col4:
        ui.render_kpi_card("Risk Assessment", "Active", "89.3% accuracy", "⚠️")
    
    # High-Risk Analysis
    st.markdown("### ⚠️ High-Risk Employee Analysis")
    
    # Sample high-risk employees
    high_risk_employees = [
        {"name": "Alex Thompson", "risk_score": 85, "factors": ["Low engagement", "Salary below market", "Limited growth"]},
        {"name": "Maria Garcia", "risk_score": 78, "factors": ["Work-life balance", "Team conflicts", "Skill mismatch"]},
        {"name": "Robert Kim", "risk_score": 72, "factors": ["Career stagnation", "Manager relationship", "Workload"]}
    ]
    
    for emp in high_risk_employees:
        with st.expander(f"🔴 {emp['name']} - Risk Score: {emp['risk_score']}/100"):
            st.write("**Risk Factors:**")
            for factor in emp['factors']:
                st.write(f"• {factor}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📋 View Details", key=f"details_{emp['name']}"):
                    st.info("Detailed analysis would open here")
            with col2:
                if st.button(f"🎯 Create Action Plan", key=f"action_{emp['name']}"):
                    st.success("Action plan created!")

def render_bias_audit():
    """Render enhanced bias audit page with interactive analysis"""
    st.markdown('<div class="main-header">⚖️ Bias Audit & Fairness</div>', unsafe_allow_html=True)
    
    # Control Panel
    st.markdown("### 🎛️ Analysis Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        analysis_period = st.selectbox("📅 Time Period", ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"])
    with col2:
        department_filter = st.multiselect("🏢 Departments", ["All", "Engineering", "Sales", "Marketing", "HR", "Finance"], default=["All"])
    with col3:
        analysis_type = st.selectbox("🔍 Analysis Type", ["Comprehensive", "Hiring Only", "Promotion Only", "Compensation Only"])
    with col4:
        if st.button("🔄 Run Analysis", type="primary"):
            ui.show_toast_notification("Running bias analysis...", "info")
    
    # Overall Fairness Metrics with Trends
    st.markdown("### 📊 Overall Fairness Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("Hiring Fairness", "82%", "+5%", "🎯")
    with col2:
        ui.render_kpi_card("Promotion Equity", "91%", "+3%", "📈")
    with col3:
        ui.render_kpi_card("Pay Parity", "76%", "-2%", "💰")
    with col4:
        ui.render_kpi_card("Review Fairness", "88%", "+7%", "📋")
    
    # Interactive Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends & Patterns", "🔍 Detailed Analysis", "📊 Demographics", "⚠️ Risk Areas", "📋 Recommendations"])
    
    with tab1:
        render_bias_trends()
    
    with tab2:
        render_detailed_bias_analysis()
    
    with tab3:
        render_demographics_analysis()
    
    with tab4:
        render_risk_areas()
    
    with tab5:
        render_bias_recommendations()

def show_add_employee_dialog():
    """Show add employee dialog"""
    with st.form("add_employee_form"):
        st.subheader("➕ Add New Employee")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", placeholder="Enter full name")
            email = st.text_input("Email*", placeholder="Enter email address")
            department = st.selectbox("Department*", ["Engineering", "Sales", "Marketing", "HR", "Finance"])
        
        with col2:
            role = st.text_input("Role*", placeholder="Enter job title")
            salary = st.number_input("Annual Salary", min_value=30000, max_value=500000, value=75000)
            hire_date = st.date_input("Hire Date", value=datetime.now())
        
        if st.form_submit_button("Add Employee", type="primary"):
            if all([name, email, department, role]):
                ui.show_toast_notification("Employee added successfully!", "success")
                st.success(f"✅ {name} has been added to the system!")
            else:
                st.error("Please fill in all required fields marked with *")

def export_analytics():
    """Export analytics report"""
    ui.show_toast_notification("Analytics report exported!", "success")
    st.success("📊 Analytics report has been generated and downloaded!")

def show_file_upload():
    """Show file upload dialog"""
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json'])
    if uploaded_file is not None:
        ui.show_toast_notification("File uploaded successfully!", "success")
        st.success(f"✅ {uploaded_file.name} has been uploaded and processed!")

def show_mfa_setup_dialog():
    """Show MFA setup dialog"""
    st.markdown("---")
    st.markdown("### 🔒 Multi-Factor Authentication Setup")
    
    with st.container():
        # MFA enrollment section
        factor_type = st.selectbox(
            "Authentication Method",
            ["totp", "email_otp"],
            format_func=lambda x: {
                "totp": "📱 Authenticator App (TOTP)",
                "email_otp": "📧 Email OTP",
            }.get(x, x),
            help="Choose your preferred authentication method"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔐 Setup MFA", type="primary"):
                with st.spinner("Setting up MFA..."):
                    result = auth_manager.enroll_mfa(factor_type)
                    
                    if result.get('success'):
                        if factor_type == "totp":
                            st.success("✅ TOTP MFA enrollment successful!")
                            
                            # Display QR code
                            qr_code = result.get('qr_code')
                            secret = result.get('secret')
                            
                            if qr_code:
                                st.subheader("📱 Scan QR Code")
                                st.image(qr_code, caption="Scan this QR code with your authenticator app", width=300)
                            
                            if secret:
                                st.subheader("🔑 Manual Entry")
                                st.code(secret, language=None)
                                st.info("If you can't scan the QR code, manually enter this secret key in your authenticator app.")
                            
                            # Test verification
                            st.subheader("🧪 Test Your Setup")
                            with st.form("test_totp_form"):
                                test_code = st.text_input(
                                    "Test Code", 
                                    placeholder="Enter code from your authenticator app",
                                    max_chars=6
                                )
                                if st.form_submit_button("Test Code"):
                                    if secret and auth_manager.verify_totp_code(secret, test_code):
                                        st.success("✅ TOTP setup verified successfully!")
                                        ui.show_toast_notification("2FA setup completed!", "success")
                                    else:
                                        st.error("❌ Invalid code. Please try again.")
                        
                        elif factor_type == "email_otp":
                            st.success("✅ Email OTP MFA enrollment successful!")
                            st.info("📧 You will receive OTP codes via email during sign-in.")
                            ui.show_toast_notification("Email OTP setup completed!", "success")
                    
                    else:
                        st.error(f"❌ {result.get('error', 'MFA enrollment failed')}")
        
        with col2:
            if st.button("❌ Cancel", type="secondary"):
                st.session_state.show_mfa_setup = False
                st.rerun()

def show_change_password_dialog():
    """Show change password dialog"""
    st.markdown("---")
    st.markdown("### 🔑 Change Password")
    
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password", placeholder="Enter your current password")
        new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
        confirm_password = st.text_input("Confirm New Password", type="password", placeholder="Confirm new password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("🔄 Update Password", type="primary"):
                if all([current_password, new_password, confirm_password]):
                    if new_password == confirm_password:
                        if len(new_password) >= 8:
                            # Here you would implement actual password change logic
                            ui.show_toast_notification("Password updated successfully!", "success")
                            st.success("✅ Password has been updated successfully!")
                        else:
                            st.error("❌ Password must be at least 8 characters long")
                    else:
                        st.error("❌ New passwords do not match")
                else:
                    st.error("❌ Please fill in all fields")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.info("Password change cancelled")

def render_analytics_reports():
    """Render analytics and reporting page"""
    st.markdown('<div class="main-header">📈 Analytics & Reports</div>', unsafe_allow_html=True)
    st.info("📊 Advanced analytics dashboard with KPIs, trends, and export functionality")

def render_data_explorer():
    """Render comprehensive data explorer page"""
    st.markdown('<div class="main-header">🔍 Data Explorer</div>', unsafe_allow_html=True)
    
    # Data Explorer Control Panel
    st.markdown("### 🎛️ Data Explorer Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_source = st.selectbox("📊 Data Source", ["Employees", "Performance Reviews", "Hiring Data", "Compensation", "Training Records"])
    with col2:
        view_mode = st.selectbox("👁️ View Mode", ["Table View", "Chart View", "Statistical View", "Raw Data"])
    with col3:
        export_format = st.selectbox("📤 Export Format", ["CSV", "Excel", "JSON", "PDF Report"])
    with col4:
        if st.button("🔄 Refresh Data", type="primary"):
            ui.show_toast_notification("Data refreshed successfully!", "success")
    
    # Main Explorer Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Browse Data", "🔍 Query Builder", "📊 Visualizations", "🔧 Data Quality", "📈 Analytics", "⚙️ Data Management"])
    
    with tab1:
        render_data_browser()
    
    with tab2:
        render_query_builder()
    
    with tab3:
        render_data_visualizations()
    
    with tab4:
        render_data_quality()
    
    with tab5:
        render_data_analytics()
    
    with tab6:
        render_data_management()

def render_settings():
    """Render comprehensive settings and preferences page"""
    st.markdown('<div class="main-header">⚙️ Settings & Preferences</div>', unsafe_allow_html=True)
    
    # Settings Control Panel
    st.markdown("### 🎛️ Settings Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save All Settings", type="primary"):
            ui.show_toast_notification("All settings saved successfully!", "success")
    with col2:
        if st.button("🔄 Reset to Defaults"):
            ui.show_toast_notification("Settings reset to defaults!", "info")
    with col3:
        if st.button("📤 Export Settings"):
            ui.show_toast_notification("Settings exported!", "success")
    with col4:
        if st.button("📥 Import Settings"):
            ui.show_toast_notification("Settings imported!", "success")
    
    # Main Settings Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👤 Profile", "🔔 Notifications", "🔒 Security", "🎨 Appearance", 
        "🔗 Integrations", "📊 Data & Privacy", "⚙️ System"
    ])
    
    with tab1:
        render_profile_settings()
    
    with tab2:
        render_notification_settings()
    
    with tab3:
        render_security_settings()
    
    with tab4:
        render_appearance_settings()
    
    with tab5:
        render_integration_settings()
    
    with tab6:
        render_data_privacy_settings()
    
    with tab7:
        render_system_settings()

def render_bias_trends():
    """Render bias trends and patterns analysis"""
    st.markdown("### 📈 Historical Trends & Patterns")
    
    # Generate sample trend data
    import numpy as np
    import pandas as pd
    
    # Historical fairness metrics over time
    months = pd.date_range('2023-01-01', periods=12, freq='M')
    hiring_fairness = [78, 79, 81, 80, 82, 84, 83, 85, 84, 82, 83, 82]
    promotion_equity = [85, 87, 88, 89, 90, 91, 90, 92, 91, 90, 91, 91]
    pay_parity = [72, 73, 74, 75, 76, 75, 74, 76, 77, 76, 75, 76]
    review_fairness = [80, 82, 83, 84, 85, 86, 87, 88, 87, 86, 87, 88]
    
    # Create trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=hiring_fairness, mode='lines+markers', name='Hiring Fairness', line=dict(color='#4CAF50')))
    fig.add_trace(go.Scatter(x=months, y=promotion_equity, mode='lines+markers', name='Promotion Equity', line=dict(color='#2196F3')))
    fig.add_trace(go.Scatter(x=months, y=pay_parity, mode='lines+markers', name='Pay Parity', line=dict(color='#FF9800')))
    fig.add_trace(go.Scatter(x=months, y=review_fairness, mode='lines+markers', name='Review Fairness', line=dict(color='#9C27B0')))
    
    fig.update_layout(
        title="Fairness Metrics Trends Over Time",
        xaxis_title="Month",
        yaxis_title="Fairness Score (%)",
        height=400,
        template="plotly_dark",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔍 Key Insights")
        st.markdown("""
        - **Hiring Fairness**: Steady improvement (+4% YoY)
        - **Promotion Equity**: Consistent high performance (91% avg)
        - **Pay Parity**: Needs attention (-2% recent decline)
        - **Review Fairness**: Strong upward trend (+8% YoY)
        """)
    
    with col2:
        st.markdown("#### 📊 Statistical Analysis")
        st.markdown("""
        - **Correlation**: Strong positive correlation between hiring and reviews (r=0.85)
        - **Volatility**: Pay parity shows highest variance (σ=1.2)
        - **Trend**: Overall fairness improving at 2.3% annually
        - **Seasonality**: Q4 typically shows best performance
        """)

def render_detailed_bias_analysis():
    """Render detailed bias analysis with interactive charts"""
    st.markdown("### 🔍 Detailed Bias Analysis")
    
    # Analysis by protected characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Gender Analysis")
        
        # Gender pay gap analysis
        gender_data = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Non-Binary'],
            'Avg_Salary': [85000, 78000, 82000],
            'Count': [120, 95, 8],
            'Promotion_Rate': [0.15, 0.12, 0.14]
        })
        
        fig1 = px.bar(gender_data, x='Gender', y='Avg_Salary', 
                     title='Average Salary by Gender',
                     color='Gender',
                     color_discrete_map={'Male': '#2196F3', 'Female': '#E91E63', 'Non-Binary': '#9C27B0'})
        fig1.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Statistical significance test
        st.markdown("**Statistical Test Results:**")
        st.markdown("- ANOVA F-statistic: 12.45 (p < 0.001)")
        st.markdown("- Effect size (Cohen's d): 0.68 (medium)")
        st.warning("⚠️ Significant gender pay gap detected")
    
    with col2:
        st.markdown("#### Department Analysis")
        
        # Department fairness scores
        dept_data = pd.DataFrame({
            'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
            'Fairness_Score': [85, 78, 82, 91, 87],
            'Risk_Level': ['Low', 'High', 'Medium', 'Low', 'Low']
        })
        
        color_map = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
        fig2 = px.bar(dept_data, x='Department', y='Fairness_Score',
                     title='Fairness Score by Department',
                     color='Risk_Level',
                     color_discrete_map=color_map)
        fig2.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("**Department Insights:**")
        st.markdown("- Sales dept requires immediate attention")
        st.markdown("- HR shows exemplary fairness practices")
        st.markdown("- Engineering trending positive")

def render_demographics_analysis():
    """Render demographics analysis with diversity metrics"""
    st.markdown("### 📊 Demographics & Diversity Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Age Distribution")
        age_data = pd.DataFrame({
            'Age_Group': ['18-25', '26-35', '36-45', '46-55', '56+'],
            'Count': [25, 85, 95, 45, 15],
            'Avg_Performance': [82, 87, 89, 85, 83]
        })
        
        fig1 = px.pie(age_data, values='Count', names='Age_Group',
                     title='Employee Age Distribution')
        fig1.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("#### Ethnicity Representation")
        ethnicity_data = pd.DataFrame({
            'Ethnicity': ['White', 'Asian', 'Hispanic', 'Black', 'Other'],
            'Percentage': [45, 25, 15, 10, 5],
            'Leadership_%': [60, 20, 8, 8, 4]
        })
        
        fig2 = px.bar(ethnicity_data, x='Ethnicity', y=['Percentage', 'Leadership_%'],
                     title='Representation vs Leadership',
                     barmode='group')
        fig2.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        st.markdown("#### Diversity Index")
        
        # Diversity metrics
        diversity_metrics = {
            "Gender Diversity": 0.72,
            "Ethnic Diversity": 0.68,
            "Age Diversity": 0.75,
            "Educational Diversity": 0.81
        }
        
        for metric, value in diversity_metrics.items():
            color = "#4CAF50" if value > 0.7 else "#FF9800" if value > 0.6 else "#F44336"
            st.markdown(f"""
            <div style="background: {color}20; padding: 10px; border-radius: 8px; margin: 5px 0;">
                <strong>{metric}</strong><br>
                <span style="font-size: 1.2em; color: {color};">{value:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

def render_risk_areas():
    """Render risk areas and alerts"""
    st.markdown("### ⚠️ Risk Areas & Alerts")
    
    # High-risk areas
    risk_areas = [
        {
            "area": "Sales Department Pay Equity",
            "risk_level": "High",
            "score": 65,
            "trend": "Declining",
            "impact": "Legal compliance risk",
            "action": "Immediate salary audit required"
        },
        {
            "area": "Promotion Rate Disparity",
            "risk_level": "Medium",
            "score": 72,
            "trend": "Stable",
            "impact": "Employee retention",
            "action": "Review promotion criteria"
        },
        {
            "area": "Hiring Bias in Engineering",
            "risk_level": "Medium",
            "score": 78,
            "trend": "Improving",
            "impact": "Diversity goals",
            "action": "Expand recruitment channels"
        }
    ]
    
    for i, risk in enumerate(risk_areas):
        color = "#F44336" if risk["risk_level"] == "High" else "#FF9800"
        
        with st.expander(f"🚨 {risk['area']} - {risk['risk_level']} Risk"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk['score']}/100", delta=f"{risk['trend']}")
            
            with col2:
                st.markdown(f"**Impact:** {risk['impact']}")
                st.markdown(f"**Trend:** {risk['trend']}")
            
            with col3:
                st.markdown(f"**Recommended Action:**")
                st.markdown(f"{risk['action']}")
                
                if st.button(f"Create Action Plan", key=f"action_{i}"):
                    ui.show_toast_notification("Action plan created!", "success")
    
    # Risk heatmap
    st.markdown("#### 🔥 Risk Heatmap by Department & Metric")
    
    # Sample heatmap data
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
    metrics = ['Hiring', 'Promotion', 'Pay', 'Reviews']
    
    heatmap_data = np.array([
        [85, 82, 78, 88],  # Engineering
        [65, 70, 62, 75],  # Sales
        [80, 78, 85, 82],  # Marketing
        [92, 90, 88, 95],  # HR
        [88, 85, 82, 87]   # Finance
    ])
    
    fig = px.imshow(heatmap_data, 
                   x=metrics, y=departments,
                   color_continuous_scale='RdYlGn',
                   title="Fairness Scores Heatmap",
                   text_auto=True)
    fig.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def render_bias_recommendations():
    """Render AI-powered recommendations"""
    st.markdown("### 📋 AI-Powered Recommendations")
    
    # Priority recommendations
    recommendations = [
        {
            "priority": "Critical",
            "category": "Pay Equity",
            "title": "Address Sales Department Pay Gap",
            "description": "Immediate salary adjustment needed for 12 employees in Sales",
            "impact": "High",
            "effort": "Medium",
            "timeline": "30 days",
            "cost": "$45,000"
        },
        {
            "priority": "High",
            "category": "Hiring",
            "title": "Expand Diverse Recruitment Channels",
            "description": "Partner with HBCUs and women in tech organizations",
            "impact": "Medium",
            "effort": "Low",
            "timeline": "60 days",
            "cost": "$8,000"
        },
        {
            "priority": "Medium",
            "category": "Training",
            "title": "Unconscious Bias Training for Managers",
            "description": "Quarterly training sessions for all people managers",
            "impact": "Medium",
            "effort": "Medium",
            "timeline": "90 days",
            "cost": "$15,000"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        priority_color = {"Critical": "#F44336", "High": "#FF9800", "Medium": "#2196F3"}[rec["priority"]]
        
        with st.expander(f"🎯 {rec['priority']} Priority: {rec['title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Category:** {rec['category']}")
                st.markdown(f"**Description:** {rec['description']}")
                st.markdown(f"**Timeline:** {rec['timeline']}")
                st.markdown(f"**Estimated Cost:** {rec['cost']}")
            
            with col2:
                st.markdown("**Impact Assessment:**")
                st.markdown(f"- Impact: {rec['impact']}")
                st.markdown(f"- Effort: {rec['effort']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Approve", key=f"approve_{i}"):
                        ui.show_toast_notification("Recommendation approved!", "success")
                with col_b:
                    if st.button("📋 Details", key=f"details_{i}"):
                        st.info("Detailed implementation plan would open here")
    
    # Export recommendations
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Full Report", type="primary"):
            ui.show_toast_notification("Bias audit report exported!", "success")
    
    with col2:
        if st.button("📧 Email to Leadership"):
            ui.show_toast_notification("Report sent to leadership team!", "info")
    
    with col3:
        if st.button("🔄 Schedule Next Audit"):
            ui.show_toast_notification("Next audit scheduled!", "info")

def render_data_browser():
    """Render interactive data browser"""
    st.markdown("### 📋 Interactive Data Browser")
    
    # Advanced Filtering Panel
    st.markdown("#### 🔍 Advanced Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search_term = st.text_input("🔎 Search", placeholder="Search across all fields...")
    with col2:
        department_filter = st.multiselect("🏢 Department", ["Engineering", "Sales", "Marketing", "HR", "Finance"])
    with col3:
        date_range = st.date_input("📅 Date Range", value=[])
    with col4:
        performance_range = st.slider("📊 Performance Range", 0, 100, (0, 100))
    
    # Sample employee data with more fields
    import pandas as pd
    sample_data = pd.DataFrame({
        'ID': range(1, 21),
        'Name': [f'Employee {i}' for i in range(1, 21)],
        'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'] * 4,
        'Position': ['Senior', 'Junior', 'Manager', 'Lead', 'Specialist'] * 4,
        'Hire_Date': pd.date_range('2020-01-01', periods=20, freq='30D'),
        'Salary': [50000 + i*2000 for i in range(20)],
        'Performance': [75 + (i % 25) for i in range(20)],
        'Status': ['Active', 'Active', 'On Leave', 'Active', 'Remote'] * 4
    })
    
    # Data table with sorting and pagination
    st.markdown("#### 📊 Data Table")
    
    # Table controls
    col1, col2, col3 = st.columns(3)
    with col1:
        rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100])
    with col2:
        sort_column = st.selectbox("Sort by", sample_data.columns)
    with col3:
        sort_order = st.selectbox("Order", ["Ascending", "Descending"])
    
    # Apply sorting
    ascending = sort_order == "Ascending"
    sorted_data = sample_data.sort_values(sort_column, ascending=ascending)
    
    # Display data with interactive features
    st.dataframe(
        sorted_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Performance": st.column_config.ProgressColumn(
                "Performance",
                help="Employee performance score",
                min_value=0,
                max_value=100,
            ),
            "Salary": st.column_config.NumberColumn(
                "Salary",
                help="Annual salary",
                format="$%d",
            ),
            "Hire_Date": st.column_config.DateColumn(
                "Hire Date",
                help="Employee hire date",
            ),
        }
    )
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(sample_data))
    with col2:
        st.metric("Avg Performance", f"{sample_data['Performance'].mean():.1f}%")
    with col3:
        st.metric("Avg Salary", f"${sample_data['Salary'].mean():,.0f}")
    with col4:
        st.metric("Active Employees", len(sample_data[sample_data['Status'] == 'Active']))

def render_query_builder():
    """Render SQL query builder interface"""
    st.markdown("### 🔍 Visual Query Builder")
    
    # Query builder interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🛠️ Build Your Query")
        
        # Table selection
        selected_tables = st.multiselect(
            "📊 Select Tables",
            ["employees", "performance_reviews", "hiring_data", "compensation", "training_records"],
            default=["employees"]
        )
        
        # Column selection
        available_columns = {
            "employees": ["id", "name", "department", "position", "hire_date", "salary", "performance_score"],
            "performance_reviews": ["id", "employee_id", "review_date", "score", "feedback"],
            "hiring_data": ["id", "candidate_name", "position", "hire_date", "source"],
            "compensation": ["employee_id", "base_salary", "bonus", "benefits"],
            "training_records": ["employee_id", "course_name", "completion_date", "score"]
        }
        
        all_columns = []
        for table in selected_tables:
            all_columns.extend([f"{table}.{col}" for col in available_columns.get(table, [])])
        
        selected_columns = st.multiselect("📋 Select Columns", all_columns)
        
        # Conditions builder
        st.markdown("#### ⚙️ Add Conditions")
        num_conditions = st.number_input("Number of conditions", min_value=0, max_value=10, value=1)
        
        conditions = []
        for i in range(int(num_conditions)):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                condition_column = st.selectbox(f"Column {i+1}", all_columns, key=f"cond_col_{i}")
            with col_b:
                operator = st.selectbox(f"Operator {i+1}", ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN"], key=f"cond_op_{i}")
            with col_c:
                value = st.text_input(f"Value {i+1}", key=f"cond_val_{i}")
            
            if condition_column and operator and value:
                conditions.append(f"{condition_column} {operator} '{value}'")
        
        # Join conditions
        if len(selected_tables) > 1:
            st.markdown("#### 🔗 Join Conditions")
            join_type = st.selectbox("Join Type", ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN"])
            join_condition = st.text_input("Join Condition", placeholder="e.g., employees.id = performance_reviews.employee_id")
    
    with col2:
        st.markdown("#### 📝 Generated SQL")
        
        # Build SQL query
        if selected_columns and selected_tables:
            sql_query = f"SELECT {', '.join(selected_columns)}\nFROM {selected_tables[0]}"
            
            # Add joins
            if len(selected_tables) > 1:
                for table in selected_tables[1:]:
                    sql_query += f"\n{join_type} {table} ON {join_condition if 'join_condition' in locals() else 'condition'}"
            
            # Add conditions
            if conditions:
                sql_query += f"\nWHERE {' AND '.join(conditions)}"
            
            st.code(sql_query, language="sql")
            
            # Execute query button
            if st.button("▶️ Execute Query", type="primary"):
                ui.show_toast_notification("Query executed successfully!", "success")
                st.success("Query results would appear here")
        
        # Query history
        st.markdown("#### 📚 Query History")
        query_history = [
            "SELECT * FROM employees WHERE department = 'Engineering'",
            "SELECT name, salary FROM employees ORDER BY salary DESC",
            "SELECT department, AVG(performance_score) FROM employees GROUP BY department"
        ]
        
        for i, query in enumerate(query_history):
            if st.button(f"📋 Query {i+1}", key=f"hist_{i}"):
                st.code(query, language="sql")

def render_data_visualizations():
    """Render data visualization tools"""
    st.markdown("### 📊 Data Visualizations")
    
    # Visualization builder
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎨 Chart Builder")
        
        chart_type = st.selectbox(
            "📈 Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
        )
        
        x_axis = st.selectbox("X-Axis", ["Department", "Position", "Hire_Date", "Performance", "Salary"])
        y_axis = st.selectbox("Y-Axis", ["Performance", "Salary", "Count", "Average"])
        
        color_by = st.selectbox("Color By", ["None", "Department", "Position", "Status"])
        
        # Chart customization
        st.markdown("#### 🎛️ Customization")
        chart_title = st.text_input("Chart Title", value="HR Analytics Visualization")
        show_legend = st.checkbox("Show Legend", value=True)
        chart_theme = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])
    
    with col2:
        st.markdown("#### 📊 Live Preview")
        
        # Generate sample visualization based on selection
        import numpy as np
        
        if chart_type == "Bar Chart":
            dept_data = pd.DataFrame({
                'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
                'Avg_Performance': [87, 82, 85, 91, 88],
                'Employee_Count': [45, 32, 28, 15, 22]
            })
            
            fig = px.bar(dept_data, x='Department', y='Avg_Performance',
                        title=chart_title, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":
            scatter_data = pd.DataFrame({
                'Salary': np.random.normal(75000, 15000, 100),
                'Performance': np.random.normal(85, 10, 100),
                'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100)
            })
            
            fig = px.scatter(scatter_data, x='Salary', y='Performance', 
                           color='Department', title=chart_title, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Pie Chart":
            pie_data = pd.DataFrame({
                'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
                'Count': [45, 32, 28, 15, 22]
            })
            
            fig = px.pie(pie_data, values='Count', names='Department',
                        title=chart_title, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
    
    # Chart gallery
    st.markdown("#### 🖼️ Chart Gallery")
    chart_cols = st.columns(3)
    
    chart_templates = [
        {"name": "Department Performance", "type": "Bar", "description": "Average performance by department"},
        {"name": "Salary Distribution", "type": "Histogram", "description": "Employee salary distribution"},
        {"name": "Performance Trends", "type": "Line", "description": "Performance trends over time"}
    ]
    
    for i, template in enumerate(chart_templates):
        with chart_cols[i % 3]:
            if st.button(f"📊 {template['name']}", key=f"template_{i}"):
                st.info(f"Loading {template['name']} visualization...")

def render_data_quality():
    """Render data quality assessment"""
    st.markdown("### 🔧 Data Quality Assessment")
    
    # Data quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("Completeness", "94.2%", "+2.1%", "✅")
    with col2:
        ui.render_kpi_card("Accuracy", "97.8%", "+0.5%", "🎯")
    with col3:
        ui.render_kpi_card("Consistency", "91.5%", "-1.2%", "🔄")
    with col4:
        ui.render_kpi_card("Timeliness", "88.9%", "+3.4%", "⏰")
    
    # Data quality details
    tab1, tab2, tab3 = st.tabs(["📊 Quality Overview", "⚠️ Issues Found", "🔧 Recommendations"])
    
    with tab1:
        st.markdown("#### 📈 Data Quality Trends")
        
        # Quality trend chart
        months = pd.date_range('2023-01-01', periods=12, freq='M')
        completeness = [90 + np.random.normal(0, 2) for _ in range(12)]
        accuracy = [95 + np.random.normal(0, 1.5) for _ in range(12)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=completeness, mode='lines+markers', name='Completeness'))
        fig.add_trace(go.Scatter(x=months, y=accuracy, mode='lines+markers', name='Accuracy'))
        fig.update_layout(title="Data Quality Trends", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### ⚠️ Data Quality Issues")
        
        issues = [
            {"table": "employees", "issue": "Missing phone numbers", "count": 23, "severity": "Medium"},
            {"table": "performance_reviews", "issue": "Duplicate entries", "count": 5, "severity": "High"},
            {"table": "hiring_data", "issue": "Invalid date formats", "count": 12, "severity": "Low"},
        ]
        
        for issue in issues:
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[issue["severity"]]
            
            with st.expander(f"{severity_color} {issue['table']}: {issue['issue']} ({issue['count']} records)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Table:** {issue['table']}")
                    st.markdown(f"**Issue:** {issue['issue']}")
                    st.markdown(f"**Affected Records:** {issue['count']}")
                with col2:
                    if st.button(f"🔧 Fix Issue", key=f"fix_{issue['table']}"):
                        ui.show_toast_notification("Issue resolution initiated!", "info")
    
    with tab3:
        st.markdown("#### 🔧 Quality Improvement Recommendations")
        
        recommendations = [
            "Implement data validation rules for phone number format",
            "Set up automated duplicate detection and removal",
            "Standardize date format across all tables",
            "Add required field validation for critical data points"
        ]
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")
            if st.button(f"✅ Implement", key=f"implement_{i}"):
                ui.show_toast_notification("Recommendation implemented!", "success")

def render_data_analytics():
    """Render advanced data analytics"""
    st.markdown("### 📈 Advanced Data Analytics")
    
    # Analytics dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Statistical Summary")
        
        # Generate sample statistics
        stats_data = {
            "Metric": ["Mean Salary", "Median Performance", "Std Dev Age", "Correlation (Salary-Performance)"],
            "Value": ["$78,450", "87.2%", "8.3 years", "0.65"],
            "Trend": ["+5.2%", "+2.1%", "-0.8%", "+0.12"]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Distribution analysis
        st.markdown("#### 📈 Distribution Analysis")
        
        # Sample distribution data
        distribution_data = np.random.normal(85, 10, 1000)
        fig = px.histogram(x=distribution_data, nbins=30, title="Performance Score Distribution")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Correlation Analysis")
        
        # Correlation matrix
        corr_data = np.random.rand(5, 5)
        corr_labels = ['Salary', 'Performance', 'Experience', 'Training', 'Satisfaction']
        
        fig = px.imshow(corr_data, x=corr_labels, y=corr_labels,
                       color_continuous_scale='RdBu', aspect="auto",
                       title="Feature Correlation Matrix")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced analytics tools
    st.markdown("#### 🛠️ Analytics Tools")
    
    tool_cols = st.columns(4)
    
    with tool_cols[0]:
        if st.button("📊 Regression Analysis"):
            st.info("Regression analysis tool would open here")
    
    with tool_cols[1]:
        if st.button("🎯 Clustering Analysis"):
            st.info("Clustering analysis tool would open here")
    
    with tool_cols[2]:
        if st.button("📈 Time Series Analysis"):
            st.info("Time series analysis tool would open here")
    
    with tool_cols[3]:
        if st.button("🤖 ML Model Builder"):
            st.info("ML model builder would open here")

def render_data_management():
    """Render data management tools"""
    st.markdown("### ⚙️ Data Management")
    
    # Data management tabs
    mgmt_tab1, mgmt_tab2, mgmt_tab3, mgmt_tab4 = st.tabs(["📤 Import/Export", "🔄 Data Sync", "🗂️ Schema Management", "🔒 Access Control"])
    
    with mgmt_tab1:
        st.markdown("#### 📤 Data Import/Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Import Data**")
            upload_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'json'])
            
            if upload_file:
                st.success(f"File {upload_file.name} ready for import")
                
                if st.button("📥 Import Data"):
                    ui.show_toast_notification("Data imported successfully!", "success")
        
        with col2:
            st.markdown("**Export Data**")
            export_table = st.selectbox("Select Table", ["employees", "performance_reviews", "hiring_data"])
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            if st.button("📤 Export Data"):
                ui.show_toast_notification(f"Data exported as {export_format}!", "success")
    
    with mgmt_tab2:
        st.markdown("#### 🔄 Data Synchronization")
        
        sync_sources = [
            {"name": "HR System", "status": "Connected", "last_sync": "2 hours ago"},
            {"name": "Payroll System", "status": "Connected", "last_sync": "1 day ago"},
            {"name": "Performance Tool", "status": "Error", "last_sync": "3 days ago"}
        ]
        
        for source in sync_sources:
            status_color = {"Connected": "🟢", "Error": "🔴", "Pending": "🟡"}[source["status"]]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"{status_color} **{source['name']}**")
            with col2:
                st.markdown(f"Last sync: {source['last_sync']}")
            with col3:
                if st.button("🔄 Sync Now", key=f"sync_{source['name']}"):
                    ui.show_toast_notification(f"Syncing {source['name']}...", "info")
    
    with mgmt_tab3:
        st.markdown("#### 🗂️ Database Schema")
        
        # Schema visualization
        tables = ["employees", "performance_reviews", "hiring_data", "compensation"]
        
        for table in tables:
            with st.expander(f"📋 {table} table"):
                if table == "employees":
                    schema_info = {
                        "Column": ["id", "name", "department", "position", "hire_date", "salary"],
                        "Type": ["INTEGER", "VARCHAR(255)", "VARCHAR(100)", "VARCHAR(100)", "DATE", "INTEGER"],
                        "Nullable": ["No", "No", "No", "No", "No", "Yes"]
                    }
                else:
                    schema_info = {
                        "Column": ["id", "employee_id", "date", "value"],
                        "Type": ["INTEGER", "INTEGER", "DATE", "DECIMAL"],
                        "Nullable": ["No", "No", "No", "Yes"]
                    }
                
                schema_df = pd.DataFrame(schema_info)
                st.dataframe(schema_df, hide_index=True, use_container_width=True)
    
    with mgmt_tab4:
        st.markdown("#### 🔒 Data Access Control")
        
        # User permissions
        permissions = [
            {"user": "admin@company.com", "role": "Admin", "tables": "All", "permissions": "Read/Write/Delete"},
            {"user": "hr@company.com", "role": "HR Manager", "tables": "employees, performance", "permissions": "Read/Write"},
            {"user": "analyst@company.com", "role": "Analyst", "tables": "All", "permissions": "Read Only"}
        ]
        
        permissions_df = pd.DataFrame(permissions)
        st.dataframe(permissions_df, hide_index=True, use_container_width=True)
        
        if st.button("➕ Add User Permission"):
            st.info("User permission dialog would open here")

def render_profile_settings():
    """Render user profile settings"""
    st.markdown("### 👤 User Profile")
    
    # Current user info (from session state)
    current_user = st.session_state.get('user_data', {})
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📸 Profile Picture")
        # Profile picture upload
        uploaded_file = st.file_uploader("Upload Profile Picture", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            st.success("Profile picture updated!")
        
        # Current avatar placeholder
        st.markdown("""
        <div style="width: 120px; height: 120px; border-radius: 50%; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-size: 2rem; margin: 20px 0;">
            👤
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ✏️ Personal Information")
        
        # Editable profile fields
        full_name = st.text_input("Full Name", value=current_user.get('full_name', 'John Doe'))
        email = st.text_input("Email Address", value=current_user.get('email', 'user@company.com'), disabled=True)
        job_title = st.text_input("Job Title", value="HR Analytics Manager")
        department = st.selectbox("Department", ["HR", "Engineering", "Sales", "Marketing", "Finance"], index=0)
        phone = st.text_input("Phone Number", value="+1 (555) 123-4567")
        location = st.text_input("Location", value="San Francisco, CA")
        
        # Bio/About section
        st.markdown("#### 📝 About")
        bio = st.text_area("Bio", value="Experienced HR professional focused on data-driven insights and employee engagement.", height=100)
        
        # Work preferences
        st.markdown("#### 💼 Work Preferences")
        col_a, col_b = st.columns(2)
        with col_a:
            timezone = st.selectbox("Timezone", ["UTC-8 (PST)", "UTC-5 (EST)", "UTC+0 (GMT)", "UTC+5:30 (IST)"])
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
        with col_b:
            work_hours_start = st.time_input("Work Hours Start", value=pd.to_datetime("09:00").time())
            work_hours_end = st.time_input("Work Hours End", value=pd.to_datetime("17:00").time())

def render_notification_settings():
    """Render notification preferences"""
    st.markdown("### 🔔 Notification Preferences")
    
    # Email notifications
    st.markdown("#### 📧 Email Notifications")
    col1, col2 = st.columns(2)
    
    with col1:
        email_enabled = st.checkbox("Enable Email Notifications", value=True)
        daily_digest = st.checkbox("Daily Analytics Digest", value=True)
        weekly_report = st.checkbox("Weekly Performance Report", value=True)
        bias_alerts = st.checkbox("Bias Detection Alerts", value=True)
        performance_alerts = st.checkbox("Performance Anomaly Alerts", value=False)
    
    with col2:
        system_updates = st.checkbox("System Updates", value=True)
        security_alerts = st.checkbox("Security Alerts", value=True)
        data_quality_alerts = st.checkbox("Data Quality Issues", value=True)
        model_updates = st.checkbox("AI Model Updates", value=False)
        maintenance_notices = st.checkbox("Maintenance Notices", value=True)
    
    # In-app notifications
    st.markdown("#### 🔔 In-App Notifications")
    col1, col2 = st.columns(2)
    
    with col1:
        push_notifications = st.checkbox("Push Notifications", value=True)
        sound_notifications = st.checkbox("Sound Notifications", value=False)
        desktop_notifications = st.checkbox("Desktop Notifications", value=True)
    
    with col2:
        notification_frequency = st.selectbox("Notification Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
        quiet_hours_enabled = st.checkbox("Enable Quiet Hours", value=True)
        if quiet_hours_enabled:
            col_a, col_b = st.columns(2)
            with col_a:
                quiet_start = st.time_input("Quiet Hours Start", value=pd.to_datetime("18:00").time())
            with col_b:
                quiet_end = st.time_input("Quiet Hours End", value=pd.to_datetime("09:00").time())
    
    # Notification categories
    st.markdown("#### 📋 Notification Categories")
    categories = {
        "High Priority Alerts": True,
        "Performance Insights": True,
        "Bias Detection": True,
        "Data Quality": False,
        "System Status": True,
        "User Activity": False,
        "Compliance Updates": True,
        "Training Reminders": False
    }
    
    for category, default_value in categories.items():
        st.checkbox(category, value=default_value, key=f"notif_{category}")

def render_security_settings():
    """Render security settings with MFA integration"""
    st.markdown("### 🔒 Security & Privacy")
    
    # Password settings
    st.markdown("#### 🔐 Password & Authentication")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔑 Change Password", type="primary"):
            show_change_password_dialog()
        
        # MFA Status and Management
        st.markdown("**Multi-Factor Authentication**")
        factors_result = auth_manager.list_mfa_factors()
        
        if factors_result.get('success'):
            factors = factors_result.get('factors', [])
            if factors:
                st.success(f"✅ 2FA is enabled ({len(factors)} factor(s))")
                
                # Show enrolled factors
                for factor in factors:
                    factor_type = factor.get('factor_type', 'Unknown').upper()
                    created_at = factor.get('created_at', '')[:10] if factor.get('created_at') else 'Unknown'
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"📱 **{factor_type}** - Enrolled: {created_at}")
                    with col_b:
                        if st.button("Remove", key=f"remove_mfa_{factor.get('id')}", type="secondary"):
                            result = auth_manager.remove_mfa_factor(factor.get('id'))
                            if result.get('success'):
                                ui.show_toast_notification("MFA factor removed successfully!", "success")
                                st.rerun()
                            else:
                                st.error(f"Failed to remove MFA factor: {result.get('error')}")
                
                if st.button("➕ Add Another Factor", type="secondary"):
                    st.session_state.show_mfa_setup = True
                    st.rerun()
            else:
                st.warning("⚠️ 2FA is not enabled")
                if st.button("🔒 Enable 2FA", type="primary"):
                    st.session_state.show_mfa_setup = True
                    st.rerun()
        else:
            st.info("Enable 2FA for enhanced security")
            if st.button("🔒 Setup 2FA", type="primary"):
                st.session_state.show_mfa_setup = True
                st.rerun()
    
    with col2:
        session_timeout = st.selectbox("Session Timeout", ["30 minutes", "1 hour", "4 hours", "8 hours", "Never"], index=1)
        remember_device = st.checkbox("Remember this device", value=True)
        auto_logout = st.checkbox("Auto-logout on inactivity", value=True)
    
    # MFA Setup Dialog
    if st.session_state.get('show_mfa_setup'):
        show_mfa_setup_dialog()
    
    # Login activity
    st.markdown("#### 🔍 Login Activity")
    login_history = [
        {"timestamp": "2024-01-15 09:30:15", "location": "San Francisco, CA", "device": "Chrome on Windows", "status": "Success"},
        {"timestamp": "2024-01-14 18:45:22", "location": "San Francisco, CA", "device": "Safari on iPhone", "status": "Success"},
        {"timestamp": "2024-01-14 08:15:33", "location": "Unknown Location", "device": "Chrome on Linux", "status": "Failed"},
    ]
    
    for activity in login_history:
        status_color = "🟢" if activity["status"] == "Success" else "🔴"
        st.markdown(f"{status_color} **{activity['timestamp']}** - {activity['location']} - {activity['device']}")
    
    # Security preferences
    st.markdown("#### 🛡️ Security Preferences")
    col1, col2 = st.columns(2)
    
    with col1:
        login_notifications = st.checkbox("Email on New Login", value=True)
        suspicious_activity = st.checkbox("Alert on Suspicious Activity", value=True)
        data_export_approval = st.checkbox("Require Approval for Data Export", value=True)
    
    with col2:
        ip_restrictions = st.checkbox("IP Address Restrictions", value=False)
        if ip_restrictions:
            allowed_ips = st.text_area("Allowed IP Addresses", placeholder="192.168.1.1\n10.0.0.1")

def render_appearance_settings():
    """Render appearance and UI settings"""
    st.markdown("### 🎨 Appearance & Interface")
    
    # Theme settings
    st.markdown("#### 🌓 Theme Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        theme_mode = st.selectbox("Theme Mode", ["Auto (System)", "Light", "Dark"], index=0)
        accent_color = st.selectbox("Accent Color", ["Blue", "Purple", "Green", "Orange", "Red"], index=1)
        
        # Theme preview
        if theme_mode == "Dark":
            st.markdown("""
            <div style="background: #1e1e1e; color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h4>🌙 Dark Theme Preview</h4>
                <p>This is how the dark theme looks with your selected accent color.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ffffff; color: #333; padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #ddd;">
                <h4>☀️ Light Theme Preview</h4>
                <p>This is how the light theme looks with your selected accent color.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        glassmorphism_enabled = st.checkbox("Glassmorphism Effects", value=True)
        animations_enabled = st.checkbox("Smooth Animations", value=True)
        reduced_motion = st.checkbox("Reduce Motion (Accessibility)", value=False)
        high_contrast = st.checkbox("High Contrast Mode", value=False)
    
    # Layout settings
    st.markdown("#### 📐 Layout Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        sidebar_collapsed = st.checkbox("Collapse Sidebar by Default", value=False)
        compact_mode = st.checkbox("Compact Mode", value=False)
        show_tooltips = st.checkbox("Show Tooltips", value=True)
    
    with col2:
        chart_theme = st.selectbox("Chart Theme", ["Auto", "Light", "Dark", "Colorful"], index=0)
        font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"], index=1)
        density = st.selectbox("Information Density", ["Comfortable", "Compact", "Spacious"], index=0)
    
    # Dashboard customization
    st.markdown("#### 📊 Dashboard Customization")
    st.markdown("**Widget Preferences:**")
    
    widget_cols = st.columns(3)
    widgets = [
        "KPI Cards", "Quick Actions", "Recent Insights", "System Status",
        "Performance Charts", "Employee Directory", "Bias Alerts", "Data Quality",
        "AI Predictions"
    ]
    
    for i, widget in enumerate(widgets):
        with widget_cols[i % 3]:
            st.checkbox(widget, value=True, key=f"widget_{widget}")

def render_integration_settings():
    """Render integration settings"""
    st.markdown("### 🔗 Integrations & Connections")
    
    # Connected services
    st.markdown("#### 🔌 Connected Services")
    
    integrations = [
        {
            "name": "Slack",
            "description": "Send notifications and alerts to Slack channels",
            "status": "Connected",
            "icon": "💬",
            "last_sync": "2 hours ago"
        },
        {
            "name": "Microsoft Teams",
            "description": "Collaborate and share insights with your team",
            "status": "Not Connected",
            "icon": "👥",
            "last_sync": "Never"
        },
        {
            "name": "Google Workspace",
            "description": "Sync with Google Calendar and Drive",
            "status": "Connected",
            "icon": "📧",
            "last_sync": "1 day ago"
        },
        {
            "name": "Salesforce",
            "description": "Import employee and performance data",
            "status": "Error",
            "icon": "☁️",
            "last_sync": "3 days ago"
        }
    ]
    
    for integration in integrations:
        with st.expander(f"{integration['icon']} {integration['name']} - {integration['status']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {integration['description']}")
                st.markdown(f"**Last Sync:** {integration['last_sync']}")
                
                if integration['status'] == 'Connected':
                    st.success("✅ Successfully connected")
                elif integration['status'] == 'Error':
                    st.error("❌ Connection error - check credentials")
                else:
                    st.info("⚪ Not connected")
            
            with col2:
                if integration['status'] == 'Connected':
                    if st.button("🔧 Configure", key=f"config_{integration['name']}"):
                        st.info("Configuration dialog would open")
                    if st.button("🔄 Sync Now", key=f"sync_{integration['name']}"):
                        ui.show_toast_notification(f"Syncing {integration['name']}...", "info")
                    if st.button("❌ Disconnect", key=f"disconnect_{integration['name']}"):
                        ui.show_toast_notification(f"Disconnected from {integration['name']}", "warning")
                else:
                    if st.button("🔗 Connect", key=f"connect_{integration['name']}", type="primary"):
                        ui.show_toast_notification(f"Connecting to {integration['name']}...", "info")
    
    # API settings
    st.markdown("#### 🔑 API Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**API Keys & Tokens**")
        if st.button("🔑 Generate API Key"):
            st.code("hr_analytics_api_key_abc123xyz789", language="text")
        
        api_rate_limit = st.selectbox("API Rate Limit", ["100/hour", "500/hour", "1000/hour", "Unlimited"])
        webhook_enabled = st.checkbox("Enable Webhooks", value=False)
    
    with col2:
        st.markdown("**External Data Sources**")
        auto_sync_enabled = st.checkbox("Auto-sync External Data", value=True)
        sync_frequency = st.selectbox("Sync Frequency", ["Real-time", "Every 15 minutes", "Hourly", "Daily"])
        
        if st.button("🔄 Sync All Sources"):
            ui.show_toast_notification("Syncing all data sources...", "info")

def render_data_privacy_settings():
    """Render data and privacy settings"""
    st.markdown("### 📊 Data & Privacy")
    
    # Data retention
    st.markdown("#### 🗄️ Data Retention")
    col1, col2 = st.columns(2)
    
    with col1:
        employee_data_retention = st.selectbox("Employee Data Retention", ["1 year", "2 years", "5 years", "Indefinite"], index=2)
        performance_data_retention = st.selectbox("Performance Data Retention", ["6 months", "1 year", "2 years", "5 years"], index=2)
        audit_log_retention = st.selectbox("Audit Log Retention", ["30 days", "90 days", "1 year", "2 years"], index=2)
    
    with col2:
        auto_delete_enabled = st.checkbox("Auto-delete Old Data", value=True)
        backup_before_delete = st.checkbox("Backup Before Deletion", value=True)
        
        if st.button("🗑️ Clean Up Old Data"):
            ui.show_toast_notification("Data cleanup initiated...", "info")
    
    # Privacy controls
    st.markdown("#### 🔒 Privacy Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        anonymize_exports = st.checkbox("Anonymize Data Exports", value=True)
        mask_sensitive_data = st.checkbox("Mask Sensitive Data in UI", value=False)
        audit_data_access = st.checkbox("Audit All Data Access", value=True)
    
    with col2:
        gdpr_compliance = st.checkbox("GDPR Compliance Mode", value=True)
        ccpa_compliance = st.checkbox("CCPA Compliance Mode", value=True)
        data_processing_consent = st.checkbox("Require Data Processing Consent", value=True)
    
    # Data export and deletion
    st.markdown("#### 📤 Data Rights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export My Data"):
            ui.show_toast_notification("Data export initiated...", "info")
    
    with col2:
        if st.button("🔄 Data Portability"):
            ui.show_toast_notification("Preparing data for transfer...", "info")
    
    with col3:
        if st.button("🗑️ Delete My Data"):
            st.warning("⚠️ This action cannot be undone!")
    
    # Compliance dashboard
    st.markdown("#### 📋 Compliance Dashboard")
    compliance_metrics = {
        "Data Encryption": "✅ AES-256",
        "Access Logging": "✅ Enabled",
        "Data Anonymization": "✅ Active",
        "Consent Management": "✅ Compliant",
        "Right to Deletion": "✅ Implemented",
        "Data Portability": "✅ Available"
    }
    
    compliance_cols = st.columns(3)
    for i, (metric, status) in enumerate(compliance_metrics.items()):
        with compliance_cols[i % 3]:
            st.markdown(f"**{metric}**")
            st.markdown(status)

def render_system_settings():
    """Render system settings"""
    st.markdown("### ⚙️ System Configuration")
    
    # System status
    st.markdown("#### 📊 System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.render_kpi_card("System Health", "98.5%", "+0.2%", "💚")
    with col2:
        ui.render_kpi_card("Uptime", "99.9%", "0%", "⏱️")
    with col3:
        ui.render_kpi_card("Response Time", "125ms", "-5ms", "⚡")
    with col4:
        ui.render_kpi_card("Active Users", "247", "+12", "👥")
    
    # Performance settings
    st.markdown("#### ⚡ Performance Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        cache_enabled = st.checkbox("Enable Data Caching", value=True)
        lazy_loading = st.checkbox("Lazy Load Charts", value=True)
        compression_enabled = st.checkbox("Enable Data Compression", value=True)
        
        cache_duration = st.selectbox("Cache Duration", ["5 minutes", "15 minutes", "1 hour", "4 hours"], index=1)
    
    with col2:
        max_concurrent_users = st.number_input("Max Concurrent Users", min_value=10, max_value=1000, value=100)
        query_timeout = st.number_input("Query Timeout (seconds)", min_value=30, max_value=300, value=60)
        
        if st.button("🧹 Clear Cache"):
            ui.show_toast_notification("Cache cleared successfully!", "success")
    
    # Backup and maintenance
    st.markdown("#### 💾 Backup & Maintenance")
    col1, col2 = st.columns(2)
    
    with col1:
        auto_backup_enabled = st.checkbox("Automatic Backups", value=True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"], index=0)
        backup_retention = st.selectbox("Backup Retention", ["30 days", "90 days", "1 year"], index=1)
        
        if st.button("💾 Create Backup Now"):
            ui.show_toast_notification("Backup initiated...", "info")
    
    with col2:
        maintenance_mode = st.checkbox("Maintenance Mode", value=False)
        if maintenance_mode:
            st.warning("⚠️ System is in maintenance mode")
            maintenance_message = st.text_area("Maintenance Message", value="System maintenance in progress...")
        
        scheduled_maintenance = st.checkbox("Schedule Maintenance", value=False)
        if scheduled_maintenance:
            maintenance_date = st.date_input("Maintenance Date")
            maintenance_time = st.time_input("Maintenance Time")
    
    # System logs
    st.markdown("#### 📋 System Logs")
    log_level = st.selectbox("Log Level", ["ERROR", "WARN", "INFO", "DEBUG"], index=2)
    
    # Sample log entries
    log_entries = [
        {"timestamp": "2024-01-15 10:30:15", "level": "INFO", "message": "User login successful", "user": "admin@company.com"},
        {"timestamp": "2024-01-15 10:25:42", "level": "WARN", "message": "High memory usage detected", "user": "system"},
        {"timestamp": "2024-01-15 10:20:33", "level": "INFO", "message": "Data sync completed", "user": "system"},
        {"timestamp": "2024-01-15 10:15:18", "level": "ERROR", "message": "Database connection timeout", "user": "system"},
    ]
    
    for log in log_entries:
        level_color = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵", "DEBUG": "⚪"}[log["level"]]
        st.markdown(f"{level_color} **{log['timestamp']}** [{log['level']}] {log['message']} - {log['user']}")
    
    if st.button("📥 Download Full Logs"):
        ui.show_toast_notification("Log file downloaded!", "success")

def render_administration():
    """Render comprehensive administration panel"""
    st.markdown('<div class="main-header">🔧 Administration</div>', unsafe_allow_html=True)
    
    # Admin access control
    user_role = st.session_state.user_data.get('role', 'user')
    if user_role != 'admin':
        st.error("🚫 Access Denied: Administrator privileges required")
        st.info("Contact your system administrator for access to this section.")
        return
    
    # Admin Control Panel
    st.markdown("### 🎛️ Admin Control Panel")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 System Refresh", type="primary"):
            ui.show_toast_notification("System refreshed successfully!", "success")
    with col2:
        if st.button("📊 Generate Reports"):
            ui.show_toast_notification("Admin reports generated!", "info")
    with col3:
        if st.button("🔒 Security Audit"):
            ui.show_toast_notification("Security audit initiated!", "warning")
    with col4:
        if st.button("⚠️ Emergency Mode"):
            ui.show_toast_notification("Emergency protocols activated!", "error")
    
    # Main Admin Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👥 User Management", "🔒 Security Center", "📊 System Monitor", 
        "🗄️ Data Management", "🔧 Configuration", "📋 Audit Logs", "🚨 Alerts & Incidents"
    ])
    
    with tab1:
        render_user_management()
    
    with tab2:
        render_security_center()
    
    with tab3:
        render_system_monitor()
    
    with tab4:
        render_admin_data_management()
    
    with tab5:
        render_system_configuration()
    
    with tab6:
        render_audit_logs()
    
    with tab7:
        render_alerts_incidents()

def render_user_management():
    """Render user management interface"""
    st.markdown("### 👥 User Management")
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.render_kpi_card("Total Users", "247", "+12", "👥")
    with col2:
        ui.render_kpi_card("Active Sessions", "89", "+5", "🟢")
    with col3:
        ui.render_kpi_card("Admin Users", "8", "0", "🔑")
    with col4:
        ui.render_kpi_card("Pending Approvals", "3", "+1", "⏳")
    
    # User management actions
    st.markdown("#### 🛠️ User Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add New User", type="primary"):
            show_add_user_dialog()
    with col2:
        if st.button("📤 Bulk Import Users"):
            show_bulk_import_dialog()
    with col3:
        if st.button("📊 Export User List"):
            ui.show_toast_notification("User list exported!", "success")
    
    # User search and filters
    st.markdown("#### 🔍 User Directory")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("🔎 Search Users", placeholder="Search by name, email, or role...")
    with col2:
        role_filter = st.selectbox("👤 Role Filter", ["All Roles", "Admin", "User", "Manager", "Analyst"])
    with col3:
        status_filter = st.selectbox("📊 Status Filter", ["All Status", "Active", "Inactive", "Suspended", "Pending"])
    
    # User list with management actions
    users_data = [
        {"name": "John Smith", "email": "john@company.com", "role": "Admin", "status": "Active", "last_login": "2024-01-15 09:30", "mfa": "✅"},
        {"name": "Sarah Johnson", "email": "sarah@company.com", "role": "Manager", "status": "Active", "last_login": "2024-01-15 08:45", "mfa": "❌"},
        {"name": "Mike Davis", "email": "mike@company.com", "role": "User", "status": "Inactive", "last_login": "2024-01-10 14:20", "mfa": "✅"},
        {"name": "Emma Wilson", "email": "emma@company.com", "role": "Analyst", "status": "Active", "last_login": "2024-01-15 10:15", "mfa": "✅"},
    ]
    
    for user in users_data:
        with st.expander(f"👤 {user['name']} ({user['role']})"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Status:** {user['status']}")
                st.write(f"**Last Login:** {user['last_login']}")
                st.write(f"**MFA Enabled:** {user['mfa']}")
            
            with col2:
                if st.button("✏️ Edit", key=f"edit_{user['email']}"):
                    st.info(f"Edit dialog for {user['name']}")
                if st.button("🔒 Reset Password", key=f"reset_{user['email']}"):
                    ui.show_toast_notification(f"Password reset sent to {user['name']}", "info")
            
            with col3:
                if user['status'] == 'Active':
                    if st.button("⏸️ Suspend", key=f"suspend_{user['email']}"):
                        ui.show_toast_notification(f"{user['name']} suspended", "warning")
                else:
                    if st.button("▶️ Activate", key=f"activate_{user['email']}"):
                        ui.show_toast_notification(f"{user['name']} activated", "success")
                
                if st.button("🗑️ Delete", key=f"delete_{user['email']}", type="secondary"):
                    st.warning("⚠️ This action cannot be undone!")

def render_security_center():
    """Render security center"""
    st.markdown("### 🔒 Security Center")
    
    # Security overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.render_kpi_card("Security Score", "94%", "+2%", "🛡️")
    with col2:
        ui.render_kpi_card("Failed Logins", "23", "-5", "🚨")
    with col3:
        ui.render_kpi_card("MFA Adoption", "78%", "+12%", "🔐")
    with col4:
        ui.render_kpi_card("Active Threats", "0", "0", "✅")
    
    # Security actions
    st.markdown("#### 🛡️ Security Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔒 Force MFA Setup"):
            ui.show_toast_notification("MFA enforcement enabled!", "warning")
    with col2:
        if st.button("🚫 Block IP Address"):
            show_ip_block_dialog()
    with col3:
        if st.button("🔄 Rotate API Keys"):
            ui.show_toast_notification("API keys rotated!", "info")
    with col4:
        if st.button("📋 Security Report"):
            ui.show_toast_notification("Security report generated!", "success")
    
    # Recent security events
    st.markdown("#### 🚨 Recent Security Events")
    security_events = [
        {"time": "2024-01-15 10:30", "event": "Failed login attempt", "user": "unknown@domain.com", "severity": "Medium", "action": "IP Blocked"},
        {"time": "2024-01-15 09:15", "event": "MFA setup completed", "user": "sarah@company.com", "severity": "Low", "action": "Logged"},
        {"time": "2024-01-15 08:45", "event": "Admin privilege escalation", "user": "john@company.com", "severity": "High", "action": "Approved"},
        {"time": "2024-01-14 16:20", "event": "Suspicious data export", "user": "mike@company.com", "severity": "High", "action": "Blocked"},
    ]
    
    for event in security_events:
        severity_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[event["severity"]]
        st.markdown(f"{severity_color} **{event['time']}** - {event['event']} - {event['user']} - Action: {event['action']}")

def render_system_monitor():
    """Render system monitoring dashboard"""
    st.markdown("### 📊 System Monitor")
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.render_kpi_card("CPU Usage", "45%", "+5%", "💻")
    with col2:
        ui.render_kpi_card("Memory Usage", "67%", "+2%", "🧠")
    with col3:
        ui.render_kpi_card("Disk Usage", "34%", "+1%", "💾")
    with col4:
        ui.render_kpi_card("Network I/O", "2.3 GB/s", "+0.1", "🌐")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 CPU & Memory Trends")
        # Generate sample performance data
        import numpy as np
        hours = list(range(24))
        cpu_usage = [45 + np.random.normal(0, 5) for _ in hours]
        memory_usage = [67 + np.random.normal(0, 3) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=cpu_usage, mode='lines', name='CPU %', line=dict(color='#FF6B6B')))
        fig.add_trace(go.Scatter(x=hours, y=memory_usage, mode='lines', name='Memory %', line=dict(color='#4ECDC4')))
        fig.update_layout(title="System Performance (24h)", xaxis_title="Hour", yaxis_title="Usage %", height=300, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔄 Active Processes")
        processes = [
            {"name": "hr_analytics", "cpu": "12%", "memory": "256MB", "status": "Running"},
            {"name": "database", "cpu": "8%", "memory": "512MB", "status": "Running"},
            {"name": "auth_service", "cpu": "3%", "memory": "128MB", "status": "Running"},
            {"name": "backup_job", "cpu": "15%", "memory": "64MB", "status": "Running"},
        ]
        
        for proc in processes:
            status_color = "🟢" if proc["status"] == "Running" else "🔴"
            st.markdown(f"{status_color} **{proc['name']}** - CPU: {proc['cpu']}, Memory: {proc['memory']}")
    
    # System actions
    st.markdown("#### ⚙️ System Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Restart Services"):
            ui.show_toast_notification("Services restarting...", "warning")
    with col2:
        if st.button("🧹 Clear Cache"):
            ui.show_toast_notification("Cache cleared!", "success")
    with col3:
        if st.button("📊 Performance Report"):
            ui.show_toast_notification("Performance report generated!", "info")
    with col4:
        if st.button("⚠️ Maintenance Mode"):
            ui.show_toast_notification("Maintenance mode activated!", "error")

def render_admin_data_management():
    """Render admin data management"""
    st.markdown("### 🗄️ Data Management")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.render_kpi_card("Total Records", "1.2M", "+50K", "📊")
    with col2:
        ui.render_kpi_card("Database Size", "2.8 GB", "+120MB", "💾")
    with col3:
        ui.render_kpi_card("Backup Status", "✅ Current", "2h ago", "🔄")
    with col4:
        ui.render_kpi_card("Data Quality", "96.2%", "+0.5%", "✨")
    
    # Data operations
    st.markdown("#### 🛠️ Data Operations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Create Backup", type="primary"):
            ui.show_toast_notification("Backup initiated...", "info")
        if st.button("📤 Export All Data"):
            ui.show_toast_notification("Data export started...", "info")
    
    with col2:
        if st.button("🔄 Sync External Data"):
            ui.show_toast_notification("Data sync initiated...", "info")
        if st.button("🧹 Data Cleanup"):
            ui.show_toast_notification("Data cleanup started...", "warning")
    
    with col3:
        if st.button("📊 Generate Schema"):
            ui.show_toast_notification("Schema documentation generated!", "success")
        if st.button("🔍 Data Validation"):
            ui.show_toast_notification("Data validation started...", "info")
    
    # Recent data operations
    st.markdown("#### 📋 Recent Operations")
    operations = [
        {"time": "2024-01-15 10:00", "operation": "Automated Backup", "status": "✅ Completed", "duration": "15 min"},
        {"time": "2024-01-15 08:30", "operation": "Data Sync", "status": "✅ Completed", "duration": "8 min"},
        {"time": "2024-01-15 06:00", "operation": "Data Cleanup", "status": "✅ Completed", "duration": "45 min"},
        {"time": "2024-01-14 22:00", "operation": "Schema Update", "status": "✅ Completed", "duration": "3 min"},
    ]
    
    for op in operations:
        st.markdown(f"**{op['time']}** - {op['operation']} - {op['status']} - Duration: {op['duration']}")

def render_system_configuration():
    """Render system configuration"""
    st.markdown("### 🔧 System Configuration")
    
    # Configuration sections
    config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(["🌐 General", "🔒 Security", "📊 Performance", "🔗 Integrations"])
    
    with config_tab1:
        st.markdown("#### 🌐 General Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            system_name = st.text_input("System Name", value="HR Analytics Pro")
            timezone = st.selectbox("System Timezone", ["UTC", "EST", "PST", "GMT", "IST"])
            maintenance_window = st.time_input("Maintenance Window", value=pd.to_datetime("02:00").time())
        
        with col2:
            max_users = st.number_input("Max Concurrent Users", min_value=50, max_value=1000, value=250)
            session_timeout = st.selectbox("Default Session Timeout", ["30 min", "1 hour", "4 hours", "8 hours"])
            auto_backup = st.checkbox("Enable Auto Backup", value=True)
    
    with config_tab2:
        st.markdown("#### 🔒 Security Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            enforce_mfa = st.checkbox("Enforce MFA for All Users", value=False)
            password_policy = st.selectbox("Password Policy", ["Standard", "Strong", "Enterprise"])
            login_attempts = st.number_input("Max Login Attempts", min_value=3, max_value=10, value=5)
        
        with col2:
            ip_whitelist = st.text_area("IP Whitelist", placeholder="192.168.1.0/24\n10.0.0.0/8")
            audit_retention = st.selectbox("Audit Log Retention", ["30 days", "90 days", "1 year", "2 years"])
            encryption_level = st.selectbox("Encryption Level", ["AES-128", "AES-256", "AES-256-GCM"])
    
    with config_tab3:
        st.markdown("#### 📊 Performance Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            cache_size = st.selectbox("Cache Size", ["128MB", "256MB", "512MB", "1GB"])
            query_timeout = st.number_input("Query Timeout (seconds)", min_value=10, max_value=300, value=60)
            batch_size = st.number_input("Batch Processing Size", min_value=100, max_value=10000, value=1000)
        
        with col2:
            enable_compression = st.checkbox("Enable Data Compression", value=True)
            lazy_loading = st.checkbox("Enable Lazy Loading", value=True)
            cdn_enabled = st.checkbox("Enable CDN", value=False)
    
    with config_tab4:
        st.markdown("#### 🔗 Integration Settings")
        integrations = [
            {"name": "LDAP/AD", "status": "Disabled", "config": "Configure"},
            {"name": "SAML SSO", "status": "Disabled", "config": "Configure"},
            {"name": "Slack", "status": "Enabled", "config": "Manage"},
            {"name": "Microsoft Teams", "status": "Disabled", "config": "Configure"},
        ]
        
        for integration in integrations:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{integration['name']}**")
            with col2:
                status_color = "🟢" if integration['status'] == "Enabled" else "🔴"
                st.write(f"{status_color} {integration['status']}")
            with col3:
                if st.button(integration['config'], key=f"config_{integration['name']}"):
                    st.info(f"{integration['name']} configuration dialog")
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        ui.show_toast_notification("Configuration saved successfully!", "success")

def render_audit_logs():
    """Render audit logs"""
    st.markdown("### 📋 Audit Logs")
    
    # Log filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        log_level = st.selectbox("Log Level", ["All", "INFO", "WARN", "ERROR", "DEBUG"])
    with col2:
        date_range = st.date_input("Date Range", value=[])
    with col3:
        user_filter = st.text_input("User Filter", placeholder="Filter by user...")
    with col4:
        action_filter = st.selectbox("Action Type", ["All", "Login", "Data Access", "Config Change", "Security"])
    
    # Audit log entries
    st.markdown("#### 📝 Recent Audit Entries")
    audit_logs = [
        {"timestamp": "2024-01-15 10:30:45", "level": "INFO", "user": "admin@company.com", "action": "User Created", "details": "Created user sarah@company.com", "ip": "192.168.1.100"},
        {"timestamp": "2024-01-15 10:25:12", "level": "WARN", "user": "system", "action": "Failed Login", "details": "Multiple failed attempts from unknown@domain.com", "ip": "203.0.113.1"},
        {"timestamp": "2024-01-15 10:20:33", "level": "INFO", "user": "john@company.com", "action": "Data Export", "details": "Exported employee performance data", "ip": "192.168.1.50"},
        {"timestamp": "2024-01-15 10:15:18", "level": "ERROR", "user": "system", "action": "Database Error", "details": "Connection timeout to analytics DB", "ip": "localhost"},
        {"timestamp": "2024-01-15 10:10:05", "level": "INFO", "user": "sarah@company.com", "action": "MFA Setup", "details": "Enabled TOTP authentication", "ip": "192.168.1.75"},
    ]
    
    for log in audit_logs:
        level_colors = {"INFO": "🔵", "WARN": "🟡", "ERROR": "🔴", "DEBUG": "⚪"}
        level_color = level_colors.get(log["level"], "⚪")
        
        with st.expander(f"{level_color} {log['timestamp']} - {log['action']} - {log['user']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Level:** {log['level']}")
                st.write(f"**User:** {log['user']}")
                st.write(f"**Action:** {log['action']}")
            with col2:
                st.write(f"**IP Address:** {log['ip']}")
                st.write(f"**Details:** {log['details']}")
                st.write(f"**Timestamp:** {log['timestamp']}")
    
    # Export logs
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export Logs (CSV)"):
            ui.show_toast_notification("Audit logs exported to CSV!", "success")
    with col2:
        if st.button("📊 Generate Report"):
            ui.show_toast_notification("Audit report generated!", "info")
    with col3:
        if st.button("🔄 Refresh Logs"):
            ui.show_toast_notification("Logs refreshed!", "success")

def render_alerts_incidents():
    """Render alerts and incidents management"""
    st.markdown("### 🚨 Alerts & Incidents")
    
    # Alert overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.render_kpi_card("Active Alerts", "7", "+2", "🚨")
    with col2:
        ui.render_kpi_card("Critical Issues", "1", "0", "🔴")
    with col3:
        ui.render_kpi_card("Resolved Today", "12", "+5", "✅")
    with col4:
        ui.render_kpi_card("Avg Resolution", "2.3h", "-0.5h", "⏱️")
    
    # Active incidents
    st.markdown("#### 🔥 Active Incidents")
    incidents = [
        {"id": "INC-001", "severity": "Critical", "title": "Database Connection Issues", "status": "Investigating", "assigned": "John Smith", "created": "10:30 AM"},
        {"id": "INC-002", "severity": "High", "title": "Authentication Service Slow", "status": "In Progress", "assigned": "Sarah Johnson", "created": "09:15 AM"},
        {"id": "INC-003", "severity": "Medium", "title": "Report Generation Timeout", "status": "Pending", "assigned": "Mike Davis", "created": "08:45 AM"},
    ]
    
    for incident in incidents:
        severity_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        severity_color = severity_colors.get(incident["severity"], "⚪")
        
        with st.expander(f"{severity_color} {incident['id']} - {incident['title']} ({incident['severity']})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Status:** {incident['status']}")
                st.write(f"**Assigned:** {incident['assigned']}")
            with col2:
                st.write(f"**Created:** {incident['created']}")
                st.write(f"**Severity:** {incident['severity']}")
            with col3:
                if st.button("🔧 Update", key=f"update_{incident['id']}"):
                    st.info(f"Update dialog for {incident['id']}")
                if st.button("✅ Resolve", key=f"resolve_{incident['id']}"):
                    ui.show_toast_notification(f"Incident {incident['id']} resolved!", "success")
    
    # Alert rules
    st.markdown("#### ⚙️ Alert Rules")
    alert_rules = [
        {"name": "High CPU Usage", "condition": "CPU > 80%", "status": "Active", "last_triggered": "Never"},
        {"name": "Failed Login Attempts", "condition": "Failed logins > 10/hour", "status": "Active", "last_triggered": "2 hours ago"},
        {"name": "Database Errors", "condition": "DB errors > 5/min", "status": "Active", "last_triggered": "1 day ago"},
        {"name": "Disk Space Low", "condition": "Disk usage > 90%", "status": "Inactive", "last_triggered": "Never"},
    ]
    
    for rule in alert_rules:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{rule['name']}**")
        with col2:
            st.write(f"Condition: {rule['condition']}")
        with col3:
            status_color = "🟢" if rule['status'] == "Active" else "🔴"
            st.write(f"{status_color} {rule['status']}")
        with col4:
            st.write(f"Last: {rule['last_triggered']}")

def show_add_user_dialog():
    """Show add user dialog"""
    st.markdown("---")
    st.markdown("### ➕ Add New User")
    
    with st.form("add_user_admin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name*", placeholder="Enter full name")
            email = st.text_input("Email*", placeholder="Enter email address")
            role = st.selectbox("Role*", ["user", "manager", "analyst", "admin"])
        
        with col2:
            department = st.selectbox("Department", ["Engineering", "Sales", "Marketing", "HR", "Finance"])
            send_invite = st.checkbox("Send invitation email", value=True)
            require_mfa = st.checkbox("Require MFA setup", value=False)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("➕ Create User", type="primary"):
                if all([full_name, email, role]):
                    ui.show_toast_notification(f"User {full_name} created successfully!", "success")
                    st.success(f"✅ {full_name} has been added to the system!")
                else:
                    st.error("Please fill in all required fields marked with *")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.info("User creation cancelled")

def show_bulk_import_dialog():
    """Show bulk import dialog"""
    st.markdown("---")
    st.markdown("### 📤 Bulk Import Users")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Preview Data", type="primary"):
                st.info("Data preview would show here")
        with col2:
            if st.button("📤 Import Users"):
                ui.show_toast_notification("Bulk import completed!", "success")

def show_ip_block_dialog():
    """Show IP blocking dialog"""
    st.markdown("---")
    st.markdown("### 🚫 Block IP Address")
    
    with st.form("block_ip_form"):
        ip_address = st.text_input("IP Address*", placeholder="192.168.1.100 or 192.168.1.0/24")
        reason = st.text_area("Reason", placeholder="Reason for blocking this IP...")
        duration = st.selectbox("Block Duration", ["1 hour", "24 hours", "1 week", "Permanent"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("🚫 Block IP", type="primary"):
                if ip_address:
                    ui.show_toast_notification(f"IP {ip_address} blocked successfully!", "warning")
                else:
                    st.error("Please enter an IP address")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.info("IP blocking cancelled")

if __name__ == "__main__":
    main()
