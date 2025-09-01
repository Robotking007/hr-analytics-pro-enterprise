import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import networkx as nx
from apscheduler.schedulers.background import BackgroundScheduler
import warnings
warnings.filterwarnings('ignore')

def render_workflow_dashboard():
    """Main Workflow Automation Dashboard"""
    st.title("🔄 Workflow Automation & Integrations")
    
    if 'workflows' not in st.session_state:
        st.session_state.workflows = []
        generate_sample_workflow_data()
    
    st.sidebar.markdown("### 🔄 Workflow Tools")
    workflow_page = st.sidebar.radio("Select Module", [
        "📊 Workflow Dashboard",
        "📝 Leave Approvals", 
        "💰 Expense Requests",
        "🔗 Integrations",
        "📅 Scheduled Reports",
        "🔧 Workflow Builder"
    ])
    
    if workflow_page == "📊 Workflow Dashboard":
        render_workflow_overview()
    elif workflow_page == "📝 Leave Approvals":
        render_leave_approvals()
    elif workflow_page == "💰 Expense Requests":
        render_expense_requests()
    elif workflow_page == "🔗 Integrations":
        render_integrations()
    elif workflow_page == "📅 Scheduled Reports":
        render_scheduled_reports()
    elif workflow_page == "🔧 Workflow Builder":
        render_workflow_builder()

def render_workflow_overview():
    """Workflow Overview Dashboard"""
    st.header("📊 Workflow Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pending Approvals", "23", "↑ 5")
    with col2:
        st.metric("Avg Processing Time", "2.3 days", "↓ 0.5")
    with col3:
        st.metric("Automation Rate", "87%", "↑ 12%")
    with col4:
        st.metric("SLA Compliance", "94%", "↑ 3%")
    
    # Workflow performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        processing_times = np.random.uniform(1.5, 4.0, 12)
        
        fig = px.line(x=dates, y=processing_times,
                     title="Average Processing Time (Days)")
        fig.add_hline(y=3, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        workflow_types = ['Leave', 'Expense', 'Equipment', 'Training', 'Promotion']
        volumes = [45, 32, 18, 25, 12]
        
        fig = px.pie(values=volumes, names=workflow_types,
                    title="Workflow Volume Distribution")
        st.plotly_chart(fig, use_container_width=True)

def render_leave_approvals():
    """Leave Approval System"""
    st.header("📝 Leave Approval Workflow")
    
    tab1, tab2 = st.tabs(["📋 Pending Requests", "➕ Submit Request"])
    
    with tab1:
        leave_requests = [
            {"id": "LR001", "employee": "John Smith", "type": "Annual", "days": 5, "status": "Pending Manager"},
            {"id": "LR002", "employee": "Sarah Johnson", "type": "Sick", "days": 3, "status": "Pending HR"},
            {"id": "LR003", "employee": "Mike Chen", "type": "Parental", "days": 60, "status": "Pending Director"}
        ]
        
        for req in leave_requests:
            with st.expander(f"📄 {req['id']} - {req['employee']} ({req['days']} days)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Type: {req['type']}")
                    st.write(f"Days: {req['days']}")
                with col2:
                    st.write(f"Status: {req['status']}")
                    ai_score = np.random.uniform(0.6, 0.95)
                    st.write(f"AI Score: {ai_score:.2f}")
                with col3:
                    if st.button("✅ Approve", key=f"app_{req['id']}"):
                        st.success("Approved!")
                    if st.button("❌ Reject", key=f"rej_{req['id']}"):
                        st.error("Rejected")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            leave_type = st.selectbox("Leave Type", ["Annual", "Sick", "Personal", "Parental"], key="leave_submit")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
        with col2:
            reason = st.text_area("Reason")
            emergency_contact = st.text_input("Emergency Contact")
        
        if st.button("Submit Leave Request"):
            st.success("Leave request submitted and routed automatically!")

def render_expense_requests():
    """Expense Request System"""
    st.header("💰 Expense Request Workflow")
    
    tab1, tab2 = st.tabs(["📋 Pending Expenses", "➕ Submit Expense"])
    
    with tab1:
        expenses = [
            {"id": "EX001", "employee": "Lisa Wang", "category": "Travel", "amount": 1250, "status": "Pending Finance"},
            {"id": "EX002", "employee": "Alex Brown", "category": "Equipment", "amount": 850, "status": "Pending Manager"}
        ]
        
        for exp in expenses:
            with st.expander(f"💳 {exp['id']} - {exp['employee']} (${exp['amount']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Category: {exp['category']}")
                    st.write(f"Amount: ${exp['amount']}")
                with col2:
                    st.write(f"Status: {exp['status']}")
                    compliance = np.random.choice(['Compliant', 'Review Required'])
                    st.write(f"Compliance: {compliance}")
                with col3:
                    if st.button("✅ Approve", key=f"app_exp_{exp['id']}"):
                        st.success("Expense approved!")
                    if st.button("❌ Reject", key=f"rej_exp_{exp['id']}"):
                        st.error("Expense rejected")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            expense_category = st.selectbox("Category", ["Travel", "Equipment", "Training", "Meals"], key="exp_cat")
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
            expense_date = st.date_input("Expense Date")
        with col2:
            description = st.text_area("Description")
            receipt_upload = st.file_uploader("Upload Receipt", type=['pdf', 'jpg', 'png'])
        
        if st.button("Submit Expense"):
            st.success("Expense submitted and routed for approval!")

def render_integrations():
    """System Integrations"""
    st.header("🔗 System Integrations")
    
    tab1, tab2 = st.tabs(["💬 Slack Integration", "📧 Email Automation"])
    
    with tab1:
        st.subheader("💬 Slack Integration")
        
        col1, col2 = st.columns(2)
        with col1:
            slack_connected = st.checkbox("Slack Connected", value=True, disabled=True)
            if slack_connected:
                st.success("✅ Connected to Slack")
                hr_channel = st.text_input("HR Channel", value="#hr-updates")
                payroll_channel = st.text_input("Payroll Channel", value="#payroll")
        
        with col2:
            notify_leave = st.checkbox("Leave Notifications", value=True)
            notify_expense = st.checkbox("Expense Notifications", value=True)
            notify_birthdays = st.checkbox("Birthday Alerts", value=True)
        
        if st.button("Test Slack Integration"):
            st.info("🔔 Test message sent to #hr-updates")
    
    with tab2:
        st.subheader("📧 Email Automation")
        
        template_type = st.selectbox("Email Template", 
                                   ["Leave Approval", "Expense Approval", "Birthday Wishes"], 
                                   key="email_template")
        
        auto_reminders = st.checkbox("Automatic Reminders", value=True)
        escalation_emails = st.checkbox("Escalation Notifications", value=True)

def render_scheduled_reports():
    """Scheduled Reports Management"""
    st.header("📅 Scheduled Reports")
    
    tab1, tab2 = st.tabs(["📊 Active Schedules", "➕ Create Schedule"])
    
    with tab1:
        schedules = [
            {"name": "Weekly Attendance", "frequency": "Weekly", "next_run": "2024-09-08", "status": "Active"},
            {"name": "Monthly Payroll", "frequency": "Monthly", "next_run": "2024-10-01", "status": "Active"}
        ]
        
        for schedule in schedules:
            with st.expander(f"📊 {schedule['name']} - {schedule['frequency']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Frequency: {schedule['frequency']}")
                with col2:
                    st.write(f"Next Run: {schedule['next_run']}")
                with col3:
                    if st.button("▶️ Run Now", key=f"run_{schedule['name']}"):
                        st.success("Report generated!")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            report_name = st.text_input("Report Name")
            report_type = st.selectbox("Report Type", 
                                     ["Attendance", "Payroll", "Performance", "Engagement"], 
                                     key="sched_report")
            frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], key="sched_freq")
        with col2:
            recipients = st.text_area("Email Recipients")
            include_charts = st.checkbox("Include Charts", value=True)
        
        if st.button("Create Schedule"):
            st.success("Report schedule created!")

def render_workflow_builder():
    """Visual Workflow Builder"""
    st.header("🔧 Workflow Builder")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Add Components:**")
        component = st.selectbox("Component Type", 
                               ["Start", "Approval", "Condition", "Action", "End"], 
                               key="workflow_comp")
        
        if component == "Approval":
            approver = st.selectbox("Approver", ["Manager", "HR", "Finance"], key="approver")
            timeout = st.number_input("Timeout (days)", value=3)
        
        if st.button("Add Component"):
            st.success(f"{component} added to workflow")
    
    with col2:
        st.write("**Workflow Canvas:**")
        workflow_steps = ["Start", "Manager Approval", "HR Review", "Complete"]
        
        fig = go.Figure()
        x_pos = list(range(len(workflow_steps)))
        y_pos = [0] * len(workflow_steps)
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(size=50, color='lightblue'),
            text=workflow_steps,
            textposition="middle center"
        ))
        
        fig.update_layout(
            title="Sample Workflow",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def generate_sample_workflow_data():
    """Generate sample workflow data"""
    approval_types = ['Leave Request', 'Expense Claim', 'Equipment Request']
    employees = ['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Wang']
    statuses = ['Pending Manager', 'Pending HR', 'Pending Finance']
    
    sample_approvals = []
    for i in range(15):
        approval = {
            "ID": f"WF{i+1:03d}",
            "Type": np.random.choice(approval_types),
            "Employee": np.random.choice(employees),
            "Status": np.random.choice(statuses),
            "Submitted": (datetime.now() - timedelta(days=np.random.randint(1, 10))).strftime('%Y-%m-%d'),
            "Amount": f"${np.random.randint(100, 2000)}" if np.random.choice(approval_types) == 'Expense Claim' else "N/A"
        }
        sample_approvals.append(approval)
    
    st.session_state.pending_approvals = sample_approvals

if __name__ == "__main__":
    render_workflow_dashboard()
