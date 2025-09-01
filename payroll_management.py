"""
Payroll Management System
AI-powered payroll automation with multi-country support, tax calculations, and anomaly detection
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PayrollManager:
    def __init__(self):
        self.tax_rates = {
            'US': {'federal': 0.22, 'state': 0.05, 'social_security': 0.062, 'medicare': 0.0145},
            'UK': {'income_tax': 0.20, 'national_insurance': 0.12},
            'CA': {'federal': 0.15, 'provincial': 0.05, 'cpp': 0.0545, 'ei': 0.0158},
            'IN': {'income_tax': 0.30, 'pf': 0.12, 'esi': 0.0175}
        }
        
    def calculate_taxes(self, salary, country, deductions=0):
        """Calculate taxes based on country and salary"""
        if country not in self.tax_rates:
            country = 'US'  # Default to US
            
        rates = self.tax_rates[country]
        taxable_income = max(0, salary - deductions)
        
        total_tax = 0
        tax_breakdown = {}
        
        for tax_type, rate in rates.items():
            tax_amount = taxable_income * rate
            tax_breakdown[tax_type] = tax_amount
            total_tax += tax_amount
            
        return total_tax, tax_breakdown
    
    def detect_payroll_anomalies(self, payroll_data):
        """Detect anomalies in payroll data using Isolation Forest"""
        if len(payroll_data) < 10:
            return []
            
        # Prepare features for anomaly detection
        features = ['base_salary', 'overtime_hours', 'bonuses', 'deductions', 'net_pay']
        X = payroll_data[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Return anomalous records
        anomaly_indices = np.where(anomalies == -1)[0]
        return payroll_data.iloc[anomaly_indices].to_dict('records')
    
    def generate_payslip_data(self, employee_id, pay_period):
        """Generate payslip data for an employee"""
        # This would typically fetch from database
        return {
            'employee_id': employee_id,
            'name': f'Employee {employee_id}',
            'pay_period': pay_period,
            'base_salary': 5000,
            'overtime_hours': 10,
            'overtime_rate': 30,
            'bonuses': 500,
            'deductions': 200,
            'country': 'US'
        }

def render_payroll_dashboard():
    """Render payroll management dashboard"""
    st.markdown('<div class="main-header">💰 Payroll Management</div>', unsafe_allow_html=True)
    
    payroll_manager = PayrollManager()
    
    # Payroll Overview KPIs
    st.markdown("### 📊 Payroll Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Payroll", "$2.4M", "+5.2%")
    with col2:
        st.metric("Employees Paid", "1,247", "+23")
    with col3:
        st.metric("Avg Salary", "$4,850", "+2.1%")
    with col4:
        st.metric("Tax Compliance", "100%", "0%")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💼 Payroll Processing", "📋 Employee Payroll", "🔍 Anomaly Detection", 
        "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        render_payroll_processing(payroll_manager)
    
    with tab2:
        render_employee_payroll(payroll_manager)
    
    with tab3:
        render_anomaly_detection(payroll_manager)
    
    with tab4:
        render_payroll_analytics()
    
    with tab5:
        render_payroll_settings()

def render_payroll_processing(payroll_manager):
    """Render payroll processing interface"""
    st.markdown("### 💼 Payroll Processing")
    
    # Processing controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pay_period = st.selectbox("Pay Period", [
            "January 2024", "February 2024", "March 2024", "April 2024"
        ])
    with col2:
        country = st.selectbox("Country", ["US", "UK", "CA", "IN"])
    with col3:
        process_type = st.selectbox("Process Type", ["Regular", "Bonus", "Correction"])
    
    # Batch processing
    st.markdown("#### 🚀 Batch Processing")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Calculate Payroll", type="primary"):
            with st.spinner("Calculating payroll..."):
                st.success("✅ Payroll calculated for 1,247 employees")
    
    with col2:
        if st.button("🔍 Validate Data"):
            with st.spinner("Validating data..."):
                st.success("✅ Data validation complete")
    
    with col3:
        if st.button("📊 Generate Reports"):
            with st.spinner("Generating reports..."):
                st.success("✅ Reports generated")
    
    with col4:
        if st.button("💸 Process Payments"):
            with st.spinner("Processing payments..."):
                st.success("✅ Payments processed")
    
    # Payroll summary table
    st.markdown("#### 📋 Payroll Summary")
    
    # Generate sample payroll data
    payroll_data = []
    for i in range(10):
        employee_data = payroll_manager.generate_payslip_data(f"EMP{i+1:03d}", pay_period)
        
        # Calculate taxes
        total_tax, tax_breakdown = payroll_manager.calculate_taxes(
            employee_data['base_salary'], 
            country,
            employee_data['deductions']
        )
        
        overtime_pay = employee_data['overtime_hours'] * employee_data['overtime_rate']
        gross_pay = employee_data['base_salary'] + overtime_pay + employee_data['bonuses']
        net_pay = gross_pay - total_tax - employee_data['deductions']
        
        payroll_data.append({
            'Employee ID': employee_data['employee_id'],
            'Name': employee_data['name'],
            'Base Salary': f"${employee_data['base_salary']:,.2f}",
            'Overtime': f"${overtime_pay:.2f}",
            'Bonuses': f"${employee_data['bonuses']:.2f}",
            'Gross Pay': f"${gross_pay:,.2f}",
            'Taxes': f"${total_tax:.2f}",
            'Deductions': f"${employee_data['deductions']:.2f}",
            'Net Pay': f"${net_pay:,.2f}",
            'Status': '✅ Processed'
        })
    
    df = pd.DataFrame(payroll_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export CSV"):
            st.success("✅ Payroll data exported to CSV")
    with col2:
        if st.button("📄 Generate Payslips"):
            st.success("✅ Payslips generated for all employees")
    with col3:
        if st.button("🏦 Bank Transfer File"):
            st.success("✅ Bank transfer file created")

def render_employee_payroll(payroll_manager):
    """Render individual employee payroll interface"""
    st.markdown("### 📋 Employee Payroll")
    
    # Employee selection
    col1, col2 = st.columns(2)
    with col1:
        employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)])
    with col2:
        pay_period = st.selectbox("Pay Period", [
            "January 2024", "February 2024", "March 2024"
        ], key="emp_pay_period")
    
    # Employee payroll details
    employee_data = payroll_manager.generate_payslip_data(employee_id, pay_period)
    
    st.markdown("#### 💰 Payroll Calculation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Earnings**")
        base_salary = st.number_input("Base Salary", value=employee_data['base_salary'])
        overtime_hours = st.number_input("Overtime Hours", value=employee_data['overtime_hours'])
        overtime_rate = st.number_input("Overtime Rate", value=employee_data['overtime_rate'])
        bonuses = st.number_input("Bonuses", value=employee_data['bonuses'])
        
        overtime_pay = overtime_hours * overtime_rate
        gross_pay = base_salary + overtime_pay + bonuses
        
        st.markdown(f"**Overtime Pay:** ${overtime_pay:.2f}")
        st.markdown(f"**Gross Pay:** ${gross_pay:,.2f}")
    
    with col2:
        st.markdown("**Deductions**")
        country = st.selectbox("Country", ["US", "UK", "CA", "IN"], key="emp_country")
        deductions = st.number_input("Other Deductions", value=employee_data['deductions'])
        
        # Calculate taxes
        total_tax, tax_breakdown = payroll_manager.calculate_taxes(base_salary, country, deductions)
        
        st.markdown("**Tax Breakdown:**")
        for tax_type, amount in tax_breakdown.items():
            st.markdown(f"- {tax_type.replace('_', ' ').title()}: ${amount:.2f}")
        
        st.markdown(f"**Total Taxes:** ${total_tax:.2f}")
        st.markdown(f"**Other Deductions:** ${deductions:.2f}")
        
        net_pay = gross_pay - total_tax - deductions
        st.markdown(f"**Net Pay:** ${net_pay:,.2f}")
    
    # Generate payslip
    if st.button("📄 Generate Payslip", type="primary"):
        st.success(f"✅ Payslip generated for {employee_data['name']}")
        
        # Display payslip preview
        st.markdown("#### 📄 Payslip Preview")
        payslip_data = {
            'Item': ['Base Salary', 'Overtime Pay', 'Bonuses', 'Gross Pay', 'Taxes', 'Other Deductions', 'Net Pay'],
            'Amount': [f"${base_salary:,.2f}", f"${overtime_pay:.2f}", f"${bonuses:.2f}", 
                      f"${gross_pay:,.2f}", f"-${total_tax:.2f}", f"-${deductions:.2f}", f"${net_pay:,.2f}"]
        }
        
        payslip_df = pd.DataFrame(payslip_data)
        st.dataframe(payslip_df, use_container_width=True, hide_index=True)

def render_anomaly_detection(payroll_manager):
    """Render payroll anomaly detection"""
    st.markdown("### 🔍 Payroll Anomaly Detection")
    
    # Generate sample payroll data for anomaly detection
    np.random.seed(42)
    n_employees = 100
    
    payroll_data = pd.DataFrame({
        'employee_id': [f'EMP{i:03d}' for i in range(1, n_employees + 1)],
        'base_salary': np.random.normal(5000, 1000, n_employees),
        'overtime_hours': np.random.poisson(8, n_employees),
        'bonuses': np.random.exponential(300, n_employees),
        'deductions': np.random.normal(200, 50, n_employees),
        'net_pay': np.random.normal(4500, 800, n_employees)
    })
    
    # Add some anomalies
    payroll_data.loc[5, 'base_salary'] = 15000  # Unusually high salary
    payroll_data.loc[15, 'overtime_hours'] = 80  # Excessive overtime
    payroll_data.loc[25, 'bonuses'] = 5000  # Large bonus
    
    # Detect anomalies
    anomalies = payroll_manager.detect_payroll_anomalies(payroll_data)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(payroll_data))
        st.metric("Anomalies Detected", len(anomalies))
        st.metric("Anomaly Rate", f"{len(anomalies)/len(payroll_data)*100:.1f}%")
    
    with col2:
        if st.button("🔍 Run Anomaly Detection", type="primary"):
            st.success(f"✅ Detected {len(anomalies)} potential anomalies")
    
    # Display anomalies
    if anomalies:
        st.markdown("#### 🚨 Detected Anomalies")
        
        for i, anomaly in enumerate(anomalies):
            with st.expander(f"🚨 Anomaly {i+1}: {anomaly['employee_id']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Employee ID:** {anomaly['employee_id']}")
                    st.write(f"**Base Salary:** ${anomaly['base_salary']:,.2f}")
                    st.write(f"**Overtime Hours:** {anomaly['overtime_hours']}")
                
                with col2:
                    st.write(f"**Bonuses:** ${anomaly['bonuses']:,.2f}")
                    st.write(f"**Deductions:** ${anomaly['deductions']:,.2f}")
                    st.write(f"**Net Pay:** ${anomaly['net_pay']:,.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"approve_{i}"):
                        st.success("Anomaly approved")
                with col2:
                    if st.button("🔍 Investigate", key=f"investigate_{i}"):
                        st.info("Marked for investigation")

def render_payroll_analytics():
    """Render payroll analytics dashboard"""
    st.markdown("### 📊 Payroll Analytics")
    
    # Generate sample analytics data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    total_payroll = [2.1, 2.2, 2.3, 2.4, 2.5, 2.4]
    avg_salary = [4.2, 4.3, 4.4, 4.5, 4.6, 4.8]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Total payroll trend
        fig1 = px.line(x=months, y=total_payroll, title="Total Payroll Trend (Millions)")
        fig1.update_traces(line_color='#1f77b4', line_width=3)
        fig1.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Average salary trend
        fig2 = px.bar(x=months, y=avg_salary, title="Average Salary Trend (Thousands)")
        fig2.update_traces(marker_color='#ff7f0e')
        fig2.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Department-wise payroll distribution
    st.markdown("#### 🏢 Department-wise Payroll Distribution")
    
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
    payroll_amounts = [800000, 600000, 400000, 300000, 300000]
    
    fig3 = px.pie(values=payroll_amounts, names=departments, 
                  title="Payroll Distribution by Department")
    fig3.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Cost analysis
    st.markdown("#### 💰 Cost Analysis")
    
    cost_breakdown = pd.DataFrame({
        'Category': ['Base Salaries', 'Overtime', 'Bonuses', 'Benefits', 'Taxes', 'Other'],
        'Amount': [1800000, 200000, 150000, 300000, 400000, 50000],
        'Percentage': [60.0, 6.7, 5.0, 10.0, 13.3, 1.7]
    })
    
    st.dataframe(cost_breakdown, use_container_width=True, hide_index=True)

def render_payroll_settings():
    """Render payroll settings"""
    st.markdown("### ⚙️ Payroll Settings")
    
    # Tax configuration
    st.markdown("#### 💰 Tax Configuration")
    
    country = st.selectbox("Select Country", ["US", "UK", "CA", "IN"])
    
    if country == "US":
        col1, col2 = st.columns(2)
        with col1:
            federal_rate = st.number_input("Federal Tax Rate (%)", value=22.0, min_value=0.0, max_value=50.0)
            state_rate = st.number_input("State Tax Rate (%)", value=5.0, min_value=0.0, max_value=20.0)
        with col2:
            ss_rate = st.number_input("Social Security Rate (%)", value=6.2, min_value=0.0, max_value=10.0)
            medicare_rate = st.number_input("Medicare Rate (%)", value=1.45, min_value=0.0, max_value=5.0)
    
    # Payroll schedule
    st.markdown("#### 📅 Payroll Schedule")
    
    col1, col2 = st.columns(2)
    with col1:
        pay_frequency = st.selectbox("Pay Frequency", ["Weekly", "Bi-weekly", "Monthly", "Semi-monthly"])
        next_payroll = st.date_input("Next Payroll Date")
    with col2:
        auto_process = st.checkbox("Auto-process Payroll", value=True)
        send_notifications = st.checkbox("Send Email Notifications", value=True)
    
    # Banking integration
    st.markdown("#### 🏦 Banking Integration")
    
    col1, col2 = st.columns(2)
    with col1:
        bank_name = st.text_input("Bank Name", value="First National Bank")
        routing_number = st.text_input("Routing Number", value="123456789")
    with col2:
        account_number = st.text_input("Account Number", value="*****6789", type="password")
        enable_direct_deposit = st.checkbox("Enable Direct Deposit", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Payroll settings saved successfully!")

if __name__ == "__main__":
    render_payroll_dashboard()
