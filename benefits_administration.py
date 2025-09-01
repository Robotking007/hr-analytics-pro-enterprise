"""
Benefits Administration System
AI-powered benefits management with employee self-service portal and intelligent recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class BenefitsManager:
    def __init__(self):
        self.benefit_types = {
            'health': {
                'name': 'Health Insurance',
                'plans': ['Basic', 'Premium', 'Family'],
                'costs': [200, 350, 500],
                'coverage': ['Medical', 'Dental', 'Vision']
            },
            'retirement': {
                'name': 'Retirement Plans',
                'plans': ['401k Basic', '401k Plus', 'Pension'],
                'costs': [0, 50, 100],
                'coverage': ['Company Match', 'Investment Options', 'Vesting']
            },
            'life': {
                'name': 'Life Insurance',
                'plans': ['Basic', '2x Salary', '5x Salary'],
                'costs': [0, 25, 75],
                'coverage': ['Term Life', 'AD&D', 'Beneficiary']
            },
            'disability': {
                'name': 'Disability Insurance',
                'plans': ['Short-term', 'Long-term', 'Both'],
                'costs': [15, 30, 40],
                'coverage': ['Income Protection', 'Medical Coverage']
            },
            'wellness': {
                'name': 'Wellness Programs',
                'plans': ['Gym Membership', 'Mental Health', 'Full Wellness'],
                'costs': [30, 50, 80],
                'coverage': ['Fitness', 'Counseling', 'Preventive Care']
            }
        }
        
        self.employee_profiles = self._generate_employee_profiles()
    
    def _generate_employee_profiles(self):
        """Generate sample employee profiles for recommendations"""
        np.random.seed(42)
        profiles = []
        
        for i in range(50):
            profile = {
                'employee_id': f'EMP{i+1:03d}',
                'age': np.random.randint(22, 65),
                'salary': np.random.normal(60000, 20000),
                'family_size': np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05]),
                'health_conditions': np.random.choice(['None', 'Chronic', 'Acute'], p=[0.7, 0.2, 0.1]),
                'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2])
            }
            profiles.append(profile)
        
        return profiles
    
    def recommend_benefits(self, employee_profile):
        """AI-powered benefit recommendations based on employee profile"""
        recommendations = []
        
        age = employee_profile['age']
        salary = employee_profile['salary']
        family_size = employee_profile['family_size']
        health_conditions = employee_profile['health_conditions']
        risk_tolerance = employee_profile['risk_tolerance']
        
        # Health insurance recommendations
        if family_size > 2:
            recommendations.append({
                'type': 'health',
                'plan': 'Family',
                'reason': 'Recommended for families with multiple dependents',
                'priority': 'High',
                'estimated_cost': 500
            })
        elif health_conditions != 'None':
            recommendations.append({
                'type': 'health',
                'plan': 'Premium',
                'reason': 'Enhanced coverage for existing health conditions',
                'priority': 'High',
                'estimated_cost': 350
            })
        else:
            recommendations.append({
                'type': 'health',
                'plan': 'Basic',
                'reason': 'Cost-effective coverage for healthy individuals',
                'priority': 'Medium',
                'estimated_cost': 200
            })
        
        # Retirement recommendations
        if age > 40:
            recommendations.append({
                'type': 'retirement',
                'plan': '401k Plus',
                'reason': 'Maximize retirement savings with enhanced matching',
                'priority': 'High',
                'estimated_cost': 50
            })
        elif salary > 70000:
            recommendations.append({
                'type': 'retirement',
                'plan': '401k Plus',
                'reason': 'Higher income allows for enhanced retirement planning',
                'priority': 'Medium',
                'estimated_cost': 50
            })
        else:
            recommendations.append({
                'type': 'retirement',
                'plan': '401k Basic',
                'reason': 'Start building retirement savings with company match',
                'priority': 'Medium',
                'estimated_cost': 0
            })
        
        # Life insurance recommendations
        if family_size > 1:
            if salary > 80000:
                recommendations.append({
                    'type': 'life',
                    'plan': '5x Salary',
                    'reason': 'Comprehensive protection for high-earning families',
                    'priority': 'High',
                    'estimated_cost': 75
                })
            else:
                recommendations.append({
                    'type': 'life',
                    'plan': '2x Salary',
                    'reason': 'Adequate protection for family dependents',
                    'priority': 'Medium',
                    'estimated_cost': 25
                })
        
        # Disability insurance
        if salary > 50000:
            recommendations.append({
                'type': 'disability',
                'plan': 'Both',
                'reason': 'Protect income with comprehensive disability coverage',
                'priority': 'Medium',
                'estimated_cost': 40
            })
        
        # Wellness programs
        if age < 35 or health_conditions == 'None':
            recommendations.append({
                'type': 'wellness',
                'plan': 'Gym Membership',
                'reason': 'Maintain health and prevent future medical costs',
                'priority': 'Low',
                'estimated_cost': 30
            })
        
        return recommendations
    
    def calculate_benefits_cost(self, selected_benefits):
        """Calculate total cost of selected benefits"""
        total_cost = 0
        cost_breakdown = {}
        
        for benefit_type, plan in selected_benefits.items():
            if benefit_type in self.benefit_types:
                plans = self.benefit_types[benefit_type]['plans']
                costs = self.benefit_types[benefit_type]['costs']
                
                if plan in plans:
                    plan_index = plans.index(plan)
                    cost = costs[plan_index]
                    cost_breakdown[benefit_type] = cost
                    total_cost += cost
        
        return total_cost, cost_breakdown

def render_benefits_dashboard():
    """Render benefits administration dashboard"""
    st.markdown('<div class="main-header">🏥 Benefits Administration</div>', unsafe_allow_html=True)
    
    benefits_manager = BenefitsManager()
    
    # Benefits Overview KPIs
    st.markdown("### 📊 Benefits Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Benefits Cost", "$1.8M", "+3.2%")
    with col2:
        st.metric("Enrolled Employees", "1,156", "+45")
    with col3:
        st.metric("Avg Cost per Employee", "$1,558", "-2.1%")
    with col4:
        st.metric("Enrollment Rate", "92.7%", "+1.5%")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Employee Portal", "📋 Benefits Catalog", "🤖 AI Recommendations", 
        "📊 Analytics", "⚙️ Administration"
    ])
    
    with tab1:
        render_employee_portal(benefits_manager)
    
    with tab2:
        render_benefits_catalog(benefits_manager)
    
    with tab3:
        render_ai_recommendations(benefits_manager)
    
    with tab4:
        render_benefits_analytics()
    
    with tab5:
        render_benefits_administration()

def render_employee_portal(benefits_manager):
    """Render employee self-service portal"""
    st.markdown("### 🏥 Employee Benefits Portal")
    
    # Employee selection
    employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)])
    
    # Get employee profile
    employee_profile = next((p for p in benefits_manager.employee_profiles if p['employee_id'] == employee_id), 
                           benefits_manager.employee_profiles[0])
    
    # Employee information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age", employee_profile['age'])
    with col2:
        st.metric("Salary", f"${employee_profile['salary']:,.0f}")
    with col3:
        st.metric("Family Size", employee_profile['family_size'])
    
    # Current enrollments
    st.markdown("#### 📋 Current Enrollments")
    
    current_benefits = {
        'Health Insurance': 'Premium Plan',
        'Retirement': '401k Basic',
        'Life Insurance': '2x Salary',
        'Wellness': 'Gym Membership'
    }
    
    for benefit, plan in current_benefits.items():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{benefit}:** {plan}")
        with col2:
            st.write("Status: ✅ Active")
        with col3:
            if st.button("📝", key=f"edit_{benefit}"):
                st.info(f"Edit {benefit} enrollment")
    
    # Benefits enrollment
    st.markdown("#### 🔄 Modify Benefits")
    
    with st.expander("🏥 Health Insurance"):
        health_plan = st.selectbox("Select Health Plan", 
                                 benefits_manager.benefit_types['health']['plans'])
        health_cost = benefits_manager.benefit_types['health']['costs'][
            benefits_manager.benefit_types['health']['plans'].index(health_plan)
        ]
        st.write(f"Monthly Cost: ${health_cost}")
        
        if st.button("Update Health Insurance"):
            st.success(f"✅ Health insurance updated to {health_plan}")
    
    with st.expander("💰 Retirement Plans"):
        retirement_plan = st.selectbox("Select Retirement Plan", 
                                     benefits_manager.benefit_types['retirement']['plans'])
        retirement_cost = benefits_manager.benefit_types['retirement']['costs'][
            benefits_manager.benefit_types['retirement']['plans'].index(retirement_plan)
        ]
        st.write(f"Monthly Cost: ${retirement_cost}")
        
        if st.button("Update Retirement Plan"):
            st.success(f"✅ Retirement plan updated to {retirement_plan}")
    
    # Benefits summary
    st.markdown("#### 💰 Benefits Cost Summary")
    
    selected_benefits = {
        'health': health_plan,
        'retirement': retirement_plan
    }
    
    total_cost, cost_breakdown = benefits_manager.calculate_benefits_cost(selected_benefits)
    
    summary_data = []
    for benefit_type, cost in cost_breakdown.items():
        benefit_name = benefits_manager.benefit_types[benefit_type]['name']
        summary_data.append({
            'Benefit': benefit_name,
            'Plan': selected_benefits[benefit_type],
            'Monthly Cost': f"${cost}",
            'Annual Cost': f"${cost * 12}"
        })
    
    summary_data.append({
        'Benefit': '**Total**',
        'Plan': '',
        'Monthly Cost': f"**${total_cost}**",
        'Annual Cost': f"**${total_cost * 12}**"
    })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

def render_benefits_catalog(benefits_manager):
    """Render benefits catalog"""
    st.markdown("### 📋 Benefits Catalog")
    
    # Benefits overview
    for benefit_type, details in benefits_manager.benefit_types.items():
        with st.expander(f"{details['name']} - {len(details['plans'])} Plans Available"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Plans:**")
                for i, plan in enumerate(details['plans']):
                    cost = details['costs'][i]
                    st.write(f"• **{plan}** - ${cost}/month")
            
            with col2:
                st.markdown("**Coverage Includes:**")
                for coverage in details['coverage']:
                    st.write(f"• {coverage}")
            
            # Plan comparison
            st.markdown("**Plan Comparison:**")
            comparison_data = []
            for i, plan in enumerate(details['plans']):
                comparison_data.append({
                    'Plan': plan,
                    'Monthly Cost': f"${details['costs'][i]}",
                    'Annual Cost': f"${details['costs'][i] * 12}",
                    'Coverage Level': ['Basic', 'Standard', 'Premium'][min(i, 2)]
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

def render_ai_recommendations(benefits_manager):
    """Render AI-powered benefits recommendations"""
    st.markdown("### 🤖 AI Benefits Recommendations")
    
    # Employee selection for recommendations
    employee_id = st.selectbox("Select Employee for Recommendations", 
                              [f"EMP{i:03d}" for i in range(1, 21)], 
                              key="rec_employee")
    
    # Get employee profile
    employee_profile = next((p for p in benefits_manager.employee_profiles if p['employee_id'] == employee_id), 
                           benefits_manager.employee_profiles[0])
    
    # Display employee profile
    st.markdown("#### 👤 Employee Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", employee_profile['age'])
    with col2:
        st.metric("Salary", f"${employee_profile['salary']:,.0f}")
    with col3:
        st.metric("Family Size", employee_profile['family_size'])
    with col4:
        st.metric("Risk Tolerance", employee_profile['risk_tolerance'])
    
    # Generate recommendations
    if st.button("🤖 Generate AI Recommendations", type="primary"):
        recommendations = benefits_manager.recommend_benefits(employee_profile)
        
        st.markdown("#### 💡 Personalized Recommendations")
        
        for i, rec in enumerate(recommendations):
            benefit_name = benefits_manager.benefit_types[rec['type']]['name']
            
            # Priority color coding
            priority_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            priority_icon = priority_colors.get(rec['priority'], '⚪')
            
            with st.expander(f"{priority_icon} {benefit_name} - {rec['plan']} ({rec['priority']} Priority)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Recommended Plan:** {rec['plan']}")
                    st.write(f"**Monthly Cost:** ${rec['estimated_cost']}")
                    st.write(f"**Annual Cost:** ${rec['estimated_cost'] * 12}")
                
                with col2:
                    st.write(f"**Priority:** {rec['priority']}")
                    st.write(f"**Reason:** {rec['reason']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Accept Recommendation", key=f"accept_{i}"):
                        st.success(f"✅ {benefit_name} recommendation accepted!")
                with col2:
                    if st.button("❌ Decline", key=f"decline_{i}"):
                        st.info(f"Recommendation declined for {benefit_name}")
        
        # Total recommended cost
        total_recommended_cost = sum(rec['estimated_cost'] for rec in recommendations)
        st.markdown(f"#### 💰 Total Recommended Monthly Cost: ${total_recommended_cost}")
        st.markdown(f"**Annual Cost:** ${total_recommended_cost * 12}")
        
        # Cost as percentage of salary
        cost_percentage = (total_recommended_cost * 12) / employee_profile['salary'] * 100
        st.markdown(f"**Cost as % of Salary:** {cost_percentage:.1f}%")
        
        if cost_percentage > 15:
            st.warning("⚠️ Recommended benefits exceed 15% of salary. Consider adjusting selections.")
        else:
            st.success("✅ Recommended benefits are within optimal cost range.")

def render_benefits_analytics():
    """Render benefits analytics dashboard"""
    st.markdown("### 📊 Benefits Analytics")
    
    # Enrollment statistics
    st.markdown("#### 📈 Enrollment Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enrollment by benefit type
        benefit_names = ['Health Insurance', 'Retirement', 'Life Insurance', 'Disability', 'Wellness']
        enrollment_rates = [95, 87, 78, 65, 45]
        
        fig1 = px.bar(x=benefit_names, y=enrollment_rates, 
                     title="Enrollment Rates by Benefit Type (%)")
        fig1.update_traces(marker_color='#1f77b4')
        fig1.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cost distribution
        costs = [800000, 500000, 300000, 150000, 100000]
        
        fig2 = px.pie(values=costs, names=benefit_names, 
                     title="Benefits Cost Distribution")
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly trends
    st.markdown("#### 📅 Monthly Trends")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    total_cost = [1.6, 1.65, 1.7, 1.75, 1.8, 1.85]
    enrollment = [1100, 1120, 1140, 1150, 1156, 1160]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=months, y=total_cost, name='Total Cost (M)', 
                             line=dict(color='#ff7f0e', width=3)))
    fig3.add_trace(go.Scatter(x=months, y=[e/1000 for e in enrollment], name='Enrollment (K)', 
                             line=dict(color='#2ca02c', width=3), yaxis='y2'))
    
    fig3.update_layout(
        title="Benefits Cost and Enrollment Trends",
        xaxis_title="Month",
        yaxis_title="Cost (Millions)",
        yaxis2=dict(title="Enrollment (Thousands)", overlaying='y', side='right'),
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Department analysis
    st.markdown("#### 🏢 Department Analysis")
    
    dept_data = pd.DataFrame({
        'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'],
        'Employees': [450, 300, 200, 150, 100],
        'Avg Benefits Cost': [1800, 1600, 1500, 1400, 1700],
        'Enrollment Rate': [98, 95, 92, 90, 94]
    })
    
    st.dataframe(dept_data, use_container_width=True, hide_index=True)

def render_benefits_administration():
    """Render benefits administration panel"""
    st.markdown("### ⚙️ Benefits Administration")
    
    # Plan management
    st.markdown("#### 📋 Plan Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add New Plan", type="primary"):
            st.success("✅ New plan creation form opened")
    
    with col2:
        if st.button("📝 Modify Existing Plan"):
            st.success("✅ Plan modification interface opened")
    
    with col3:
        if st.button("🗑️ Retire Plan"):
            st.success("✅ Plan retirement process initiated")
    
    # Enrollment periods
    st.markdown("#### 📅 Enrollment Periods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Open Enrollment**")
        open_start = st.date_input("Start Date", value=datetime(2024, 11, 1))
        open_end = st.date_input("End Date", value=datetime(2024, 11, 30))
        
        if st.button("🚀 Start Open Enrollment"):
            st.success("✅ Open enrollment period activated")
    
    with col2:
        st.markdown("**Special Enrollment Events**")
        event_types = st.multiselect("Qualifying Events", [
            "Marriage", "Birth/Adoption", "Job Loss", "Divorce", "Death in Family"
        ])
        
        if st.button("📝 Configure Events"):
            st.success("✅ Special enrollment events configured")
    
    # Compliance and reporting
    st.markdown("#### 📊 Compliance & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 ACA Compliance Report"):
            st.success("✅ ACA compliance report generated")
    
    with col2:
        if st.button("💰 Cost Analysis Report"):
            st.success("✅ Cost analysis report generated")
    
    with col3:
        if st.button("📈 Utilization Report"):
            st.success("✅ Benefits utilization report generated")
    
    # Vendor management
    st.markdown("#### 🤝 Vendor Management")
    
    vendors = pd.DataFrame({
        'Vendor': ['HealthCorp', 'RetirePlan Inc', 'LifeGuard', 'WellnessCo'],
        'Service': ['Health Insurance', 'Retirement Plans', 'Life Insurance', 'Wellness Programs'],
        'Contract End': ['2024-12-31', '2025-06-30', '2024-09-30', '2025-03-31'],
        'Status': ['Active', 'Active', 'Renewal Needed', 'Active']
    })
    
    st.dataframe(vendors, use_container_width=True, hide_index=True)
    
    # Notifications and reminders
    st.markdown("#### 🔔 Automated Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enrollment_reminders = st.checkbox("Enrollment Deadline Reminders", value=True)
        benefit_changes = st.checkbox("Benefit Plan Changes", value=True)
    
    with col2:
        renewal_alerts = st.checkbox("Contract Renewal Alerts", value=True)
        compliance_updates = st.checkbox("Compliance Updates", value=True)
    
    if st.button("💾 Save Notification Settings", type="primary"):
        st.success("✅ Notification settings saved successfully!")

if __name__ == "__main__":
    render_benefits_dashboard()
