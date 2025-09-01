"""
Time & Attendance Tracking System
AI-powered attendance management with clock-in/out, shift scheduling, PTO management, and predictive analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import holidays
import warnings
warnings.filterwarnings('ignore')

class TimeAttendanceManager:
    def __init__(self):
        self.us_holidays = holidays.US()
        self.shift_types = {
            'morning': {'start': time(9, 0), 'end': time(17, 0), 'hours': 8},
            'evening': {'start': time(17, 0), 'end': time(1, 0), 'hours': 8},
            'night': {'start': time(23, 0), 'end': time(7, 0), 'hours': 8},
            'flexible': {'start': time(8, 0), 'end': time(18, 0), 'hours': 8}
        }
        
    def generate_attendance_data(self, days=30):
        """Generate sample attendance data"""
        np.random.seed(42)
        employees = [f'EMP{i:03d}' for i in range(1, 101)]
        
        attendance_records = []
        base_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            is_holiday = current_date.date() in self.us_holidays
            
            for emp_id in employees:
                # Skip weekends and holidays for most employees
                if is_weekend or is_holiday:
                    if np.random.random() > 0.1:  # 10% work on weekends/holidays
                        continue
                
                # Simulate attendance patterns
                attendance_prob = 0.95 if not (is_weekend or is_holiday) else 0.1
                
                if np.random.random() < attendance_prob:
                    # Generate clock-in/out times
                    base_start = datetime.combine(current_date.date(), time(9, 0))
                    clock_in = base_start + timedelta(minutes=np.random.normal(0, 15))
                    
                    # Work duration with some variation
                    work_hours = np.random.normal(8, 0.5)
                    clock_out = clock_in + timedelta(hours=work_hours)
                    
                    # Calculate overtime
                    regular_hours = min(8, work_hours)
                    overtime_hours = max(0, work_hours - 8)
                    
                    attendance_records.append({
                        'employee_id': emp_id,
                        'date': current_date.date(),
                        'clock_in': clock_in.time(),
                        'clock_out': clock_out.time(),
                        'total_hours': work_hours,
                        'regular_hours': regular_hours,
                        'overtime_hours': overtime_hours,
                        'status': 'Present',
                        'location': np.random.choice(['Office', 'Remote', 'Field'], p=[0.6, 0.3, 0.1])
                    })
                else:
                    # Absent record
                    absence_type = np.random.choice(['Sick', 'Personal', 'Vacation', 'Unexcused'], 
                                                  p=[0.4, 0.2, 0.3, 0.1])
                    attendance_records.append({
                        'employee_id': emp_id,
                        'date': current_date.date(),
                        'clock_in': None,
                        'clock_out': None,
                        'total_hours': 0,
                        'regular_hours': 0,
                        'overtime_hours': 0,
                        'status': absence_type,
                        'location': None
                    })
        
        return pd.DataFrame(attendance_records)
    
    def predict_absenteeism(self, attendance_df, employee_df=None):
        """Predict absenteeism using XGBoost"""
        if len(attendance_df) < 100:
            return []
        
        # Prepare features
        features_df = attendance_df.copy()
        features_df['is_absent'] = (features_df['status'] != 'Present').astype(int)
        features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
        features_df['month'] = pd.to_datetime(features_df['date']).dt.month
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
        
        # Employee-level features
        emp_stats = features_df.groupby('employee_id').agg({
            'is_absent': ['mean', 'sum'],
            'total_hours': 'mean',
            'overtime_hours': 'mean'
        }).reset_index()
        
        emp_stats.columns = ['employee_id', 'absence_rate', 'total_absences', 'avg_hours', 'avg_overtime']
        features_df = features_df.merge(emp_stats, on='employee_id')
        
        # Select features for prediction
        feature_cols = ['day_of_week', 'month', 'is_monday', 'is_friday', 
                       'absence_rate', 'avg_hours', 'avg_overtime']
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_absent']
        
        # Train XGBoost model
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        model.fit(X, y)
        
        # Predict for next week
        predictions = []
        next_week = datetime.now() + timedelta(days=1)
        
        for emp_id in features_df['employee_id'].unique()[:10]:  # Top 10 at-risk employees
            emp_data = features_df[features_df['employee_id'] == emp_id].iloc[-1]
            
            for day in range(7):
                future_date = next_week + timedelta(days=day)
                
                # Create feature vector for prediction
                pred_features = [
                    future_date.weekday(),  # day_of_week
                    future_date.month,      # month
                    1 if future_date.weekday() == 0 else 0,  # is_monday
                    1 if future_date.weekday() == 4 else 0,  # is_friday
                    emp_data['absence_rate'],
                    emp_data['avg_hours'],
                    emp_data['avg_overtime']
                ]
                
                prob = model.predict_proba([pred_features])[0][1]
                
                if prob > 0.3:  # High risk threshold
                    predictions.append({
                        'employee_id': emp_id,
                        'date': future_date.date(),
                        'absence_probability': prob,
                        'risk_level': 'High' if prob > 0.5 else 'Medium'
                    })
        
        return predictions
    
    def detect_attendance_anomalies(self, attendance_df):
        """Detect anomalies in attendance patterns"""
        if len(attendance_df) < 50:
            return []
        
        # Prepare features for anomaly detection
        features = attendance_df[attendance_df['status'] == 'Present'].copy()
        
        if len(features) < 10:
            return []
        
        # Convert time to minutes for analysis
        features['clock_in_minutes'] = features['clock_in'].apply(
            lambda x: x.hour * 60 + x.minute if pd.notna(x) else 540  # Default 9 AM
        )
        features['clock_out_minutes'] = features['clock_out'].apply(
            lambda x: x.hour * 60 + x.minute if pd.notna(x) else 1020  # Default 5 PM
        )
        
        # Features for anomaly detection
        anomaly_features = ['total_hours', 'clock_in_minutes', 'clock_out_minutes', 'overtime_hours']
        X = features[anomaly_features].fillna(features[anomaly_features].mean())
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Return anomalous records
        anomaly_indices = np.where(anomalies == -1)[0]
        return features.iloc[anomaly_indices].to_dict('records')

def render_time_attendance_dashboard():
    """Render time and attendance dashboard"""
    st.markdown('<div class="main-header">⏰ Time & Attendance</div>', unsafe_allow_html=True)
    
    time_manager = TimeAttendanceManager()
    
    # Time & Attendance Overview KPIs
    st.markdown("### 📊 Attendance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Present Today", "1,156", "+12")
    with col2:
        st.metric("Attendance Rate", "94.2%", "+1.3%")
    with col3:
        st.metric("Avg Hours/Day", "8.2", "+0.1")
    with col4:
        st.metric("Overtime Hours", "2,340", "-5.2%")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Clock In/Out", "📅 Shift Management", "🏖️ PTO Management", 
        "🔍 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        render_clock_interface(time_manager)
    
    with tab2:
        render_shift_management(time_manager)
    
    with tab3:
        render_pto_management(time_manager)
    
    with tab4:
        render_attendance_analytics(time_manager)
    
    with tab5:
        render_attendance_settings()

def render_clock_interface(time_manager):
    """Render clock in/out interface"""
    st.markdown("### 🕐 Employee Clock In/Out")
    
    # Employee selection for demo
    employee_id = st.selectbox("Select Employee", [f"EMP{i:03d}" for i in range(1, 21)])
    
    # Current status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Current Status**")
        current_time = datetime.now()
        st.write(f"**Time:** {current_time.strftime('%H:%M:%S')}")
        st.write(f"**Date:** {current_time.strftime('%Y-%m-%d')}")
        
        # Mock current status
        is_clocked_in = st.session_state.get(f'{employee_id}_clocked_in', False)
        clock_in_time = st.session_state.get(f'{employee_id}_clock_in_time', None)
        
        if is_clocked_in:
            st.success("✅ Clocked In")
            if clock_in_time:
                hours_worked = (current_time - clock_in_time).total_seconds() / 3600
                st.write(f"**Hours Worked:** {hours_worked:.1f}")
        else:
            st.info("⏰ Not Clocked In")
    
    with col2:
        st.markdown("**Clock Actions**")
        
        if not is_clocked_in:
            if st.button("🟢 Clock In", type="primary", use_container_width=True):
                st.session_state[f'{employee_id}_clocked_in'] = True
                st.session_state[f'{employee_id}_clock_in_time'] = current_time
                st.success(f"✅ {employee_id} clocked in at {current_time.strftime('%H:%M')}")
                st.rerun()
        else:
            if st.button("🔴 Clock Out", type="primary", use_container_width=True):
                st.session_state[f'{employee_id}_clocked_in'] = False
                if clock_in_time:
                    hours_worked = (current_time - clock_in_time).total_seconds() / 3600
                    st.success(f"✅ {employee_id} clocked out. Total hours: {hours_worked:.1f}")
                st.rerun()
        
        if st.button("☕ Break", use_container_width=True):
            st.info("Break started")
        
        if st.button("📍 Update Location", use_container_width=True):
            location = st.selectbox("Select Location", ["Office", "Remote", "Field"], key="location_update")
            st.success(f"Location updated to {location}")
    
    with col3:
        st.markdown("**Today's Summary**")
        st.write("**Clock In:** 09:15 AM")
        st.write("**Break Time:** 1.0 hours")
        st.write("**Hours Worked:** 7.5")
        st.write("**Status:** On Time")
    
    # Recent attendance
    st.markdown("#### 📋 Recent Attendance")
    
    # Generate sample recent attendance
    recent_data = []
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        recent_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Day': date.strftime('%A'),
            'Clock In': '09:15 AM',
            'Clock Out': '05:30 PM',
            'Hours': '8.25',
            'Status': '✅ Present',
            'Location': 'Office'
        })
    
    df = pd.DataFrame(recent_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_shift_management(time_manager):
    """Render shift management interface"""
    st.markdown("### 📅 Shift Management")
    
    # Shift overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Shifts", "247", "+5")
    with col2:
        st.metric("Coverage Rate", "98.5%", "+2.1%")
    with col3:
        st.metric("Overtime Shifts", "23", "-3")
    
    # Shift scheduling
    st.markdown("#### 🗓️ Shift Scheduling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Create New Shift**")
        shift_date = st.date_input("Shift Date")
        shift_type = st.selectbox("Shift Type", list(time_manager.shift_types.keys()))
        assigned_employee = st.selectbox("Assign Employee", [f"EMP{i:03d}" for i in range(1, 21)])
        
        shift_details = time_manager.shift_types[shift_type]
        st.write(f"**Start Time:** {shift_details['start']}")
        st.write(f"**End Time:** {shift_details['end']}")
        st.write(f"**Duration:** {shift_details['hours']} hours")
        
        if st.button("➕ Create Shift", type="primary"):
            st.success(f"✅ Shift created for {assigned_employee} on {shift_date}")
    
    with col2:
        st.markdown("**Shift Calendar**")
        
        # Weekly shift view
        week_start = datetime.now() - timedelta(days=datetime.now().weekday())
        
        shift_schedule = []
        for day in range(7):
            current_day = week_start + timedelta(days=day)
            shift_schedule.append({
                'Day': current_day.strftime('%A'),
                'Date': current_day.strftime('%m/%d'),
                'Morning': 'EMP001, EMP002',
                'Evening': 'EMP003, EMP004',
                'Night': 'EMP005',
                'Coverage': '100%'
            })
        
        st.dataframe(pd.DataFrame(shift_schedule), use_container_width=True, hide_index=True)
    
    # Shift analytics
    st.markdown("#### 📊 Shift Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Shift distribution
        shift_types = ['Morning', 'Evening', 'Night', 'Flexible']
        shift_counts = [120, 80, 30, 17]
        
        fig1 = px.pie(values=shift_counts, names=shift_types, 
                     title="Shift Distribution")
        fig1.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Coverage by day
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        coverage = [98, 99, 97, 98, 96, 85, 80]
        
        fig2 = px.bar(x=days, y=coverage, title="Coverage Rate by Day (%)")
        fig2.update_traces(marker_color='#1f77b4')
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

def render_pto_management(time_manager):
    """Render PTO and leave management"""
    st.markdown("### 🏖️ PTO & Leave Management")
    
    # PTO overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Pending Requests", "23", "+5")
    with col2:
        st.metric("Approved Today", "8", "+2")
    with col3:
        st.metric("Avg PTO Balance", "18.5 days", "-1.2")
    with col4:
        st.metric("Coverage Issues", "2", "-1")
    
    # PTO request form
    st.markdown("#### 📝 Submit PTO Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_id = st.selectbox("Employee", [f"EMP{i:03d}" for i in range(1, 21)], key="pto_employee")
        leave_type = st.selectbox("Leave Type", ["Vacation", "Sick", "Personal", "Bereavement", "Jury Duty"])
        start_date = st.date_input("Start Date", key="pto_start")
        end_date = st.date_input("End Date", key="pto_end")
        
        if start_date and end_date:
            days_requested = (end_date - start_date).days + 1
            st.write(f"**Days Requested:** {days_requested}")
    
    with col2:
        reason = st.text_area("Reason (Optional)", height=100)
        manager_approval = st.selectbox("Requires Manager Approval", ["Yes", "No"])
        
        if st.button("📤 Submit Request", type="primary"):
            st.success(f"✅ PTO request submitted for {employee_id}")
            st.info("Request sent to manager for approval")
    
    # Pending requests
    st.markdown("#### ⏳ Pending Requests")
    
    pending_requests = [
        {"Employee": "EMP001", "Type": "Vacation", "Dates": "2024-03-15 to 2024-03-20", "Days": 6, "Status": "Pending", "Manager": "MGR001"},
        {"Employee": "EMP002", "Type": "Sick", "Dates": "2024-03-18", "Days": 1, "Status": "Pending", "Manager": "MGR002"},
        {"Employee": "EMP003", "Type": "Personal", "Dates": "2024-03-22 to 2024-03-23", "Days": 2, "Status": "Pending", "Manager": "MGR001"},
    ]
    
    for i, request in enumerate(pending_requests):
        with st.expander(f"📋 {request['Employee']} - {request['Type']} ({request['Days']} days)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Employee:** {request['Employee']}")
                st.write(f"**Type:** {request['Type']}")
                st.write(f"**Dates:** {request['Dates']}")
            
            with col2:
                st.write(f"**Days:** {request['Days']}")
                st.write(f"**Manager:** {request['Manager']}")
                st.write(f"**Status:** {request['Status']}")
            
            with col3:
                col_approve, col_deny = st.columns(2)
                with col_approve:
                    if st.button("✅ Approve", key=f"approve_pto_{i}"):
                        st.success("Request approved!")
                with col_deny:
                    if st.button("❌ Deny", key=f"deny_pto_{i}"):
                        st.error("Request denied!")
    
    # PTO calendar
    st.markdown("#### 📅 PTO Calendar")
    
    # Generate sample PTO calendar data
    calendar_data = []
    for i in range(30):
        date = datetime.now() + timedelta(days=i)
        if np.random.random() < 0.1:  # 10% chance of PTO
            calendar_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Employee': f"EMP{np.random.randint(1, 21):03d}",
                'Type': np.random.choice(['Vacation', 'Sick', 'Personal']),
                'Status': np.random.choice(['Approved', 'Pending'], p=[0.8, 0.2])
            })
    
    if calendar_data:
        pto_df = pd.DataFrame(calendar_data)
        st.dataframe(pto_df, use_container_width=True, hide_index=True)

def render_attendance_analytics(time_manager):
    """Render attendance analytics and predictions"""
    st.markdown("### 🔍 Attendance Analytics")
    
    # Generate sample data for analytics
    attendance_df = time_manager.generate_attendance_data(30)
    
    # Analytics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Attendance Rate", "94.2%", "+1.3%")
    with col2:
        st.metric("Avg Daily Hours", "8.1", "+0.1")
    with col3:
        st.metric("Overtime Rate", "15.2%", "-2.1%")
    
    # Attendance trends
    st.markdown("#### 📈 Attendance Trends")
    
    # Daily attendance over time
    daily_stats = attendance_df.groupby('date').agg({
        'employee_id': 'count',
        'total_hours': 'mean',
        'overtime_hours': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'present_count', 'avg_hours', 'total_overtime']
    daily_stats['attendance_rate'] = (daily_stats['present_count'] / 100) * 100  # Assuming 100 employees
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(daily_stats, x='date', y='attendance_rate', 
                      title="Daily Attendance Rate (%)")
        fig1.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(daily_stats, x='date', y='total_overtime', 
                     title="Daily Overtime Hours")
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
    
    # AI Predictions
    st.markdown("#### 🤖 AI-Powered Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Absenteeism Prediction**")
        
        if st.button("🔮 Predict Absenteeism", type="primary"):
            with st.spinner("Analyzing attendance patterns..."):
                predictions = time_manager.predict_absenteeism(attendance_df)
                
                if predictions:
                    st.success(f"✅ Identified {len(predictions)} high-risk employees")
                    
                    pred_df = pd.DataFrame(predictions)
                    pred_df['absence_probability'] = pred_df['absence_probability'].round(3)
                    
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No high-risk employees identified")
    
    with col2:
        st.markdown("**Attendance Anomalies**")
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies..."):
                anomalies = time_manager.detect_attendance_anomalies(attendance_df)
                
                if anomalies:
                    st.warning(f"⚠️ Found {len(anomalies)} attendance anomalies")
                    
                    for i, anomaly in enumerate(anomalies[:5]):  # Show top 5
                        st.write(f"**{anomaly['employee_id']}** - {anomaly['date']}: {anomaly['total_hours']:.1f} hours")
                else:
                    st.success("✅ No anomalies detected")
    
    # Department analysis
    st.markdown("#### 🏢 Department Analysis")
    
    # Simulate department data
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
    dept_stats = []
    
    for dept in departments:
        dept_stats.append({
            'Department': dept,
            'Employees': np.random.randint(15, 35),
            'Attendance Rate': np.random.uniform(90, 98),
            'Avg Hours': np.random.uniform(7.8, 8.5),
            'Overtime Hours': np.random.randint(50, 200)
        })
    
    dept_df = pd.DataFrame(dept_stats)
    dept_df['Attendance Rate'] = dept_df['Attendance Rate'].round(1)
    dept_df['Avg Hours'] = dept_df['Avg Hours'].round(1)
    
    st.dataframe(dept_df, use_container_width=True, hide_index=True)

def render_attendance_settings():
    """Render attendance system settings"""
    st.markdown("### ⚙️ Attendance Settings")
    
    # Work schedule settings
    st.markdown("#### 🕐 Work Schedule Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standard Work Hours**")
        work_start = st.time_input("Work Start Time", value=time(9, 0))
        work_end = st.time_input("Work End Time", value=time(17, 0))
        break_duration = st.number_input("Break Duration (minutes)", value=60, min_value=0, max_value=120)
        
        st.markdown("**Overtime Rules**")
        overtime_threshold = st.number_input("Daily Overtime Threshold (hours)", value=8.0, min_value=6.0, max_value=12.0)
        overtime_multiplier = st.number_input("Overtime Pay Multiplier", value=1.5, min_value=1.0, max_value=3.0)
    
    with col2:
        st.markdown("**Attendance Policies**")
        grace_period = st.number_input("Late Arrival Grace Period (minutes)", value=15, min_value=0, max_value=60)
        max_daily_hours = st.number_input("Maximum Daily Hours", value=12.0, min_value=8.0, max_value=16.0)
        
        st.markdown("**PTO Policies**")
        annual_pto_days = st.number_input("Annual PTO Days", value=20, min_value=10, max_value=40)
        sick_days = st.number_input("Annual Sick Days", value=10, min_value=5, max_value=20)
        advance_notice_days = st.number_input("PTO Advance Notice (days)", value=14, min_value=1, max_value=30)
    
    # Notification settings
    st.markdown("#### 🔔 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        late_notifications = st.checkbox("Late Arrival Notifications", value=True)
        overtime_alerts = st.checkbox("Overtime Alerts", value=True)
        absence_notifications = st.checkbox("Absence Notifications", value=True)
    
    with col2:
        manager_notifications = st.checkbox("Manager Notifications", value=True)
        pto_reminders = st.checkbox("PTO Balance Reminders", value=True)
        schedule_changes = st.checkbox("Schedule Change Alerts", value=True)
    
    # Integration settings
    st.markdown("#### 🔗 Integration Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Biometric Integration**")
        biometric_enabled = st.checkbox("Enable Biometric Clock-in", value=False)
        fingerprint_scanner = st.checkbox("Fingerprint Scanner", value=False)
        facial_recognition = st.checkbox("Facial Recognition", value=False)
    
    with col2:
        st.markdown("**Mobile App Settings**")
        mobile_clockin = st.checkbox("Mobile Clock-in", value=True)
        gps_tracking = st.checkbox("GPS Location Tracking", value=True)
        offline_mode = st.checkbox("Offline Mode Support", value=True)
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Attendance settings saved successfully!")

if __name__ == "__main__":
    render_time_attendance_dashboard()
