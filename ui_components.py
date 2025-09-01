"""
Modern UI/UX Components for HR Analytics Pro
Includes dark/light themes, glassmorphism effects, and responsive design
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

class UITheme:
    def __init__(self):
        self.current_theme = st.session_state.get('theme', 'light')
    
    def get_theme_css(self):
        """Get CSS for current theme with glassmorphism effects"""
        if self.current_theme == 'dark':
            return self._get_dark_theme_css()
        else:
            return self._get_light_theme_css()
    
    def _get_light_theme_css(self):
        return """
        <style>
        /* Light Theme with Glassmorphism */
        .main-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        }
        
        .kpi-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            color: white;
        }
        
        .kpi-card:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: scale(1.05);
        }
        
        .kpi-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .kpi-label {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.9);
            margin-top: 10px;
        }
        
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .badge-success {
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            color: white;
        }
        
        .badge-warning {
            background: linear-gradient(45deg, #FF9800, #FFC107);
            color: white;
        }
        
        .badge-danger {
            background: linear-gradient(45deg, #F44336, #E91E63);
            color: white;
        }
        
        .sidebar-nav {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
        }
        
        .nav-item {
            padding: 12px 20px;
            margin: 5px 0;
            border-radius: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
            color: white;
        }
        
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(10px);
        }
        
        .nav-item.active {
            background: rgba(255, 255, 255, 0.3);
            border-left: 4px solid #ffffff;
        }
        
        .quick-action-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .quick-action-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .notification-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid #ffffff;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        </style>
        """
    
    def _get_dark_theme_css(self):
        return """
        <style>
        /* Dark Theme with Glassmorphism */
        .main-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.7);
        }
        
        .kpi-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            color: white;
        }
        
        .kpi-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: scale(1.05);
        }
        
        .kpi-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .kpi-label {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 10px;
        }
        
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .badge-success {
            background: linear-gradient(45deg, #00C851, #007E33);
            color: white;
        }
        
        .badge-warning {
            background: linear-gradient(45deg, #ffbb33, #FF8800);
            color: white;
        }
        
        .badge-danger {
            background: linear-gradient(45deg, #ff4444, #CC0000);
            color: white;
        }
        </style>
        """

class UIComponents:
    def __init__(self):
        self.theme = UITheme()
    
    def render_kpi_card(self, title: str, value: str, delta: str = None, icon: str = "📊"):
        """Render a KPI card with glassmorphism effect"""
        delta_html = f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">{delta}</div>' if delta else ""
        
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 10px;">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{title}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def render_status_badge(self, text: str, status: str = "success"):
        """Render a status badge"""
        badge_class = f"badge-{status}"
        return f'<span class="status-badge {badge_class}">{text}</span>'
    
    def render_quick_actions(self, actions: list):
        """Render quick action buttons"""
        st.markdown("### 🚀 Quick Actions")
        cols = st.columns(len(actions))
        
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button(f"{action['icon']} {action['label']}", key=f"action_{i}", use_container_width=True):
                    if action.get('callback'):
                        action['callback']()
    
    def render_insights_feed(self, insights: list):
        """Render recent insights feed"""
        st.markdown("### 📈 Recent Insights")
        
        for insight in insights:
            priority_badge = self.render_status_badge(insight['priority'], 
                                                    'danger' if insight['priority'] == 'High' else 
                                                    'warning' if insight['priority'] == 'Medium' else 'success')
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: white;">{insight['title']}</h4>
                        <p style="margin: 5px 0; color: rgba(255,255,255,0.8);">{insight['description']}</p>
                        <small style="color: rgba(255,255,255,0.6);">{insight['timestamp']}</small>
                    </div>
                    <div>{priority_badge}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_system_status(self, status_items: list):
        """Render system status monitoring"""
        st.markdown("### 🔧 System Status")
        
        cols = st.columns(len(status_items))
        for i, item in enumerate(status_items):
            with cols[i]:
                status_color = "🟢" if item['status'] == 'healthy' else "🟡" if item['status'] == 'warning' else "🔴"
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 1.5rem;">{status_color}</div>
                    <div style="color: white; font-weight: bold;">{item['name']}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">{item['description']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_loading_spinner(self, text: str = "Loading..."):
        """Render loading spinner"""
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <div class="loading-spinner"></div>
            <p style="color: white; margin-top: 10px;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_toast_notification(self, message: str, type: str = "success"):
        """Show toast notification"""
        icon = "✅" if type == "success" else "⚠️" if type == "warning" else "❌"
        st.toast(f"{icon} {message}", icon=icon)
    
    def render_employee_card(self, employee: dict, performance_score: float = None):
        """Render employee card with performance visualization"""
        risk_level = "Low" if performance_score and performance_score > 0.8 else "Medium" if performance_score and performance_score > 0.6 else "High"
        risk_badge = self.render_status_badge(risk_level, 
                                            'success' if risk_level == 'Low' else 
                                            'warning' if risk_level == 'Medium' else 'danger')
        
        performance_bar = ""
        if performance_score:
            bar_width = int(performance_score * 100)
            performance_bar = f"""
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; margin: 10px 0;">
                <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; width: {bar_width}%; border-radius: 10px;"></div>
            </div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Performance: {performance_score:.1%}</div>
            """
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: white;">{employee.get('name', 'Unknown')}</h4>
                    <p style="margin: 5px 0; color: rgba(255,255,255,0.8);">{employee.get('department', 'Unknown')} • {employee.get('position', 'Unknown')}</p>
                    {performance_bar}
                </div>
                <div>{risk_badge}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        current_theme = st.session_state.get('theme', 'light')
        new_theme = 'dark' if current_theme == 'light' else 'light'
        st.session_state.theme = new_theme
        st.rerun()

# Global UI components instance
ui = UIComponents()
