"""
Ollama Integration for HR Analytics Pro
Provides AI-powered insights, recommendations, and chatbot functionality
"""

import requests
import json
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                return True
        except Exception as e:
            st.sidebar.warning(f"⚠️ Ollama not available: {str(e)[:50]}")
        return False
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        return len(self.available_models) > 0
    
    def get_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models
    
    def generate_response(self, prompt: str, model: str = "llama2", stream: bool = False) -> str:
        """Generate response from Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class HRAssistant:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
        self.context = ""
    
    def set_context(self, employees_df: pd.DataFrame, performance_df: pd.DataFrame):
        """Set HR data context for the assistant"""
        emp_summary = f"""
        HR Data Summary:
        - Total Employees: {len(employees_df)}
        - Departments: {', '.join(employees_df['department'].unique()) if 'department' in employees_df else 'N/A'}
        - Average Salary: ${employees_df['salary'].mean():,.0f} if 'salary' in employees_df else 'N/A'
        - Age Range: {employees_df['age'].min()}-{employees_df['age'].max()} if 'age' in employees_df else 'N/A'
        - Performance Records: {len(performance_df)}
        """
        self.context = emp_summary
    
    def generate_insights(self, data_summary: str, model: str = "llama2") -> str:
        """Generate AI insights about HR data"""
        prompt = f"""
        As an HR Analytics expert, analyze this employee data and provide 3-5 key insights:
        
        {data_summary}
        
        Focus on:
        1. Performance trends
        2. Potential areas of concern
        3. Recommendations for improvement
        4. Workforce optimization opportunities
        
        Keep insights concise and actionable.
        """
        
        return self.ollama.generate_response(prompt, model)
    
    def generate_recommendations(self, employee_data: Dict, model: str = "llama2") -> str:
        """Generate personalized recommendations for an employee"""
        prompt = f"""
        As an HR consultant, provide personalized recommendations for this employee:
        
        Employee Profile:
        - Name: {employee_data.get('name', 'Unknown')}
        - Department: {employee_data.get('department', 'Unknown')}
        - Position: {employee_data.get('position', 'Unknown')}
        - Performance Score: {employee_data.get('performance_score', 'N/A')}
        - Tenure: {employee_data.get('tenure', 'Unknown')} years
        - Age: {employee_data.get('age', 'Unknown')}
        
        Provide 3-4 specific, actionable recommendations for:
        1. Career development
        2. Skill improvement
        3. Performance enhancement
        4. Growth opportunities
        
        Keep recommendations professional and constructive.
        """
        
        return self.ollama.generate_response(prompt, model)
    
    def chat_response(self, user_message: str, model: str = "llama2") -> str:
        """Generate chatbot response for HR-related queries"""
        prompt = f"""
        You are an HR Analytics Assistant. Answer this question about HR and workforce management:
        
        Context: {self.context}
        
        User Question: {user_message}
        
        Provide a helpful, professional response. If the question is about specific data, 
        refer to the context provided. Keep responses concise and actionable.
        """
        
        return self.ollama.generate_response(prompt, model)

def create_ollama_client() -> Optional[OllamaClient]:
    """Create and return Ollama client if available"""
    try:
        client = OllamaClient()
        if client.is_available():
            return client
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Ollama: {str(e)}")
    return None

def show_ai_insights(employees_df: pd.DataFrame, performance_df: pd.DataFrame, ollama_client: OllamaClient):
    """Display AI-generated insights about HR data"""
    st.header("🤖 AI-Powered Insights")
    
    if not ollama_client.is_available():
        st.warning("⚠️ Ollama is not available. Please ensure Ollama is running.")
        return
    
    # Model selection
    models = ollama_client.get_models()
    if not models:
        st.error("No Ollama models available. Please install a model (e.g., `ollama pull llama2`)")
        return
    
    selected_model = st.selectbox("Select AI Model", models, index=0)
    
    # Generate insights button
    if st.button("🔍 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing HR data with AI..."):
            assistant = HRAssistant(ollama_client)
            assistant.set_context(employees_df, performance_df)
            
            # Create data summary
            data_summary = f"""
            Employee Data Analysis:
            - Total Employees: {len(employees_df)}
            - Departments: {employees_df['department'].nunique() if 'department' in employees_df else 0}
            - Average Age: {employees_df['age'].mean():.1f} if 'age' in employees_df else 'N/A'
            - Salary Range: ${employees_df['salary'].min():,.0f} - ${employees_df['salary'].max():,.0f} if 'salary' in employees_df else 'N/A'
            - Performance Records: {len(performance_df)}
            """
            
            insights = assistant.generate_insights(data_summary, selected_model)
            
            st.markdown("### 📊 AI Analysis Results")
            st.markdown(insights)
    
    # Performance prediction insights
    st.subheader("🎯 Performance Prediction Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze High Performers"):
            with st.spinner("Analyzing high performers..."):
                # Get high performers data
                if 'performance_score' in performance_df.columns:
                    high_perf = performance_df[performance_df['performance_score'] > 8]
                    summary = f"High performers analysis: {len(high_perf)} employees with scores > 8"
                else:
                    summary = "Performance data analysis for top performers"
                
                assistant = HRAssistant(ollama_client)
                insights = assistant.generate_insights(summary, selected_model)
                st.success("High Performer Analysis:")
                st.write(insights)
    
    with col2:
        if st.button("Identify Risk Areas"):
            with st.spinner("Identifying risk areas..."):
                # Analyze potential risks
                risk_summary = f"""
                Risk Analysis:
                - Departments with performance variations
                - Age distribution concerns
                - Salary equity analysis
                - Retention risk factors
                """
                
                assistant = HRAssistant(ollama_client)
                insights = assistant.generate_insights(risk_summary, selected_model)
                st.warning("Risk Area Analysis:")
                st.write(insights)

def show_ai_chatbot(employees_df: pd.DataFrame, performance_df: pd.DataFrame, ollama_client: OllamaClient):
    """Display AI chatbot interface"""
    st.header("💬 HR AI Assistant")
    
    if not ollama_client.is_available():
        st.warning("⚠️ Ollama is not available. Please ensure Ollama is running.")
        return
    
    # Model selection
    models = ollama_client.get_models()
    if not models:
        st.error("No Ollama models available. Please install a model (e.g., `ollama pull llama2`)")
        return
    
    selected_model = st.selectbox("Select AI Model", models, key="chatbot_model")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.subheader("Ask me anything about HR analytics, performance management, or workforce insights!")
    
    # Display chat history
    for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**AI Assistant:** {ai_msg}")
        st.markdown("---")
    
    # Chat input
    user_input = st.text_input("Your question:", placeholder="e.g., What are the key performance trends?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send", type="primary"):
            if user_input:
                with st.spinner("AI is thinking..."):
                    assistant = HRAssistant(ollama_client)
                    assistant.set_context(employees_df, performance_df)
                    
                    response = assistant.chat_response(user_input, selected_model)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_input, response))
                    st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Quick questions
    st.subheader("💡 Quick Questions")
    quick_questions = [
        "What are the main performance trends?",
        "How can we improve employee retention?",
        "What factors predict high performance?",
        "Are there any bias concerns in our data?",
        "What development opportunities should we focus on?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{question[:20]}"):
            with st.spinner("AI is analyzing..."):
                assistant = HRAssistant(ollama_client)
                assistant.set_context(employees_df, performance_df)
                
                response = assistant.chat_response(question, selected_model)
                st.session_state.chat_history.append((question, response))
                st.rerun()
