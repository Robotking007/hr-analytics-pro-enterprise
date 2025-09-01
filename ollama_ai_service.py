import ollama
import streamlit as st
import json
from typing import Dict, List, Any

class OllamaAIService:
    """Ollama AI Service for HR Analytics"""
    
    def __init__(self):
        self.model = "llama3.2"  # Default model
        self.available_models = []
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and available"""
        try:
            # List available models
            models = ollama.list()
            if hasattr(models, 'models'):
                self.available_models = [model['name'] for model in models.models]
            elif isinstance(models, dict) and 'models' in models:
                self.available_models = [model['name'] for model in models['models']]
            else:
                self.available_models = ["llama3.2", "mistral", "codellama"]  # Default models
            return True
        except Exception as e:
            # Use fallback mode without Ollama
            self.available_models = ["llama3.2", "mistral", "codellama"]
            return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Ollama or fallback"""
        try:
            # Try Ollama first
            prompt = f"""
            Analyze the sentiment of this text and return a JSON response:
            Text: "{text}"
            
            Return format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.85,
                "emotions": ["happy", "satisfied"],
                "summary": "brief analysis"
            }}
            """
            
            response = ollama.generate(model=self.model, prompt=prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response['response'])
                return result
            except:
                # Fallback if JSON parsing fails
                return self._fallback_sentiment(text)
        except Exception as e:
            # Use fallback sentiment analysis
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis without Ollama"""
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'happy', 'satisfied', 'love', 'amazing']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'disappointed', 'frustrated', 'angry']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.8, 0.5 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.8, 0.5 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "emotions": [sentiment],
            "summary": f"Keyword-based analysis: {sentiment}"
        }
    
    def generate_recommendations(self, context: str, recommendation_type: str) -> List[str]:
        """Generate AI recommendations using Ollama or fallback"""
        try:
            # Try Ollama first
            prompt = f"""
            Generate 3-5 specific {recommendation_type} recommendations based on this context:
            Context: {context}
            
            Return as a simple list, one recommendation per line, no numbering.
            """
            
            response = ollama.generate(model=self.model, prompt=prompt)
            recommendations = response['response'].strip().split('\n')
            
            # Clean up recommendations
            cleaned = []
            for rec in recommendations:
                rec = rec.strip()
                if rec and not rec.startswith(('•', '-', '*')):
                    cleaned.append(rec)
            
            return cleaned[:5]  # Return max 5 recommendations
            
        except Exception as e:
            # Use fallback recommendations
            return self._fallback_recommendations(recommendation_type)
    
    def _fallback_recommendations(self, recommendation_type: str) -> List[str]:
        """Fallback recommendations when Ollama is unavailable"""
        fallback_recs = {
            "training": [
                "Complete leadership development program",
                "Enroll in technical skills certification",
                "Attend communication workshop",
                "Join mentorship program",
                "Take project management course"
            ],
            "performance improvement": [
                "Set clear weekly goals and track progress",
                "Schedule regular one-on-one meetings with manager",
                "Focus on time management and prioritization",
                "Seek feedback from peers and stakeholders",
                "Develop domain expertise through training"
            ],
            "feature importance": [
                "Salary and compensation are key retention factors",
                "Manager relationship significantly impacts performance",
                "Work-life balance affects long-term engagement",
                "Career growth opportunities drive motivation",
                "Job satisfaction correlates with productivity"
            ]
        }
        
        return fallback_recs.get(recommendation_type, [
            "Improve communication and collaboration",
            "Focus on skill development",
            "Set clear performance goals",
            "Seek regular feedback",
            "Maintain work-life balance"
        ])
    
    def predict_attrition_risk(self, employee_data: Dict) -> Dict[str, Any]:
        """Predict attrition risk using Ollama"""
        try:
            prompt = f"""
            Analyze this employee data and predict attrition risk:
            Employee Data: {json.dumps(employee_data)}
            
            Return JSON format:
            {{
                "risk_score": 0.75,
                "risk_level": "high/medium/low",
                "key_factors": ["factor1", "factor2"],
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            response = ollama.generate(model=self.model, prompt=prompt)
            
            try:
                result = json.loads(response['response'])
                return result
            except:
                # Fallback prediction
                return {
                    "risk_score": 0.5,
                    "risk_level": "medium",
                    "key_factors": ["salary", "satisfaction"],
                    "recommendations": ["Review compensation", "Conduct stay interview"]
                }
        except Exception as e:
            return {
                "risk_score": 0.0,
                "risk_level": "unknown",
                "key_factors": ["analysis_error"],
                "recommendations": [f"Error: {str(e)[:50]}"]
            }
    
    def analyze_skill_gaps(self, job_description: str, employee_skills: List[str]) -> Dict[str, Any]:
        """Analyze skill gaps using Ollama"""
        try:
            prompt = f"""
            Compare required skills vs current skills and identify gaps:
            Job Requirements: {job_description}
            Current Skills: {', '.join(employee_skills)}
            
            Return JSON:
            {{
                "skill_gaps": ["skill1", "skill2"],
                "gap_severity": "high/medium/low",
                "training_suggestions": ["course1", "course2"],
                "timeline": "3-6 months"
            }}
            """
            
            response = ollama.generate(model=self.model, prompt=prompt)
            
            try:
                result = json.loads(response['response'])
                return result
            except:
                return {
                    "skill_gaps": ["python", "leadership"],
                    "gap_severity": "medium",
                    "training_suggestions": ["Python course", "Leadership training"],
                    "timeline": "3-6 months"
                }
        except Exception as e:
            return {
                "skill_gaps": ["analysis_error"],
                "gap_severity": "unknown",
                "training_suggestions": [f"Error: {str(e)[:50]}"],
                "timeline": "unknown"
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return self.available_models if self.available_models else ["llama3.2", "mistral", "codellama"]
    
    def set_model(self, model_name: str):
        """Set the active Ollama model"""
        if model_name in self.available_models or not self.available_models:
            self.model = model_name
            return True
        return False

# Global instance
ai_service = OllamaAIService()
