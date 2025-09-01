"""
Explainable AI (XAI) Module for HR Performance Analytics
Implements SHAP, LIME, and custom explanation methods
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import shap
import lime
import lime.lime_tabular
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Comprehensive model explanation and interpretability"""
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = []
        self.explanation_cache = {}
        
    def initialize_explainers(self, model: BaseEstimator, X_train: pd.DataFrame):
        """Initialize SHAP and LIME explainers"""
        logger.info("Initializing explainers...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Initialize SHAP explainer
        try:
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                self.shap_explainer = shap.KernelExplainer(model.predict, X_train.sample(100))
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.error(f"Error initializing SHAP: {e}")
        
        # Initialize LIME explainer
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            logger.info("LIME explainer initialized")
        except Exception as e:
            logger.error(f"Error initializing LIME: {e}")
    
    def explain_prediction_shap(self, model: BaseEstimator, X_instance: pd.DataFrame,
                              max_display: int = 20) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction"""
        logger.info("Generating SHAP explanation...")
        
        try:
            if self.shap_explainer is None:
                logger.warning("SHAP explainer not initialized")
                return {}
            
            # Get SHAP values
            if hasattr(self.shap_explainer, 'shap_values'):
                shap_values = self.shap_explainer.shap_values(X_instance)
            else:
                shap_values = self.shap_explainer(X_instance)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            elif hasattr(shap_values, 'values'):
                shap_values = shap_values.values
            
            # Ensure we have the right shape
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Create explanation dictionary
            feature_contributions = dict(zip(self.feature_names, shap_values))
            
            # Sort by absolute importance
            sorted_features = sorted(feature_contributions.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            
            explanation = {
                'shap_values': shap_values.tolist(),
                'feature_contributions': feature_contributions,
                'top_features': dict(sorted_features[:max_display]),
                'base_value': getattr(self.shap_explainer, 'expected_value', 0),
                'prediction_breakdown': self._create_prediction_breakdown(
                    feature_contributions, X_instance.iloc[0]
                )
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {}
    
    def explain_prediction_lime(self, model: BaseEstimator, X_instance: pd.DataFrame,
                              num_features: int = 20) -> Dict[str, Any]:
        """Generate LIME explanation for a single prediction"""
        logger.info("Generating LIME explanation...")
        
        try:
            if self.lime_explainer is None:
                logger.warning("LIME explainer not initialized")
                return {}
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_instance.values[0],
                model.predict,
                num_features=num_features
            )
            
            # Extract feature importance
            feature_importance = dict(explanation.as_list())
            
            # Create explanation dictionary
            lime_explanation = {
                'feature_importance': feature_importance,
                'explanation_score': explanation.score,
                'local_prediction': explanation.local_pred[0] if explanation.local_pred else 0,
                'intercept': explanation.intercept[0] if explanation.intercept else 0,
                'top_features': dict(sorted(feature_importance.items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:num_features])
            }
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {}
    
    def _create_prediction_breakdown(self, feature_contributions: Dict[str, float],
                                   instance_values: pd.Series) -> List[Dict[str, Any]]:
        """Create detailed prediction breakdown"""
        breakdown = []
        
        # Sort features by absolute contribution
        sorted_features = sorted(feature_contributions.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        for feature, contribution in sorted_features[:15]:  # Top 15 features
            breakdown.append({
                'feature': feature,
                'value': instance_values.get(feature, 0),
                'contribution': contribution,
                'impact': 'positive' if contribution > 0 else 'negative',
                'magnitude': abs(contribution)
            })
        
        return breakdown
    
    def generate_global_explanations(self, model: BaseEstimator, 
                                   X_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate global model explanations"""
        logger.info("Generating global explanations...")
        
        global_explanations = {}
        
        # Feature importance from model (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            global_explanations['model_feature_importance'] = feature_importance
        
        # SHAP global explanations
        try:
            if self.shap_explainer:
                # Sample data for global explanation
                sample_data = X_data.sample(min(500, len(X_data)))
                shap_values = self.shap_explainer.shap_values(sample_data)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                elif hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                
                # Calculate mean absolute SHAP values
                mean_shap_values = np.abs(shap_values).mean(axis=0)
                shap_importance = dict(zip(self.feature_names, mean_shap_values))
                
                global_explanations['shap_feature_importance'] = shap_importance
                global_explanations['shap_summary_stats'] = {
                    'mean_prediction_impact': np.abs(shap_values).sum(axis=1).mean(),
                    'feature_interaction_strength': np.corrcoef(shap_values.T).mean()
                }
        except Exception as e:
            logger.error(f"Error generating global SHAP explanations: {e}")
        
        return global_explanations
    
    def create_explanation_visualization(self, explanation: Dict[str, Any],
                                       explanation_type: str = 'shap') -> go.Figure:
        """Create interactive explanation visualization"""
        
        if explanation_type == 'shap' and 'top_features' in explanation:
            features = list(explanation['top_features'].keys())
            values = list(explanation['top_features'].values())
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="SHAP Feature Importance",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=600
            )
            
        elif explanation_type == 'lime' and 'top_features' in explanation:
            features = list(explanation['top_features'].keys())
            values = list(explanation['top_features'].values())
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="LIME Feature Importance",
                xaxis_title="Feature Weight",
                yaxis_title="Features",
                height=600
            )
        
        return fig
    
    def explain_ensemble_prediction(self, ensemble_model, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """Explain ensemble model prediction with individual model contributions"""
        logger.info("Explaining ensemble prediction...")
        
        # Get individual model predictions
        individual_predictions = {}
        individual_explanations = {}
        
        for model_name, model in ensemble_model.models.items():
            if model_name != 'lstm':  # Skip LSTM for now
                try:
                    # Get prediction
                    pred = model.predict(X_instance)[0]
                    individual_predictions[model_name] = pred
                    
                    # Get explanation if possible
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                        individual_explanations[model_name] = feature_importance
                        
                except Exception as e:
                    logger.error(f"Error explaining {model_name}: {e}")
        
        # Calculate ensemble contribution
        ensemble_pred = ensemble_model.predict_single(X_instance)
        
        explanation = {
            'ensemble_prediction': ensemble_pred['ensemble_prediction'],
            'individual_predictions': individual_predictions,
            'model_weights': ensemble_model.ensemble_weights,
            'individual_explanations': individual_explanations,
            'confidence': ensemble_pred['confidence']
        }
        
        return explanation
    
    def generate_explanation_report(self, explanation: Dict[str, Any],
                                  employee_data: Dict[str, Any]) -> str:
        """Generate human-readable explanation report"""
        
        report = f"""
# Performance Prediction Explanation Report

## Employee Information
- **Employee ID**: {employee_data.get('employee_id', 'N/A')}
- **Name**: {employee_data.get('name', 'N/A')}
- **Department**: {employee_data.get('department', 'N/A')}
- **Position**: {employee_data.get('position', 'N/A')}

## Prediction Results
- **Predicted Performance**: {explanation.get('ensemble_prediction', 0):.2f}
- **Confidence Level**: {explanation.get('confidence', 0):.1f}%

## Key Contributing Factors
"""
        
        if 'top_features' in explanation:
            for i, (feature, contribution) in enumerate(list(explanation['top_features'].items())[:10]):
                impact = "↗️ Positive" if contribution > 0 else "↘️ Negative"
                report += f"{i+1}. **{feature}**: {contribution:.3f} ({impact})\n"
        
        report += """
## Model Explanation
This prediction is based on an ensemble of machine learning models that analyze:
- Historical performance trends
- Demographic and contextual factors
- Behavioral patterns and collaboration metrics
- Skill assessments and development indicators

## Recommendations
Based on the analysis, consider focusing on the factors with the highest positive impact to improve performance.
"""
        
        return report
    
    def feature_interaction_analysis(self, model: BaseEstimator, 
                                   X_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature interactions and dependencies"""
        logger.info("Analyzing feature interactions...")
        
        try:
            if self.shap_explainer:
                # Sample data for interaction analysis
                sample_data = X_data.sample(min(200, len(X_data)))
                shap_values = self.shap_explainer.shap_values(sample_data)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                elif hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                
                # Calculate feature interactions
                interaction_matrix = np.corrcoef(shap_values.T)
                
                # Find top interactions
                interactions = []
                n_features = len(self.feature_names)
                
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        correlation = interaction_matrix[i, j]
                        if abs(correlation) > 0.3:  # Significant interaction
                            interactions.append({
                                'feature1': self.feature_names[i],
                                'feature2': self.feature_names[j],
                                'correlation': correlation,
                                'interaction_strength': abs(correlation)
                            })
                
                # Sort by interaction strength
                interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                
                return {
                    'interaction_matrix': interaction_matrix.tolist(),
                    'top_interactions': interactions[:20],
                    'feature_names': self.feature_names
                }
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {e}")
        
        return {}
    
    def counterfactual_explanations(self, model: BaseEstimator, X_instance: pd.DataFrame,
                                  target_performance: float) -> Dict[str, Any]:
        """Generate counterfactual explanations"""
        logger.info("Generating counterfactual explanations...")
        
        try:
            original_prediction = model.predict(X_instance)[0]
            counterfactuals = []
            
            # Try modifying each feature to reach target performance
            for feature in self.feature_names[:20]:  # Limit to top 20 features
                if feature in X_instance.columns:
                    # Create modified instance
                    X_modified = X_instance.copy()
                    
                    # Try different modifications
                    original_value = X_instance[feature].iloc[0]
                    
                    for multiplier in [0.8, 0.9, 1.1, 1.2, 1.5]:
                        X_modified[feature] = original_value * multiplier
                        new_prediction = model.predict(X_modified)[0]
                        
                        # Check if we're closer to target
                        if abs(new_prediction - target_performance) < abs(original_prediction - target_performance):
                            counterfactuals.append({
                                'feature': feature,
                                'original_value': original_value,
                                'suggested_value': original_value * multiplier,
                                'change_required': f"{((multiplier - 1) * 100):+.1f}%",
                                'predicted_improvement': new_prediction - original_prediction,
                                'new_prediction': new_prediction
                            })
            
            # Sort by predicted improvement
            counterfactuals.sort(key=lambda x: abs(x['predicted_improvement']), reverse=True)
            
            return {
                'original_prediction': original_prediction,
                'target_performance': target_performance,
                'counterfactuals': counterfactuals[:10],  # Top 10 suggestions
                'feasible_improvements': [cf for cf in counterfactuals if abs(float(cf['change_required'].rstrip('%'))) <= 20]
            }
            
        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return {}
    
    def explain_prediction_lime(self, model: BaseEstimator, X_instance: pd.DataFrame,
                              num_features: int = 15) -> Dict[str, Any]:
        """Generate LIME explanation for a single prediction"""
        logger.info("Generating LIME explanation...")
        
        try:
            if self.lime_explainer is None:
                logger.warning("LIME explainer not initialized")
                return {}
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_instance.values[0],
                model.predict,
                num_features=num_features
            )
            
            # Extract explanation data
            feature_weights = dict(explanation.as_list())
            
            lime_explanation = {
                'feature_weights': feature_weights,
                'prediction_score': explanation.score,
                'local_prediction': explanation.local_pred[0] if explanation.local_pred else 0,
                'top_positive_features': {k: v for k, v in feature_weights.items() if v > 0},
                'top_negative_features': {k: v for k, v in feature_weights.items() if v < 0},
                'explanation_fidelity': explanation.score
            }
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {}
    
    def create_explanation_dashboard(self, shap_explanation: Dict[str, Any],
                                   lime_explanation: Dict[str, Any]) -> go.Figure:
        """Create comprehensive explanation dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['SHAP Feature Importance', 'LIME Feature Weights', 
                          'Prediction Breakdown', 'Feature Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # SHAP visualization
        if 'top_features' in shap_explanation:
            shap_features = list(shap_explanation['top_features'].keys())[:10]
            shap_values = list(shap_explanation['top_features'].values())[:10]
            
            fig.add_trace(
                go.Bar(x=shap_values, y=shap_features, orientation='h',
                      marker_color=['green' if v > 0 else 'red' for v in shap_values],
                      name='SHAP'),
                row=1, col=1
            )
        
        # LIME visualization
        if 'top_positive_features' in lime_explanation:
            lime_features = list(lime_explanation['feature_weights'].keys())[:10]
            lime_values = list(lime_explanation['feature_weights'].values())[:10]
            
            fig.add_trace(
                go.Bar(x=lime_values, y=lime_features, orientation='h',
                      marker_color=['blue' if v > 0 else 'orange' for v in lime_values],
                      name='LIME'),
                row=1, col=2
            )
        
        # Prediction breakdown
        if 'prediction_breakdown' in shap_explanation:
            breakdown = shap_explanation['prediction_breakdown'][:8]
            breakdown_features = [item['feature'] for item in breakdown]
            breakdown_contributions = [item['contribution'] for item in breakdown]
            
            fig.add_trace(
                go.Bar(x=breakdown_contributions, y=breakdown_features, orientation='h',
                      marker_color=['darkgreen' if v > 0 else 'darkred' for v in breakdown_contributions],
                      name='Breakdown'),
                row=2, col=1
            )
        
        # Feature comparison (SHAP vs LIME)
        if 'top_features' in shap_explanation and 'feature_weights' in lime_explanation:
            common_features = set(shap_explanation['top_features'].keys()) & set(lime_explanation['feature_weights'].keys())
            
            if common_features:
                shap_vals = [shap_explanation['top_features'][f] for f in common_features]
                lime_vals = [lime_explanation['feature_weights'][f] for f in common_features]
                
                fig.add_trace(
                    go.Scatter(x=shap_vals, y=lime_vals, mode='markers+text',
                              text=list(common_features), textposition='top center',
                              name='SHAP vs LIME'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Model Explanation Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def explain_model_decisions(self, model: BaseEstimator, X_instance: pd.DataFrame,
                              employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive explanation combining multiple methods"""
        logger.info("Generating comprehensive model explanation...")
        
        # Get both SHAP and LIME explanations
        shap_explanation = self.explain_prediction_shap(model, X_instance)
        lime_explanation = self.explain_prediction_lime(model, X_instance)
        
        # Generate counterfactuals
        current_pred = model.predict(X_instance)[0]
        target_performance = current_pred + 10  # Target 10 points higher
        counterfactuals = self.counterfactual_explanations(model, X_instance, target_performance)
        
        # Create comprehensive explanation
        comprehensive_explanation = {
            'employee_info': employee_data,
            'prediction': current_pred,
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation,
            'counterfactual_analysis': counterfactuals,
            'explanation_report': self.generate_explanation_report(
                {**shap_explanation, 'ensemble_prediction': current_pred}, employee_data
            ),
            'visualization': self.create_explanation_dashboard(shap_explanation, lime_explanation)
        }
        
        return comprehensive_explanation
    
    def cache_explanation(self, employee_id: str, explanation: Dict[str, Any]):
        """Cache explanation for future reference"""
        self.explanation_cache[employee_id] = {
            'explanation': explanation,
            'timestamp': pd.Timestamp.now()
        }
    
    def get_cached_explanation(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached explanation"""
        if employee_id in self.explanation_cache:
            cached = self.explanation_cache[employee_id]
            # Check if cache is still fresh (within 24 hours)
            if (pd.Timestamp.now() - cached['timestamp']).total_seconds() < 86400:
                return cached['explanation']
        return None
