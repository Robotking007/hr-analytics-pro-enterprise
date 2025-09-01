"""
Comprehensive Bias Detection and Fairness Monitoring System
Implements industry-standard fairness metrics and bias detection algorithms
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FairnessMonitor:
    """Comprehensive fairness and bias monitoring system"""
    
    def __init__(self, fairness_threshold: float = 0.8):
        self.fairness_threshold = fairness_threshold
        self.protected_attributes = ['gender', 'ethnicity', 'age_group']
        self.bias_history = []
        
    def calculate_demographic_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sensitive_features: pd.Series) -> Dict[str, float]:
        """Calculate demographic parity metrics"""
        try:
            # Convert to binary classification for fairness metrics
            y_pred_binary = (y_pred > y_pred.mean()).astype(int)
            y_true_binary = (y_true > y_true.mean()).astype(int)
            
            dp_diff = demographic_parity_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )
            dp_ratio = demographic_parity_ratio(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )
            
            return {
                'demographic_parity_difference': dp_diff,
                'demographic_parity_ratio': dp_ratio,
                'passes_threshold': dp_ratio >= self.fairness_threshold
            }
        except Exception as e:
            logger.error(f"Error calculating demographic parity: {e}")
            return {'demographic_parity_difference': 0, 'demographic_parity_ratio': 1, 'passes_threshold': True}
    
    def calculate_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                               sensitive_features: pd.Series) -> Dict[str, float]:
        """Calculate equalized odds metrics"""
        try:
            y_pred_binary = (y_pred > y_pred.mean()).astype(int)
            y_true_binary = (y_true > y_true.mean()).astype(int)
            
            eo_diff = equalized_odds_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )
            eo_ratio = equalized_odds_ratio(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )
            
            return {
                'equalized_odds_difference': eo_diff,
                'equalized_odds_ratio': eo_ratio,
                'passes_threshold': eo_ratio >= self.fairness_threshold
            }
        except Exception as e:
            logger.error(f"Error calculating equalized odds: {e}")
            return {'equalized_odds_difference': 0, 'equalized_odds_ratio': 1, 'passes_threshold': True}
    
    def calculate_disparate_impact(self, y_pred: np.ndarray, 
                                 sensitive_features: pd.Series) -> Dict[str, float]:
        """Calculate disparate impact ratio"""
        try:
            y_pred_binary = (y_pred > y_pred.mean()).astype(int)
            
            # Calculate selection rates for each group
            groups = sensitive_features.unique()
            selection_rates = {}
            
            for group in groups:
                group_mask = sensitive_features == group
                if group_mask.sum() > 0:
                    selection_rates[group] = y_pred_binary[group_mask].mean()
            
            if len(selection_rates) < 2:
                return {'disparate_impact_ratio': 1.0, 'passes_threshold': True}
            
            # Calculate disparate impact ratio (min/max selection rate)
            min_rate = min(selection_rates.values())
            max_rate = max(selection_rates.values())
            
            di_ratio = min_rate / max_rate if max_rate > 0 else 1.0
            
            return {
                'disparate_impact_ratio': di_ratio,
                'selection_rates': selection_rates,
                'passes_threshold': di_ratio >= self.fairness_threshold
            }
        except Exception as e:
            logger.error(f"Error calculating disparate impact: {e}")
            return {'disparate_impact_ratio': 1.0, 'passes_threshold': True}
    
    def calculate_statistical_parity(self, y_pred: np.ndarray,
                                   sensitive_features: pd.Series) -> Dict[str, float]:
        """Calculate statistical parity difference"""
        try:
            y_pred_binary = (y_pred > y_pred.mean()).astype(int)
            
            # Calculate positive prediction rates for each group
            groups = sensitive_features.unique()
            positive_rates = {}
            
            for group in groups:
                group_mask = sensitive_features == group
                if group_mask.sum() > 0:
                    positive_rates[group] = y_pred_binary[group_mask].mean()
            
            if len(positive_rates) < 2:
                return {'statistical_parity_difference': 0.0, 'passes_threshold': True}
            
            # Calculate difference between max and min rates
            sp_diff = max(positive_rates.values()) - min(positive_rates.values())
            
            return {
                'statistical_parity_difference': sp_diff,
                'positive_rates': positive_rates,
                'passes_threshold': sp_diff <= (1 - self.fairness_threshold)
            }
        except Exception as e:
            logger.error(f"Error calculating statistical parity: {e}")
            return {'statistical_parity_difference': 0.0, 'passes_threshold': True}
    
    def comprehensive_bias_audit(self, df: pd.DataFrame, y_true: np.ndarray, 
                               y_pred: np.ndarray, model_version: str) -> Dict[str, Any]:
        """Perform comprehensive bias audit across all protected attributes"""
        logger.info("Performing comprehensive bias audit...")
        
        audit_results = {
            'model_version': model_version,
            'audit_timestamp': pd.Timestamp.now(),
            'overall_fairness': True,
            'protected_attributes': {}
        }
        
        for attr in self.protected_attributes:
            if attr in df.columns:
                logger.info(f"Auditing bias for {attr}...")
                
                # Calculate all fairness metrics
                dp_metrics = self.calculate_demographic_parity(y_true, y_pred, df[attr])
                eo_metrics = self.calculate_equalized_odds(y_true, y_pred, df[attr])
                di_metrics = self.calculate_disparate_impact(y_pred, df[attr])
                sp_metrics = self.calculate_statistical_parity(y_pred, df[attr])
                
                attr_results = {
                    'demographic_parity': dp_metrics,
                    'equalized_odds': eo_metrics,
                    'disparate_impact': di_metrics,
                    'statistical_parity': sp_metrics,
                    'overall_fair': all([
                        dp_metrics['passes_threshold'],
                        eo_metrics['passes_threshold'],
                        di_metrics['passes_threshold'],
                        sp_metrics['passes_threshold']
                    ])
                }
                
                audit_results['protected_attributes'][attr] = attr_results
                
                if not attr_results['overall_fair']:
                    audit_results['overall_fairness'] = False
        
        # Store audit in history
        self.bias_history.append(audit_results)
        
        return audit_results
    
    def detect_model_drift(self, current_predictions: np.ndarray,
                          reference_predictions: np.ndarray,
                          sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias drift between model versions"""
        logger.info("Detecting model drift...")
        
        drift_results = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'affected_groups': []
        }
        
        for attr in self.protected_attributes:
            if attr in sensitive_features.columns:
                # Calculate fairness metrics for both predictions
                current_di = self.calculate_disparate_impact(current_predictions, sensitive_features[attr])
                reference_di = self.calculate_disparate_impact(reference_predictions, sensitive_features[attr])
                
                # Calculate drift magnitude
                drift_magnitude = abs(
                    current_di['disparate_impact_ratio'] - reference_di['disparate_impact_ratio']
                )
                
                if drift_magnitude > 0.1:  # 10% threshold
                    drift_results['drift_detected'] = True
                    drift_results['affected_groups'].append(attr)
                    drift_results['drift_magnitude'] = max(drift_results['drift_magnitude'], drift_magnitude)
        
        return drift_results
    
    def generate_fairness_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate human-readable fairness report"""
        report = f"""
# Fairness Audit Report
**Model Version**: {audit_results['model_version']}
**Audit Date**: {audit_results['audit_timestamp']}
**Overall Fairness**: {'✅ PASS' if audit_results['overall_fairness'] else '❌ FAIL'}

## Protected Attribute Analysis

"""
        
        for attr, results in audit_results['protected_attributes'].items():
            status = '✅ FAIR' if results['overall_fair'] else '❌ BIASED'
            report += f"""
### {attr.title()} - {status}

- **Demographic Parity Ratio**: {results['demographic_parity']['demographic_parity_ratio']:.3f}
- **Equalized Odds Ratio**: {results['equalized_odds']['equalized_odds_ratio']:.3f}  
- **Disparate Impact Ratio**: {results['disparate_impact']['disparate_impact_ratio']:.3f}
- **Statistical Parity Diff**: {results['statistical_parity']['statistical_parity_difference']:.3f}

"""
        
        return report
    
    def create_fairness_dashboard(self, audit_results: Dict[str, Any]) -> go.Figure:
        """Create interactive fairness dashboard"""
        
        # Prepare data for visualization
        attributes = list(audit_results['protected_attributes'].keys())
        dp_ratios = [audit_results['protected_attributes'][attr]['demographic_parity']['demographic_parity_ratio'] 
                    for attr in attributes]
        eo_ratios = [audit_results['protected_attributes'][attr]['equalized_odds']['equalized_odds_ratio']
                    for attr in attributes]
        di_ratios = [audit_results['protected_attributes'][attr]['disparate_impact']['disparate_impact_ratio']
                    for attr in attributes]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Demographic Parity', 'Equalized Odds', 'Disparate Impact', 'Fairness Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Demographic Parity
        fig.add_trace(
            go.Bar(x=attributes, y=dp_ratios, name='Demographic Parity',
                  marker_color=['green' if r >= self.fairness_threshold else 'red' for r in dp_ratios]),
            row=1, col=1
        )
        
        # Equalized Odds
        fig.add_trace(
            go.Bar(x=attributes, y=eo_ratios, name='Equalized Odds',
                  marker_color=['green' if r >= self.fairness_threshold else 'red' for r in eo_ratios]),
            row=1, col=2
        )
        
        # Disparate Impact
        fig.add_trace(
            go.Bar(x=attributes, y=di_ratios, name='Disparate Impact',
                  marker_color=['green' if r >= self.fairness_threshold else 'red' for r in di_ratios]),
            row=2, col=1
        )
        
        # Overall fairness indicator
        overall_score = np.mean(dp_ratios + eo_ratios + di_ratios)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Fairness Score"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.8], 'color': "lightgray"},
                        {'range': [0.8, 1], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.fairness_threshold
                    }
                }
            ),
            row=2, col=2
        )
        
        # Add fairness threshold line
        for row, col in [(1, 1), (1, 2), (2, 1)]:
            fig.add_hline(y=self.fairness_threshold, line_dash="dash", line_color="red",
                         annotation_text="Fairness Threshold", row=row, col=col)
        
        fig.update_layout(
            title="Fairness Monitoring Dashboard",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def monitor_real_time_fairness(self, predictions: np.ndarray,
                                 sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """Real-time fairness monitoring with alerts"""
        logger.info("Monitoring real-time fairness...")
        
        alerts = []
        fairness_scores = {}
        
        for attr in self.protected_attributes:
            if attr in sensitive_features.columns:
                # Calculate disparate impact for real-time monitoring
                di_metrics = self.calculate_disparate_impact(predictions, sensitive_features[attr])
                fairness_scores[attr] = di_metrics['disparate_impact_ratio']
                
                # Check for fairness violations
                if not di_metrics['passes_threshold']:
                    alerts.append({
                        'attribute': attr,
                        'violation_type': 'disparate_impact',
                        'severity': 'high' if di_metrics['disparate_impact_ratio'] < 0.6 else 'medium',
                        'ratio': di_metrics['disparate_impact_ratio'],
                        'timestamp': pd.Timestamp.now()
                    })
        
        return {
            'fairness_scores': fairness_scores,
            'alerts': alerts,
            'overall_fair': len(alerts) == 0
        }
    
    def bias_mitigation_recommendations(self, audit_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        for attr, results in audit_results['protected_attributes'].items():
            if not results['overall_fair']:
                # Demographic parity issues
                if not results['demographic_parity']['passes_threshold']:
                    recommendations.append({
                        'attribute': attr,
                        'issue': 'Demographic Parity Violation',
                        'recommendation': f'Consider rebalancing training data for {attr} groups or applying post-processing fairness constraints',
                        'priority': 'high'
                    })
                
                # Equalized odds issues
                if not results['equalized_odds']['passes_threshold']:
                    recommendations.append({
                        'attribute': attr,
                        'issue': 'Equalized Odds Violation',
                        'recommendation': f'Review feature selection and consider removing {attr}-correlated features',
                        'priority': 'high'
                    })
                
                # Disparate impact issues
                if not results['disparate_impact']['passes_threshold']:
                    recommendations.append({
                        'attribute': attr,
                        'issue': 'Disparate Impact',
                        'recommendation': f'Apply adversarial debiasing or fairness-aware machine learning techniques for {attr}',
                        'priority': 'critical'
                    })
        
        return recommendations
    
    def calculate_intersectional_bias(self, y_pred: np.ndarray,
                                    sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze intersectional bias across multiple protected attributes"""
        logger.info("Analyzing intersectional bias...")
        
        intersectional_results = {}
        
        # Create intersectional groups
        if len(self.protected_attributes) >= 2:
            available_attrs = [attr for attr in self.protected_attributes if attr in sensitive_features.columns]
            
            if len(available_attrs) >= 2:
                # Combine first two attributes for intersectional analysis
                attr1, attr2 = available_attrs[0], available_attrs[1]
                intersectional_groups = sensitive_features[attr1].astype(str) + "_" + sensitive_features[attr2].astype(str)
                
                # Calculate disparate impact for intersectional groups
                di_metrics = self.calculate_disparate_impact(y_pred, intersectional_groups)
                
                intersectional_results = {
                    'attributes_analyzed': [attr1, attr2],
                    'intersectional_groups': intersectional_groups.unique().tolist(),
                    'disparate_impact_ratio': di_metrics['disparate_impact_ratio'],
                    'selection_rates': di_metrics.get('selection_rates', {}),
                    'passes_threshold': di_metrics['passes_threshold']
                }
        
        return intersectional_results
    
    def generate_bias_alerts(self, audit_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automated bias alerts"""
        alerts = []
        
        for attr, results in audit_results['protected_attributes'].items():
            if not results['overall_fair']:
                # High severity alerts
                if results['disparate_impact']['disparate_impact_ratio'] < 0.6:
                    alerts.append({
                        'severity': 'critical',
                        'attribute': attr,
                        'message': f'Critical bias detected in {attr}: Disparate impact ratio {results["disparate_impact"]["disparate_impact_ratio"]:.3f}',
                        'action_required': 'Immediate model review and bias mitigation required',
                        'timestamp': pd.Timestamp.now()
                    })
                
                # Medium severity alerts
                elif results['demographic_parity']['demographic_parity_ratio'] < self.fairness_threshold:
                    alerts.append({
                        'severity': 'medium',
                        'attribute': attr,
                        'message': f'Demographic parity violation in {attr}: Ratio {results["demographic_parity"]["demographic_parity_ratio"]:.3f}',
                        'action_required': 'Review training data balance and feature selection',
                        'timestamp': pd.Timestamp.now()
                    })
        
        return alerts
    
    def create_bias_visualization(self, audit_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive bias visualization"""
        
        # Prepare data
        attributes = []
        dp_ratios = []
        eo_ratios = []
        di_ratios = []
        sp_diffs = []
        
        for attr, results in audit_results['protected_attributes'].items():
            attributes.append(attr)
            dp_ratios.append(results['demographic_parity']['demographic_parity_ratio'])
            eo_ratios.append(results['equalized_odds']['equalized_odds_ratio'])
            di_ratios.append(results['disparate_impact']['disparate_impact_ratio'])
            sp_diffs.append(results['statistical_parity']['statistical_parity_difference'])
        
        # Create radar chart
        fig = go.Figure()
        
        for i, attr in enumerate(attributes):
            fig.add_trace(go.Scatterpolar(
                r=[dp_ratios[i], eo_ratios[i], di_ratios[i], 1-sp_diffs[i]],
                theta=['Demographic Parity', 'Equalized Odds', 'Disparate Impact', 'Statistical Parity'],
                fill='toself',
                name=attr,
                line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            ))
        
        # Add fairness threshold line
        fig.add_trace(go.Scatterpolar(
            r=[self.fairness_threshold] * 4,
            theta=['Demographic Parity', 'Equalized Odds', 'Disparate Impact', 'Statistical Parity'],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Fairness Threshold'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Fairness Metrics Radar Chart",
            showlegend=True
        )
        
        return fig
    
    def export_audit_report(self, audit_results: Dict[str, Any], filepath: str):
        """Export detailed audit report to file"""
        report = self.generate_fairness_report(audit_results)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Fairness audit report exported to {filepath}")
    
    def get_bias_trends(self) -> pd.DataFrame:
        """Analyze bias trends over time"""
        if not self.bias_history:
            return pd.DataFrame()
        
        trends_data = []
        for audit in self.bias_history:
            for attr, results in audit['protected_attributes'].items():
                trends_data.append({
                    'timestamp': audit['audit_timestamp'],
                    'attribute': attr,
                    'demographic_parity_ratio': results['demographic_parity']['demographic_parity_ratio'],
                    'equalized_odds_ratio': results['equalized_odds']['equalized_odds_ratio'],
                    'disparate_impact_ratio': results['disparate_impact']['disparate_impact_ratio'],
                    'overall_fair': results['overall_fair']
                })
        
        return pd.DataFrame(trends_data)
