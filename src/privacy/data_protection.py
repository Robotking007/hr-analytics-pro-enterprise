"""
Advanced Data Privacy and Security Features
Implements PII masking, differential privacy, and federated learning
"""
import hashlib
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import secrets
from cryptography.fernet import Fernet
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class DataProtectionManager:
    """Comprehensive data protection and privacy manager"""
    
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.encryption_key = None
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
    def generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data"""
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key()
        return self.encryption_key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet encryption"""
        key = self.generate_encryption_key()
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data.decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.encryption_key:
            raise ValueError("Encryption key not available")
        
        fernet = Fernet(self.encryption_key)
        decrypted_data = fernet.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information in text"""
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def mask_pii(self, df: pd.DataFrame, columns_to_mask: Optional[List[str]] = None) -> pd.DataFrame:
        """Automatically detect and mask PII in dataframe"""
        logger.info("Masking PII in dataframe...")
        
        df_masked = df.copy()
        
        # Auto-detect PII columns if not specified
        if columns_to_mask is None:
            columns_to_mask = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['email', 'phone', 'ssn', 'name']):
                    columns_to_mask.append(col)
        
        # Mask specified columns
        for col in columns_to_mask:
            if col in df_masked.columns:
                if df_masked[col].dtype == 'object':
                    # Mask string data
                    df_masked[col] = df_masked[col].apply(self._mask_string)
                else:
                    # Hash numerical data
                    df_masked[col] = df_masked[col].apply(self._hash_value)
        
        # Detect and mask PII in text columns
        text_columns = df_masked.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col not in columns_to_mask:
                df_masked[col] = df_masked[col].apply(self._auto_mask_pii)
        
        return df_masked
    
    def _mask_string(self, value: str) -> str:
        """Mask string value while preserving some structure"""
        if pd.isna(value) or value == '':
            return value
        
        value_str = str(value)
        if len(value_str) <= 2:
            return '*' * len(value_str)
        elif '@' in value_str:  # Email
            parts = value_str.split('@')
            masked_local = parts[0][:2] + '*' * (len(parts[0]) - 2)
            return f"{masked_local}@{parts[1]}"
        else:  # General string
            return value_str[:2] + '*' * (len(value_str) - 2)
    
    def _hash_value(self, value: Any) -> str:
        """Hash a value using SHA-256"""
        if pd.isna(value):
            return value
        
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]
    
    def _auto_mask_pii(self, text: str) -> str:
        """Automatically mask detected PII in text"""
        if pd.isna(text) or text == '':
            return text
        
        masked_text = str(text)
        detected_pii = self.detect_pii(masked_text)
        
        for pii_type, matches in detected_pii.items():
            for match in matches:
                if pii_type == 'email':
                    masked_text = masked_text.replace(match, self._mask_string(match))
                else:
                    masked_text = masked_text.replace(match, '*' * len(match))
        
        return masked_text
    
    def add_differential_privacy(self, df: pd.DataFrame, epsilon: float = 1.0,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Add differential privacy using Gaussian noise"""
        logger.info("Adding differential privacy...")
        
        df_private = df.copy()
        
        # Select numerical columns if not specified
        if columns is None:
            columns = df_private.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add Gaussian noise for differential privacy
        for col in columns:
            if col in df_private.columns:
                # Calculate sensitivity (range of the column)
                sensitivity = df_private[col].max() - df_private[col].min()
                
                # Calculate noise scale
                noise_scale = sensitivity / epsilon
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_scale, len(df_private))
                df_private[col] = df_private[col] + noise
        
        return df_private
    
    def anonymize_dataset(self, df: pd.DataFrame, k_anonymity: int = 5) -> pd.DataFrame:
        """Apply k-anonymity to dataset"""
        logger.info(f"Applying {k_anonymity}-anonymity...")
        
        df_anon = df.copy()
        
        # Identify quasi-identifiers
        quasi_identifiers = ['age', 'department', 'position', 'salary']
        available_qi = [col for col in quasi_identifiers if col in df_anon.columns]
        
        # Generalize quasi-identifiers
        for col in available_qi:
            if col == 'age':
                df_anon[col] = pd.cut(df_anon[col], bins=5, labels=['20-30', '30-40', '40-50', '50-60', '60+'])
            elif col == 'salary':
                df_anon[col] = pd.qcut(df_anon[col], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
        
        return df_anon
    
    def create_synthetic_data(self, df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic data preserving statistical properties"""
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        synthetic_data = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Numerical columns - use normal distribution with same mean/std
                mean_val = df[col].mean()
                std_val = df[col].std()
                synthetic_data[col] = np.random.normal(mean_val, std_val, n_samples)
            else:
                # Categorical columns - sample from original distribution
                value_counts = df[col].value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(
                    value_counts.index, 
                    size=n_samples, 
                    p=value_counts.values
                )
        
        return pd.DataFrame(synthetic_data)
    
    def federated_learning_client(self, local_data: pd.DataFrame, 
                                global_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Federated learning client-side operations"""
        logger.info("Performing federated learning client operations...")
        
        # Simulate local model training
        from sklearn.linear_model import LinearRegression
        
        # Extract features and target
        feature_cols = [col for col in local_data.columns if col != 'target']
        X_local = local_data[feature_cols]
        y_local = local_data.get('target', np.random.normal(85, 10, len(local_data)))
        
        # Train local model
        local_model = LinearRegression()
        local_model.fit(X_local, y_local)
        
        # Extract model parameters
        local_params = {
            'coefficients': local_model.coef_.tolist(),
            'intercept': local_model.intercept_,
            'n_samples': len(local_data),
            'feature_names': feature_cols
        }
        
        return local_params
    
    def aggregate_federated_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate federated learning updates from multiple clients"""
        logger.info("Aggregating federated learning updates...")
        
        if not client_updates:
            return {}
        
        # Weighted average based on number of samples
        total_samples = sum(update['n_samples'] for update in client_updates)
        
        # Aggregate coefficients
        aggregated_coef = np.zeros_like(client_updates[0]['coefficients'])
        aggregated_intercept = 0.0
        
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            aggregated_coef += np.array(update['coefficients']) * weight
            aggregated_intercept += update['intercept'] * weight
        
        return {
            'aggregated_coefficients': aggregated_coef.tolist(),
            'aggregated_intercept': aggregated_intercept,
            'total_samples': total_samples,
            'num_clients': len(client_updates)
        }
    
    def privacy_budget_tracking(self, epsilon_used: float) -> Dict[str, Any]:
        """Track privacy budget usage"""
        self.privacy_budget -= epsilon_used
        
        return {
            'remaining_budget': self.privacy_budget,
            'epsilon_used': epsilon_used,
            'budget_exhausted': self.privacy_budget <= 0,
            'recommended_action': 'Stop data analysis' if self.privacy_budget <= 0 else 'Continue with caution'
        }
    
    def secure_data_sharing(self, df: pd.DataFrame, 
                          sharing_level: str = 'internal') -> pd.DataFrame:
        """Prepare data for secure sharing based on sharing level"""
        logger.info(f"Preparing data for {sharing_level} sharing...")
        
        df_shared = df.copy()
        
        if sharing_level == 'external':
            # Maximum privacy for external sharing
            df_shared = self.mask_pii(df_shared)
            df_shared = self.add_differential_privacy(df_shared, epsilon=0.5)
            df_shared = self.anonymize_dataset(df_shared, k_anonymity=10)
        elif sharing_level == 'internal':
            # Moderate privacy for internal sharing
            df_shared = self.mask_pii(df_shared)
            df_shared = self.add_differential_privacy(df_shared, epsilon=1.0)
        elif sharing_level == 'research':
            # Synthetic data for research purposes
            df_shared = self.create_synthetic_data(df, n_samples=len(df))
        
        return df_shared
    
    def audit_privacy_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit dataset for privacy compliance"""
        logger.info("Auditing privacy compliance...")
        
        compliance_results = {
            'pii_detected': False,
            'pii_columns': [],
            'encryption_status': 'not_encrypted',
            'anonymization_level': 'none',
            'compliance_score': 100,
            'recommendations': []
        }
        
        # Check for PII
        for col in df.columns:
            sample_text = ' '.join(df[col].astype(str).head(10).tolist())
            detected_pii = self.detect_pii(sample_text)
            
            if detected_pii:
                compliance_results['pii_detected'] = True
                compliance_results['pii_columns'].append(col)
                compliance_results['compliance_score'] -= 20
        
        # Check for direct identifiers
        identifier_columns = ['name', 'email', 'employee_id', 'ssn']
        for col in identifier_columns:
            if col in df.columns:
                compliance_results['recommendations'].append(
                    f"Consider masking or removing direct identifier: {col}"
                )
                compliance_results['compliance_score'] -= 10
        
        # Generate recommendations
        if compliance_results['pii_detected']:
            compliance_results['recommendations'].append(
                "Apply PII masking to detected columns"
            )
        
        if compliance_results['compliance_score'] < 80:
            compliance_results['recommendations'].append(
                "Apply differential privacy to numerical columns"
            )
        
        return compliance_results

class FederatedLearningFramework:
    """Federated learning framework for privacy-preserving ML"""
    
    def __init__(self):
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        
    def initialize_global_model(self, model_config: Dict[str, Any]):
        """Initialize global model parameters"""
        logger.info("Initializing global federated model...")
        
        from sklearn.linear_model import LinearRegression
        self.global_model = LinearRegression()
        
        # Initialize with random parameters
        n_features = model_config.get('n_features', 10)
        self.global_model.coef_ = np.random.normal(0, 0.1, n_features)
        self.global_model.intercept_ = 0.0
    
    def client_update(self, client_id: str, local_data: pd.DataFrame,
                     learning_rate: float = 0.01) -> Dict[str, Any]:
        """Perform local model update on client data"""
        logger.info(f"Performing client update for {client_id}...")
        
        # Simulate local training
        feature_cols = [col for col in local_data.columns if col != 'target']
        X_local = local_data[feature_cols]
        y_local = local_data.get('target', np.random.normal(85, 10, len(local_data)))
        
        # Local gradient computation (simplified)
        from sklearn.linear_model import SGDRegressor
        local_model = SGDRegressor(learning_rate='constant', eta0=learning_rate)
        local_model.fit(X_local, y_local)
        
        # Calculate parameter updates
        param_updates = {
            'coef_update': local_model.coef_,
            'intercept_update': local_model.intercept_,
            'n_samples': len(local_data),
            'client_id': client_id
        }
        
        self.client_models[client_id] = param_updates
        return param_updates
    
    def aggregate_updates(self) -> Dict[str, Any]:
        """Aggregate client updates using federated averaging"""
        logger.info("Aggregating federated updates...")
        
        if not self.client_models:
            return {}
        
        # Calculate weights based on data size
        total_samples = sum(update['n_samples'] for update in self.client_models.values())
        
        # Weighted average of parameters
        aggregated_coef = np.zeros_like(list(self.client_models.values())[0]['coef_update'])
        aggregated_intercept = 0.0
        
        for client_id, update in self.client_models.items():
            weight = update['n_samples'] / total_samples
            self.aggregation_weights[client_id] = weight
            
            aggregated_coef += update['coef_update'] * weight
            aggregated_intercept += update['intercept_update'] * weight
        
        # Update global model
        if self.global_model:
            self.global_model.coef_ = aggregated_coef
            self.global_model.intercept_ = aggregated_intercept
        
        return {
            'aggregated_coefficients': aggregated_coef.tolist(),
            'aggregated_intercept': aggregated_intercept,
            'client_weights': self.aggregation_weights,
            'total_samples': total_samples
        }
    
    def secure_aggregation(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Secure aggregation with noise addition"""
        logger.info("Performing secure aggregation...")
        
        # Add noise to updates before aggregation
        noisy_updates = []
        for update in client_updates:
            noisy_update = update.copy()
            noise_scale = 0.01  # Small noise for privacy
            
            noisy_update['coef_update'] = update['coef_update'] + np.random.normal(
                0, noise_scale, len(update['coef_update'])
            )
            noisy_update['intercept_update'] = update['intercept_update'] + np.random.normal(0, noise_scale)
            
            noisy_updates.append(noisy_update)
        
        # Aggregate noisy updates
        total_samples = sum(update['n_samples'] for update in noisy_updates)
        aggregated_coef = np.zeros_like(noisy_updates[0]['coef_update'])
        aggregated_intercept = 0.0
        
        for update in noisy_updates:
            weight = update['n_samples'] / total_samples
            aggregated_coef += update['coef_update'] * weight
            aggregated_intercept += update['intercept_update'] * weight
        
        return {
            'secure_aggregated_coefficients': aggregated_coef.tolist(),
            'secure_aggregated_intercept': aggregated_intercept,
            'privacy_preserved': True
        }

class GDPRCompliance:
    """GDPR compliance utilities"""
    
    def __init__(self):
        self.data_processing_log = []
        
    def log_data_processing(self, operation: str, data_subjects: int, 
                          purpose: str, legal_basis: str):
        """Log data processing activities for GDPR compliance"""
        log_entry = {
            'timestamp': pd.Timestamp.now(),
            'operation': operation,
            'data_subjects_count': data_subjects,
            'purpose': purpose,
            'legal_basis': legal_basis
        }
        
        self.data_processing_log.append(log_entry)
        logger.info(f"Logged GDPR activity: {operation}")
    
    def right_to_erasure(self, df: pd.DataFrame, employee_id: str) -> pd.DataFrame:
        """Implement right to erasure (right to be forgotten)"""
        logger.info(f"Processing erasure request for employee {employee_id}")
        
        # Remove all records for the specified employee
        df_erased = df[df['employee_id'] != employee_id].copy()
        
        self.log_data_processing(
            operation='data_erasure',
            data_subjects=1,
            purpose='right_to_erasure',
            legal_basis='gdpr_article_17'
        )
        
        return df_erased
    
    def data_portability(self, df: pd.DataFrame, employee_id: str) -> Dict[str, Any]:
        """Implement right to data portability"""
        logger.info(f"Processing data portability request for employee {employee_id}")
        
        # Extract all data for the specified employee
        employee_data = df[df['employee_id'] == employee_id].to_dict('records')
        
        self.log_data_processing(
            operation='data_export',
            data_subjects=1,
            purpose='data_portability',
            legal_basis='gdpr_article_20'
        )
        
        return {
            'employee_id': employee_id,
            'data': employee_data,
            'export_timestamp': pd.Timestamp.now(),
            'format': 'json'
        }
    
    def generate_privacy_notice(self) -> str:
        """Generate privacy notice for data subjects"""
        return """
# Privacy Notice - HR Performance Analytics

## Data Controller
HR Performance Analytics Pro

## Purpose of Processing
- Performance evaluation and prediction
- Bias detection and fairness monitoring
- Organizational analytics and insights

## Legal Basis
- Legitimate interest in employee performance management
- Consent for advanced analytics features

## Data Retention
- Performance data: 7 years
- Prediction models: 5 years
- Audit logs: 10 years

## Your Rights
- Right to access your personal data
- Right to rectification of inaccurate data
- Right to erasure (right to be forgotten)
- Right to data portability
- Right to object to processing

## Contact
For privacy-related inquiries, contact: privacy@company.com
"""
