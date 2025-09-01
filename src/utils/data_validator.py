"""
Data validation utilities for HR Performance Analytics Pro
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from loguru import logger

class DataValidator:
    """Comprehensive data validation for HR analytics"""
    
    def __init__(self):
        self.validation_rules = {
            'employee_id': {'required': True, 'unique': True, 'type': str},
            'age': {'min': 18, 'max': 75, 'type': int},
            'salary': {'min': 20000, 'max': 500000, 'type': float},
            'position_level': {'min': 1, 'max': 5, 'type': int},
            'task_completion_rate': {'min': 0, 'max': 100, 'type': float},
            'efficiency_score': {'min': 0, 'max': 100, 'type': float}
        }
    
    def validate_employee_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate employee dataset"""
        logger.info("Validating employee data...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check required columns
        required_cols = ['employee_id', 'name', 'department', 'position']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        for col, rules in self.validation_rules.items():
            if col in df.columns:
                # Type validation
                if 'type' in rules:
                    try:
                        df[col].astype(rules['type'])
                    except:
                        validation_results['warnings'].append(f"Column {col} has invalid data types")
                
                # Range validation
                if 'min' in rules and df[col].min() < rules['min']:
                    validation_results['warnings'].append(f"Column {col} has values below minimum ({rules['min']})")
                
                if 'max' in rules and df[col].max() > rules['max']:
                    validation_results['warnings'].append(f"Column {col} has values above maximum ({rules['max']})")
                
                # Uniqueness validation
                if rules.get('unique', False) and df[col].duplicated().any():
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Column {col} contains duplicate values")
        
        # Data quality summary
        validation_results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return validation_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Standardize text columns
        text_cols = df_clean.select_dtypes(include=['object']).columns
        for col in text_cols:
            if col in ['name', 'department', 'position']:
                df_clean[col] = df_clean[col].str.strip().str.title()
            elif col == 'email':
                df_clean[col] = df_clean[col].str.strip().str.lower()
        
        # Cap numerical values within reasonable ranges
        if 'age' in df_clean.columns:
            df_clean['age'] = df_clean['age'].clip(18, 75)
        
        if 'salary' in df_clean.columns:
            df_clean['salary'] = df_clean['salary'].clip(20000, 500000)
        
        # Performance scores should be 0-100
        perf_cols = [col for col in df_clean.columns if 'score' in col or 'rate' in col]
        for col in perf_cols:
            df_clean[col] = df_clean[col].clip(0, 100)
        
        return df_clean
