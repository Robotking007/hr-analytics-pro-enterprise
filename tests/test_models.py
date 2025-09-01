"""
Unit tests for HR Performance Analytics Pro models
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ensemble_models import EnsemblePerformancePredictor
from src.data.feature_engineering import FeatureEngineer
from src.data.sample_generator import SampleDataGenerator

class TestEnsembleModels:
    """Test ensemble model functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        generator = SampleDataGenerator(seed=42)
        employees_df, performance_df = generator.generate_complete_dataset(100)
        
        # Merge data
        latest_performance = performance_df.groupby('employee_id').last().reset_index()
        merged_data = employees_df.merge(latest_performance, on='employee_id', how='left')
        
        return merged_data
    
    @pytest.fixture
    def engineered_features(self, sample_data):
        """Generate engineered features"""
        feature_engineer = FeatureEngineer()
        return feature_engineer.engineer_all_features(sample_data)
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = EnsemblePerformancePredictor()
        assert model.models == {}
        assert model.ensemble_weights == {}
        assert not model.is_trained
    
    def test_model_training(self, engineered_features):
        """Test model training"""
        # Prepare data
        performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
        y = engineered_features[performance_cols].mean(axis=1)
        
        exclude_cols = ['employee_id', 'name', 'email'] + performance_cols
        feature_cols = [col for col in engineered_features.columns if col not in exclude_cols]
        X = engineered_features[feature_cols].select_dtypes(include=[np.number])
        
        # Train model
        model = EnsemblePerformancePredictor()
        results = model.train_ensemble(X, y)
        
        assert model.is_trained
        assert 'ensemble_score' in results
        assert results['ensemble_score'] > 0
        assert len(model.models) > 0
    
    def test_prediction(self, engineered_features):
        """Test single prediction"""
        # Prepare data
        performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
        y = engineered_features[performance_cols].mean(axis=1)
        
        exclude_cols = ['employee_id', 'name', 'email'] + performance_cols
        feature_cols = [col for col in engineered_features.columns if col not in exclude_cols]
        X = engineered_features[feature_cols].select_dtypes(include=[np.number])
        
        # Train and predict
        model = EnsemblePerformancePredictor()
        model.train_ensemble(X, y)
        
        # Make prediction
        prediction = model.predict_single(X.iloc[[0]])
        
        assert 'ensemble_prediction' in prediction
        assert 'confidence' in prediction
        assert isinstance(prediction['ensemble_prediction'], (int, float))
        assert 0 <= prediction['confidence'] <= 100

class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        generator = SampleDataGenerator(seed=42)
        employees_df, _ = generator.generate_complete_dataset(50)
        return employees_df
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        fe = FeatureEngineer()
        assert fe.scalers == {}
        assert fe.encoders == {}
        assert fe.feature_names == []
    
    def test_demographic_features(self, sample_data):
        """Test demographic feature creation"""
        fe = FeatureEngineer()
        result = fe.create_demographic_features(sample_data)
        
        assert 'age_normalized' in result.columns
        assert 'tenure_years' in result.columns
        assert len(result) == len(sample_data)
    
    def test_complete_feature_engineering(self, sample_data):
        """Test complete feature engineering pipeline"""
        fe = FeatureEngineer()
        result = fe.engineer_all_features(sample_data)
        
        # Should have significantly more features
        assert len(result.columns) > len(sample_data.columns)
        assert len(result) == len(sample_data)
        
        # Check for key engineered features
        assert any('_mean_12m' in col for col in result.columns)
        assert any('dept_' in col for col in result.columns)

if __name__ == "__main__":
    pytest.main([__file__])
