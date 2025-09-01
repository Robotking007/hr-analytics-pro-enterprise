"""
System Test Script for HR Performance Analytics Pro
Tests all major components and functionality
"""
import sys
import os
import asyncio
import requests
import time
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all critical imports"""
    logger.info("Testing imports...")
    
    try:
        from src.models.ensemble_models import EnsemblePerformancePredictor
        from src.data.feature_engineering import FeatureEngineer
        from src.data.sample_generator import SampleDataGenerator
        from src.bias.fairness_monitor import FairnessMonitor
        from src.explainability.model_explainer import ModelExplainer
        from src.privacy.data_protection import DataProtectionManager
        from src.utils.config import settings
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_data_generation():
    """Test sample data generation"""
    logger.info("Testing data generation...")
    
    try:
        from src.data.sample_generator import SampleDataGenerator
        generator = SampleDataGenerator()
        employees_df, performance_df = generator.generate_complete_dataset(10)
        
        assert len(employees_df) == 10
        assert len(performance_df) > 0
        logger.info("✅ Data generation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Data generation failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering pipeline"""
    logger.info("Testing feature engineering...")
    
    try:
        from src.data.sample_generator import SampleDataGenerator
        from src.data.feature_engineering import FeatureEngineer
        
        generator = SampleDataGenerator()
        employees_df, performance_df = generator.generate_complete_dataset(10)
        
        # Merge data
        latest_performance = performance_df.groupby('employee_id').last().reset_index()
        merged_data = employees_df.merge(latest_performance, on='employee_id', how='left')
        
        # Engineer features
        fe = FeatureEngineer()
        engineered_df = fe.engineer_all_features(merged_data)
        
        assert len(engineered_df.columns) > len(merged_data.columns)
        logger.info(f"✅ Feature engineering successful: {len(engineered_df.columns)} features created")
        return True
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return False

def test_model_training():
    """Test model training"""
    logger.info("Testing model training...")
    
    try:
        from src.models.ensemble_models import EnsemblePerformancePredictor
        from src.data.sample_generator import SampleDataGenerator
        from src.data.feature_engineering import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # Generate data
        generator = SampleDataGenerator()
        employees_df, performance_df = generator.generate_complete_dataset(50)
        
        # Merge and engineer features
        latest_performance = performance_df.groupby('employee_id').last().reset_index()
        merged_data = employees_df.merge(latest_performance, on='employee_id', how='left')
        
        fe = FeatureEngineer()
        engineered_df = fe.engineer_all_features(merged_data)
        
        # Prepare training data
        performance_cols = ['task_completion_rate', 'efficiency_score', 'quality_score']
        y = engineered_df[performance_cols].mean(axis=1)
        
        exclude_cols = ['employee_id', 'name', 'email'] + performance_cols
        feature_cols = [col for col in engineered_df.columns if col not in exclude_cols]
        X = engineered_df[feature_cols].select_dtypes(include=[np.number])
        
        # Train model
        model = EnsemblePerformancePredictor()
        results = model.train_ensemble(X, y)
        
        assert model.is_trained
        assert results['ensemble_score'] > 0
        logger.info(f"✅ Model training successful: R² = {results['ensemble_score']:.3f}")
        return True
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return False

def test_api_health():
    """Test API health (if running)"""
    logger.info("Testing API health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API is healthy")
            return True
        else:
            logger.warning("⚠️ API not responding (this is OK if not started)")
            return False
    except requests.exceptions.RequestException:
        logger.warning("⚠️ API not available (this is OK if not started)")
        return False

def run_system_test():
    """Run complete system test"""
    logger.info("🚀 Starting HR Performance Analytics Pro System Test")
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("API Health", test_api_health)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("🎯 SYSTEM TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
    else:
        logger.warning("⚠️ Some tests failed. Check logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_system_test()
    sys.exit(0 if success else 1)
