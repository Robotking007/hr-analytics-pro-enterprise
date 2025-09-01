"""
Advanced Feature Engineering Pipeline for HR Performance Analytics
Generates 92+ engineered features from raw employee and performance data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for HR performance prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from historical performance data"""
        logger.info("Creating temporal features...")
        
        # Ensure we have the required columns
        required_cols = ['task_completion_rate', 'efficiency_score', 'quality_score', 
                        'collaboration_score', 'innovation_score', 'leadership_score',
                        'communication_score', 'problem_solving_score', 'adaptability_score',
                        'goal_achievement_rate']
        
        # Create 12-month historical features (simulated for demo)
        for metric in required_cols:
            for month in range(1, 13):
                col_name = f"{metric}_M{month}"
                # Simulate historical data with some noise
                if metric in df.columns:
                    df[col_name] = df[metric] + np.random.normal(0, 2, len(df))
                else:
                    df[col_name] = np.random.normal(85, 10, len(df))
        
        # Calculate rolling statistics
        for metric in required_cols:
            month_cols = [f"{metric}_M{i}" for i in range(1, 13)]
            if all(col in df.columns for col in month_cols):
                df[f"{metric}_mean_12m"] = df[month_cols].mean(axis=1)
                df[f"{metric}_std_12m"] = df[month_cols].std(axis=1)
                df[f"{metric}_trend_12m"] = df[month_cols].apply(
                    lambda x: np.polyfit(range(12), x, 1)[0], axis=1
                )
                df[f"{metric}_volatility_12m"] = df[month_cols].std(axis=1) / df[month_cols].mean(axis=1)
        
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic and position-based features"""
        logger.info("Creating demographic features...")
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['Gen_Z', 'Millennial', 'Gen_X_Young', 'Gen_X_Old', 'Boomer'])
            df['age_normalized'] = (df['age'] - df['age'].mean()) / df['age'].std()
        
        # Tenure calculation (if hire_date exists)
        if 'hire_date' in df.columns:
            df['hire_date'] = pd.to_datetime(df['hire_date'])
            df['tenure_days'] = (pd.Timestamp.now() - df['hire_date']).dt.days
            df['tenure_years'] = df['tenure_days'] / 365.25
            df['tenure_group'] = pd.cut(df['tenure_years'], 
                                      bins=[0, 1, 3, 5, 10, 100],
                                      labels=['New', 'Junior', 'Mid', 'Senior', 'Veteran'])
        
        # Salary features
        if 'salary' in df.columns:
            df['salary_normalized'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()
            df['salary_quartile'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Position level encoding
        if 'position_level' not in df.columns:
            # Create position level from position if it doesn't exist
            position_mapping = {
                'Junior': 1, 'Mid': 2, 'Senior': 3, 'Manager': 4, 'Director': 5,
                'Developer': 2, 'Senior Developer': 3, 'Lead Developer': 4,
                'Analyst': 2, 'Senior Analyst': 3, 'Manager': 4
            }
            df['position_level'] = df['position'].map(
                lambda x: max([v for k, v in position_mapping.items() if k in str(x)], default=2)
            )
        
        # Education scoring
        if 'education_score' not in df.columns:
            education_mapping = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
            df['education_score'] = df['education_level'].map(education_mapping).fillna(1)
        
        return df
    
    def create_department_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create department-based features with one-hot encoding"""
        logger.info("Creating department features...")
        
        if 'department' in df.columns:
            # One-hot encode departments
            dept_dummies = pd.get_dummies(df['department'], prefix='dept')
            df = pd.concat([df, dept_dummies], axis=1)
            
            # Department size (simulated)
            dept_sizes = df['department'].value_counts().to_dict()
            df['department_size'] = df['department'].map(dept_sizes)
            
            # Department average performance (if performance metrics exist)
            if 'task_completion_rate' in df.columns:
                dept_avg_perf = df.groupby('department')['task_completion_rate'].mean().to_dict()
                df['dept_avg_performance'] = df['department'].map(dept_avg_perf)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and polynomial features"""
        logger.info("Creating interaction features...")
        
        # Key performance metrics for interactions
        perf_metrics = ['task_completion_rate', 'efficiency_score', 'quality_score',
                       'collaboration_score', 'innovation_score']
        
        # Create interaction terms
        for i, metric1 in enumerate(perf_metrics):
            for metric2 in perf_metrics[i+1:]:
                if metric1 in df.columns and metric2 in df.columns:
                    df[f"{metric1}_x_{metric2}"] = df[metric1] * df[metric2]
        
        # Age and tenure interactions
        if 'age' in df.columns and 'tenure_years' in df.columns:
            df['age_tenure_ratio'] = df['age'] / (df['tenure_years'] + 1)
            df['experience_factor'] = df['age'] * df['tenure_years']
        
        # Salary and performance interactions
        if 'salary' in df.columns and 'task_completion_rate' in df.columns:
            df['salary_performance_ratio'] = df['salary'] / (df['task_completion_rate'] + 1)
            df['value_per_dollar'] = df['task_completion_rate'] / (df['salary'] / 1000)
        
        # Education and position level interactions
        if 'education_score' in df.columns and 'position_level' in df.columns:
            df['education_position_fit'] = df['education_score'] * df['position_level']
            df['overqualification_index'] = df['education_score'] - df['position_level']
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        logger.info("Creating statistical features...")
        
        # Performance metrics for statistical analysis
        perf_metrics = ['task_completion_rate', 'efficiency_score', 'quality_score',
                       'collaboration_score', 'innovation_score', 'leadership_score',
                       'communication_score', 'problem_solving_score', 'adaptability_score']
        
        # Overall performance score
        if all(metric in df.columns for metric in perf_metrics):
            df['overall_performance'] = df[perf_metrics].mean(axis=1)
            df['performance_variance'] = df[perf_metrics].var(axis=1)
            df['performance_range'] = df[perf_metrics].max(axis=1) - df[perf_metrics].min(axis=1)
            df['performance_skewness'] = df[perf_metrics].skew(axis=1)
        
        # Soft skills vs hard skills
        soft_skills = ['collaboration_score', 'leadership_score', 'communication_score', 'adaptability_score']
        hard_skills = ['task_completion_rate', 'efficiency_score', 'quality_score', 'problem_solving_score']
        
        if all(skill in df.columns for skill in soft_skills):
            df['soft_skills_avg'] = df[soft_skills].mean(axis=1)
        if all(skill in df.columns for skill in hard_skills):
            df['hard_skills_avg'] = df[hard_skills].mean(axis=1)
        
        if 'soft_skills_avg' in df.columns and 'hard_skills_avg' in df.columns:
            df['skills_balance'] = df['soft_skills_avg'] / df['hard_skills_avg']
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and contextual features"""
        logger.info("Creating behavioral features...")
        
        # Simulated behavioral metrics (in real implementation, these would come from actual data)
        np.random.seed(42)
        n_employees = len(df)
        
        # Meeting and collaboration metrics
        df['meetings_per_week'] = np.random.poisson(8, n_employees)
        df['slack_messages_per_day'] = np.random.poisson(25, n_employees)
        df['code_commits_per_week'] = np.random.poisson(12, n_employees)
        df['peer_reviews_given'] = np.random.poisson(5, n_employees)
        df['peer_reviews_received'] = np.random.poisson(3, n_employees)
        
        # Learning and development
        df['training_hours_per_month'] = np.random.gamma(2, 5, n_employees)
        df['certifications_count'] = np.random.poisson(2, n_employees)
        df['conference_attendance'] = np.random.poisson(1, n_employees)
        
        # Network centrality (simulated)
        df['network_centrality'] = np.random.beta(2, 5, n_employees)
        df['mentorship_relationships'] = np.random.poisson(2, n_employees)
        
        # Work-life balance indicators
        df['overtime_hours_per_week'] = np.random.gamma(1, 3, n_employees)
        df['vacation_days_taken'] = np.random.poisson(15, n_employees)
        df['sick_days_taken'] = np.random.poisson(3, n_employees)
        
        # Performance consistency
        if 'task_completion_rate' in df.columns:
            df['performance_consistency'] = 100 - np.random.exponential(5, n_employees)
            df['goal_alignment_score'] = df['task_completion_rate'] + np.random.normal(0, 5, n_employees)
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features"""
        logger.info("Creating advanced features...")
        
        # Risk indicators
        if 'tenure_years' in df.columns and 'salary' in df.columns:
            df['flight_risk_score'] = (
                (df['tenure_years'] < 2).astype(int) * 0.3 +
                (df['salary'] < df['salary'].quantile(0.25)).astype(int) * 0.3 +
                np.random.uniform(0, 0.4, len(df))
            )
        
        # Potential indicators
        if 'age' in df.columns and 'position_level' in df.columns:
            df['promotion_readiness'] = (
                df['position_level'] * 20 +
                (40 - df['age']) * 0.5 +
                np.random.normal(70, 15, len(df))
            ).clip(0, 100)
        
        # Team dynamics (simulated)
        df['team_collaboration_index'] = np.random.beta(3, 2, len(df)) * 100
        df['leadership_potential'] = np.random.gamma(2, 25, len(df))
        df['innovation_index'] = np.random.beta(2, 3, len(df)) * 100
        
        # Market and external factors
        df['market_stress_factor'] = np.random.uniform(0.8, 1.2, len(df))
        df['industry_growth_rate'] = np.random.normal(0.05, 0.02, len(df))
        df['economic_indicator'] = np.random.normal(1.0, 0.1, len(df))
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = ['gender', 'ethnicity', 'education_level', 'department', 'position']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f"{col}_encoded"] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key metrics"""
        logger.info("Creating polynomial features...")
        
        # Select numerical features for polynomial expansion
        numerical_cols = ['age', 'tenure_years', 'salary', 'position_level', 'education_score']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[available_cols])
            poly_feature_names = poly.get_feature_names_out(available_cols)
            
            # Add polynomial features to dataframe
            for i, name in enumerate(poly_feature_names):
                if name not in available_cols:  # Skip original features
                    df[f"poly_{name}"] = poly_features[:, i]
        
        return df
    
    def create_performance_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance ratio features"""
        logger.info("Creating performance ratios...")
        
        # Core performance metrics
        core_metrics = ['task_completion_rate', 'efficiency_score', 'quality_score']
        
        # Create ratios between different performance aspects
        if all(metric in df.columns for metric in core_metrics):
            df['efficiency_quality_ratio'] = df['efficiency_score'] / (df['quality_score'] + 1)
            df['completion_efficiency_ratio'] = df['task_completion_rate'] / (df['efficiency_score'] + 1)
            df['quality_completion_ratio'] = df['quality_score'] / (df['task_completion_rate'] + 1)
        
        # Leadership vs individual contributor ratios
        if 'leadership_score' in df.columns and 'task_completion_rate' in df.columns:
            df['leadership_execution_balance'] = df['leadership_score'] / (df['task_completion_rate'] + 1)
        
        # Communication vs technical skills
        if 'communication_score' in df.columns and 'problem_solving_score' in df.columns:
            df['communication_technical_balance'] = df['communication_score'] / (df['problem_solving_score'] + 1)
        
        return df
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual and environmental features"""
        logger.info("Creating contextual features...")
        
        # Team size and structure (simulated)
        df['team_size'] = np.random.poisson(8, len(df))
        df['direct_reports'] = np.where(df['position_level'] >= 4, 
                                       np.random.poisson(5, len(df)), 0)
        
        # Workload indicators
        df['project_count'] = np.random.poisson(3, len(df))
        df['deadline_pressure'] = np.random.beta(2, 3, len(df)) * 100
        df['resource_availability'] = np.random.beta(3, 2, len(df)) * 100
        
        # Career progression indicators
        if 'tenure_years' in df.columns and 'position_level' in df.columns:
            df['promotion_velocity'] = df['position_level'] / (df['tenure_years'] + 1)
            df['career_plateau_risk'] = np.where(
                (df['tenure_years'] > 3) & (df['position_level'] <= 2), 1, 0
            )
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Make a copy to avoid modifying original data
        df_engineered = df.copy()
        
        # Apply all feature engineering steps
        df_engineered = self.create_demographic_features(df_engineered)
        df_engineered = self.create_temporal_features(df_engineered)
        df_engineered = self.create_department_features(df_engineered)
        df_engineered = self.create_performance_ratios(df_engineered)
        df_engineered = self.create_behavioral_features(df_engineered)
        df_engineered = self.create_contextual_features(df_engineered)
        df_engineered = self.create_advanced_features(df_engineered)
        df_engineered = self.encode_categorical_features(df_engineered)
        df_engineered = self.create_polynomial_features(df_engineered)
        
        # Handle missing values
        df_engineered = self.handle_missing_values(df_engineered)
        
        # Store feature names
        self.feature_names = df_engineered.columns.tolist()
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
        return df_engineered
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value imputation"""
        logger.info("Handling missing values...")
        
        # Numerical columns - use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def select_top_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Select top k features using statistical tests"""
        logger.info(f"Selecting top {k} features...")
        
        # Ensure we have numerical data only
        X_numerical = X.select_dtypes(include=[np.number])
        
        if X_numerical.empty:
            logger.warning("No numerical features found for selection")
            return []
        
        # Use SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=min(k, X_numerical.shape[1]))
        selector.fit(X_numerical, y)
        
        # Get selected feature names
        selected_features = X_numerical.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} top features")
        return selected_features
    
    def get_feature_importance_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of engineered features"""
        feature_categories = {
            'temporal': [col for col in df.columns if any(x in col for x in ['_M', '_12m', 'trend', 'volatility'])],
            'demographic': [col for col in df.columns if any(x in col for x in ['age', 'tenure', 'education', 'gender', 'ethnicity'])],
            'department': [col for col in df.columns if 'dept_' in col or 'department' in col],
            'performance': [col for col in df.columns if any(x in col for x in ['score', 'rate', 'performance'])],
            'behavioral': [col for col in df.columns if any(x in col for x in ['meetings', 'slack', 'commits', 'training'])],
            'interaction': [col for col in df.columns if '_x_' in col or 'ratio' in col or 'balance' in col],
            'contextual': [col for col in df.columns if any(x in col for x in ['team', 'project', 'workload', 'stress'])],
            'polynomial': [col for col in df.columns if 'poly_' in col]
        }
        
        summary = {
            'total_features': len(df.columns),
            'categories': {cat: len(features) for cat, features in feature_categories.items()},
            'feature_names': self.feature_names
        }
        
        return summary
