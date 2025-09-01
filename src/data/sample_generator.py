"""
Sample Data Generator for HR Performance Analytics
Creates realistic synthetic employee and performance data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger

class SampleDataGenerator:
    """Generate realistic sample data for HR analytics"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'Data Science']
        self.positions = {
            'Engineering': ['Junior Developer', 'Senior Developer', 'Lead Developer', 'Engineering Manager', 'VP Engineering'],
            'Marketing': ['Marketing Coordinator', 'Marketing Specialist', 'Marketing Manager', 'Marketing Director'],
            'Sales': ['Sales Rep', 'Senior Sales Rep', 'Sales Manager', 'Sales Director'],
            'HR': ['HR Coordinator', 'HR Specialist', 'HR Manager', 'HR Director'],
            'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'CFO'],
            'Operations': ['Operations Coordinator', 'Operations Manager', 'Operations Director'],
            'Data Science': ['Data Analyst', 'Data Scientist', 'Senior Data Scientist', 'Data Science Manager']
        }
        self.genders = ['Male', 'Female', 'Non-binary']
        self.ethnicities = ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Native American', 'Other']
        self.education_levels = ['Bachelor', 'Master', 'PhD']
        
    def generate_employees(self, n_employees: int = 1000) -> pd.DataFrame:
        """Generate employee master data"""
        logger.info(f"Generating {n_employees} employee records...")
        
        employees = []
        
        for i in range(n_employees):
            department = np.random.choice(self.departments)
            position = np.random.choice(self.positions[department])
            
            # Position level mapping
            position_level = self._get_position_level(position)
            
            # Age and tenure correlation
            age = np.random.randint(22, 65)
            max_tenure_years = age - 22
            tenure_years = min(np.random.exponential(3), max_tenure_years)
            hire_date = datetime.now() - timedelta(days=int(tenure_years * 365.25))
            
            # Salary based on position and experience
            base_salary = self._calculate_base_salary(department, position_level)
            salary = base_salary + (tenure_years * 2000) + np.random.normal(0, 5000)
            salary = max(salary, 35000)  # Minimum salary
            
            employee = {
                'employee_id': f'EMP{i+1:04d}',
                'name': f'Employee {i+1}',
                'email': f'employee{i+1}@company.com',
                'department': department,
                'position': position,
                'position_level': position_level,
                'salary': round(salary, 2),
                'hire_date': hire_date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': np.random.choice(self.genders, p=[0.48, 0.48, 0.04]),
                'ethnicity': np.random.choice(self.ethnicities, p=[0.6, 0.15, 0.15, 0.07, 0.02, 0.01]),
                'education_level': np.random.choice(self.education_levels, p=[0.5, 0.4, 0.1]),
                'manager_id': f'MGR{np.random.randint(1, 50):03d}' if position_level < 4 else None
            }
            
            employees.append(employee)
        
        return pd.DataFrame(employees)
    
    def generate_performance_metrics(self, employees_df: pd.DataFrame, 
                                   months_history: int = 12) -> pd.DataFrame:
        """Generate historical performance metrics"""
        logger.info(f"Generating {months_history} months of performance data...")
        
        performance_records = []
        
        for _, employee in employees_df.iterrows():
            # Base performance influenced by position level and education
            base_performance = 70 + (employee['position_level'] * 5) + (employee.get('education_score', 1) * 3)
            
            # Generate monthly performance data
            for month_offset in range(months_history):
                metric_date = datetime.now() - timedelta(days=30 * month_offset)
                
                # Add some trend and seasonality
                trend = month_offset * 0.5  # Slight improvement over time
                seasonality = 5 * np.sin(2 * np.pi * month_offset / 12)  # Annual cycle
                noise = np.random.normal(0, 5)
                
                performance_base = base_performance + trend + seasonality + noise
                
                # Individual metric variations
                metrics = {
                    'employee_id': employee['employee_id'],
                    'metric_date': metric_date.strftime('%Y-%m-%d'),
                    'task_completion_rate': max(0, min(100, performance_base + np.random.normal(0, 3))),
                    'efficiency_score': max(0, min(100, performance_base + np.random.normal(0, 4))),
                    'quality_score': max(0, min(100, performance_base + np.random.normal(0, 3))),
                    'collaboration_score': max(0, min(100, performance_base + np.random.normal(0, 5))),
                    'innovation_score': max(0, min(100, performance_base + np.random.normal(0, 6))),
                    'leadership_score': max(0, min(100, performance_base + np.random.normal(0, 4))),
                    'communication_score': max(0, min(100, performance_base + np.random.normal(0, 4))),
                    'problem_solving_score': max(0, min(100, performance_base + np.random.normal(0, 3))),
                    'adaptability_score': max(0, min(100, performance_base + np.random.normal(0, 5))),
                    'goal_achievement_rate': max(0, min(100, performance_base + np.random.normal(0, 4)))
                }
                
                performance_records.append(metrics)
        
        return pd.DataFrame(performance_records)
    
    def _get_position_level(self, position: str) -> int:
        """Map position to level"""
        if any(word in position.lower() for word in ['junior', 'coordinator']):
            return 1
        elif any(word in position.lower() for word in ['senior', 'specialist']):
            return 2
        elif any(word in position.lower() for word in ['lead', 'principal']):
            return 3
        elif any(word in position.lower() for word in ['manager']):
            return 4
        elif any(word in position.lower() for word in ['director', 'vp', 'cfo']):
            return 5
        else:
            return 2  # Default
    
    def _calculate_base_salary(self, department: str, position_level: int) -> float:
        """Calculate base salary based on department and level"""
        dept_multipliers = {
            'Engineering': 1.2,
            'Data Science': 1.15,
            'Finance': 1.1,
            'Marketing': 1.0,
            'Sales': 1.05,
            'HR': 0.95,
            'Operations': 0.9
        }
        
        base_salaries = {1: 50000, 2: 65000, 3: 80000, 4: 100000, 5: 130000}
        
        return base_salaries[position_level] * dept_multipliers.get(department, 1.0)
    
    def generate_complete_dataset(self, n_employees: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with employees and performance metrics"""
        employees_df = self.generate_employees(n_employees)
        
        # Add education score
        education_mapping = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
        employees_df['education_score'] = employees_df['education_level'].map(education_mapping)
        
        performance_df = self.generate_performance_metrics(employees_df)
        
        return employees_df, performance_df
