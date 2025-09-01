"""
Database initialization script for HR Performance Analytics Pro
"""
import asyncio
import os
from src.database.supabase_client import SupabaseClient
from loguru import logger

async def init_database():
    """Initialize Supabase database with required tables"""
    try:
        client = SupabaseClient()
        logger.info("Initializing HR Analytics database...")
        
        await client.create_tables()
        logger.info("Database initialization completed successfully!")
        
        # Create sample data if needed
        await create_sample_data(client)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def create_sample_data(client: SupabaseClient):
    """Create sample data for testing"""
    try:
        # Check if data already exists
        result = client.client.table("employees").select("count").execute()
        if result.data and len(result.data) > 0:
            logger.info("Sample data already exists, skipping creation")
            return
        
        # Sample employees
        sample_employees = [
            {
                "employee_id": "EMP001",
                "name": "John Smith",
                "email": "john.smith@company.com",
                "department": "Engineering",
                "position": "Senior Developer",
                "position_level": 3,
                "salary": 85000.00,
                "hire_date": "2022-01-15",
                "age": 32,
                "gender": "Male",
                "ethnicity": "Caucasian",
                "education_level": "Master",
                "education_score": 2,
                "manager_id": "MGR001"
            },
            {
                "employee_id": "EMP002",
                "name": "Sarah Johnson",
                "email": "sarah.johnson@company.com",
                "department": "Marketing",
                "position": "Marketing Manager",
                "position_level": 4,
                "salary": 75000.00,
                "hire_date": "2021-06-10",
                "age": 29,
                "gender": "Female",
                "ethnicity": "African American",
                "education_level": "Bachelor",
                "education_score": 1,
                "manager_id": "MGR002"
            },
            {
                "employee_id": "EMP003",
                "name": "David Chen",
                "email": "david.chen@company.com",
                "department": "Data Science",
                "position": "Data Scientist",
                "position_level": 3,
                "salary": 90000.00,
                "hire_date": "2020-03-20",
                "age": 35,
                "gender": "Male",
                "ethnicity": "Asian",
                "education_level": "PhD",
                "education_score": 3,
                "manager_id": "MGR003"
            }
        ]
        
        # Insert sample employees
        for employee in sample_employees:
            client.insert_employee(employee)
        
        # Sample performance metrics
        sample_metrics = [
            {
                "employee_id": "EMP001",
                "metric_date": "2024-08-01",
                "task_completion_rate": 92.5,
                "efficiency_score": 88.0,
                "quality_score": 90.0,
                "collaboration_score": 85.0,
                "innovation_score": 87.0,
                "leadership_score": 82.0,
                "communication_score": 89.0,
                "problem_solving_score": 91.0,
                "adaptability_score": 86.0,
                "goal_achievement_rate": 94.0
            },
            {
                "employee_id": "EMP002",
                "metric_date": "2024-08-01",
                "task_completion_rate": 89.0,
                "efficiency_score": 91.0,
                "quality_score": 88.0,
                "collaboration_score": 93.0,
                "innovation_score": 85.0,
                "leadership_score": 90.0,
                "communication_score": 95.0,
                "problem_solving_score": 87.0,
                "adaptability_score": 92.0,
                "goal_achievement_rate": 91.0
            },
            {
                "employee_id": "EMP003",
                "metric_date": "2024-08-01",
                "task_completion_rate": 95.0,
                "efficiency_score": 93.0,
                "quality_score": 96.0,
                "collaboration_score": 88.0,
                "innovation_score": 94.0,
                "leadership_score": 85.0,
                "communication_score": 87.0,
                "problem_solving_score": 97.0,
                "adaptability_score": 89.0,
                "goal_achievement_rate": 96.0
            }
        ]
        
        client.insert_performance_metrics(sample_metrics)
        logger.info("Sample data created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")

if __name__ == "__main__":
    asyncio.run(init_database())
