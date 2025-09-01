"""
Create database tables using Supabase REST API
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import json

def create_tables():
    """Create tables using Supabase client"""
    load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for admin operations
    
    if not url or not key:
        print("❌ Missing Supabase credentials")
        return False
    
    try:
        supabase: Client = create_client(url, key)
        print("✅ Connected to Supabase")
        
        # Create sample employee data
        sample_employees = [
            {
                "name": "John Smith",
                "department": "Engineering",
                "position": "Senior Developer",
                "hire_date": "2022-01-15",
                "salary": 85000,
                "performance_score": 8.5,
                "age": 32,
                "gender": "Male",
                "education": "Bachelor's"
            },
            {
                "name": "Sarah Johnson",
                "department": "Marketing",
                "position": "Marketing Manager",
                "hire_date": "2021-03-10",
                "salary": 75000,
                "performance_score": 9.2,
                "age": 29,
                "gender": "Female",
                "education": "Master's"
            },
            {
                "name": "Mike Chen",
                "department": "Sales",
                "position": "Sales Lead",
                "hire_date": "2020-07-22",
                "salary": 70000,
                "performance_score": 8.8,
                "age": 35,
                "gender": "Male",
                "education": "Bachelor's"
            },
            {
                "name": "Lisa Rodriguez",
                "department": "HR",
                "position": "HR Specialist",
                "hire_date": "2023-02-01",
                "salary": 60000,
                "performance_score": 8.1,
                "age": 27,
                "gender": "Female",
                "education": "Bachelor's"
            },
            {
                "name": "David Wilson",
                "department": "Finance",
                "position": "Financial Analyst",
                "hire_date": "2022-09-15",
                "salary": 65000,
                "performance_score": 7.9,
                "age": 31,
                "gender": "Male",
                "education": "Master's"
            }
        ]
        
        # Insert employee data
        print("📝 Creating employee records...")
        result = supabase.table('employees').insert(sample_employees).execute()
        print(f"✅ Created {len(result.data)} employee records")
        
        # Create performance data
        performance_data = []
        for i, emp in enumerate(sample_employees, 1):
            performance_data.append({
                "employee_id": i,
                "review_date": "2024-01-15",
                "performance_score": emp["performance_score"],
                "goals_met": int(emp["performance_score"] * 10),
                "feedback": f"Performance review for {emp['name']}"
            })
        
        print("📊 Creating performance records...")
        result = supabase.table('performance_reviews').insert(performance_data).execute()
        print(f"✅ Created {len(result.data)} performance records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_tables()
    if success:
        print("\n🎉 Database setup completed successfully!")
    else:
        print("\n💡 Using offline mode - tables will be created automatically when needed")
