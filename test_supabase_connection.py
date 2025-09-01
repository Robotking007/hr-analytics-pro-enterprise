"""
Test Supabase Connection and Database Setup
"""
import sys
import os
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_supabase_connection():
    """Test connection to Supabase"""
    try:
        from src.database.supabase_client import SupabaseClient
        
        logger.info("Testing Supabase connection...")
        client = SupabaseClient()
        
        # Test connection by creating tables
        success = client.create_tables()
        
        if success:
            logger.info("✅ Supabase connection successful!")
            logger.info("✅ Database tables created/verified")
            
            # Test data insertion
            from src.data.sample_generator import SampleDataGenerator
            generator = SampleDataGenerator()
            employees_df, performance_df = generator.generate_complete_dataset(5)
            
            # Insert sample employees
            for _, employee in employees_df.iterrows():
                employee_data = employee.to_dict()
                result = client.insert_employee(employee_data)
                if result:
                    logger.info(f"✅ Inserted employee: {employee['name']}")
                else:
                    logger.warning(f"⚠️ Failed to insert employee: {employee['name']}")
            
            # Insert sample performance data
            for _, perf in performance_df.iterrows():
                perf_data = perf.to_dict()
                result = client.insert_performance_data(perf_data)
                if result:
                    logger.info(f"✅ Inserted performance data for employee {perf['employee_id']}")
            
            # Test data retrieval
            employees = client.get_all_employees()
            performance = client.get_performance_data()
            
            logger.info(f"✅ Retrieved {len(employees)} employees from database")
            logger.info(f"✅ Retrieved {len(performance)} performance records from database")
            
            return True
            
        else:
            logger.error("❌ Failed to create database tables")
            return False
            
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_supabase_connection()
    if success:
        print("\n🎉 Supabase integration test passed!")
        print("Your HR Performance Analytics Pro system is ready to use with live database!")
    else:
        print("\n❌ Supabase integration test failed.")
        print("Check your credentials in .env file and try again.")
    
    sys.exit(0 if success else 1)
