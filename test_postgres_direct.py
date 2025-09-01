"""
Test Direct PostgreSQL Connection to Supabase
"""
import os
import sys
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_postgres_connection():
    """Test direct PostgreSQL connection"""
    try:
        from src.database.postgres_client import PostgreSQLClient
        
        # Get password from user
        password = getpass.getpass("Enter your Supabase database password: ")
        
        # Create client and connect
        client = PostgreSQLClient()
        
        print("🔗 Connecting to PostgreSQL database...")
        if client.connect(password):
            print("✅ Database connection successful!")
            
            # Create tables
            print("🗄️ Creating database tables...")
            if client.create_tables():
                print("✅ Tables created successfully!")
                
                # Test data insertion
                print("📊 Testing data insertion...")
                
                # Add src to path for imports
                from src.data.sample_generator import SampleDataGenerator
                
                generator = SampleDataGenerator()
                employees_df, performance_df = generator.generate_complete_dataset(5)
                
                # Insert sample employees
                success_count = 0
                for _, employee in employees_df.iterrows():
                    if client.insert_employee(employee.to_dict()):
                        success_count += 1
                
                print(f"✅ Inserted {success_count}/{len(employees_df)} employees")
                
                # Insert sample performance data
                perf_success = 0
                for _, perf in performance_df.iterrows():
                    if client.insert_performance_data(perf.to_dict()):
                        perf_success += 1
                
                print(f"✅ Inserted {perf_success}/{len(performance_df)} performance records")
                
                # Test data retrieval
                employees = client.get_all_employees(limit=10)
                performance = client.get_performance_data(limit=10)
                
                print(f"✅ Retrieved {len(employees)} employees from database")
                print(f"✅ Retrieved {len(performance)} performance records from database")
                
                client.close()
                
                print("\n🎉 PostgreSQL connection test passed!")
                print("Your database is ready for the HR Analytics system!")
                return True
                
            else:
                print("❌ Failed to create tables")
                client.close()
                return False
        else:
            print("❌ Failed to connect to database")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 PostgreSQL Connection Test for HR Performance Analytics Pro")
    print("=" * 60)
    print("Database Details:")
    print(f"Host: {os.getenv('DB_HOST', 'db.ybtcbjycttvuvqkdgxtr.supabase.co')}")
    print(f"Port: {os.getenv('DB_PORT', '5432')}")
    print(f"User: {os.getenv('DB_USER', 'postgres')}")
    print(f"Database: {os.getenv('DB_NAME', 'postgres')}")
    print("=" * 60)
    
    success = test_postgres_connection()
    sys.exit(0 if success else 1)
