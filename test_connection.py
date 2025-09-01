"""
Test Supabase connection and debug connectivity issues
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import requests

def test_connection():
    """Test Supabase connection step by step"""
    print("🔍 Testing Supabase Connection")
    print("=" * 40)
    
    # Load environment
    load_dotenv()
    
    # Check environment variables
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    print(f"📋 URL: {url}")
    print(f"📋 Key: {key[:30]}..." if key else "Key: None")
    
    if not url or not key:
        print("❌ Missing credentials in .env file")
        return False
    
    # Test basic HTTP connectivity
    print("\n🌐 Testing HTTP connectivity...")
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ HTTP Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Error: {e}")
        return False
    
    # Test Supabase client
    print("\n🔗 Testing Supabase client...")
    try:
        supabase: Client = create_client(url, key)
        print("✅ Client created successfully")
        
        # Test simple query
        result = supabase.table('employees').select('count').limit(1).execute()
        print(f"✅ Database query successful: {len(result.data)} records")
        return True
        
    except Exception as e:
        print(f"❌ Supabase Error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    print(f"\n🎯 Connection Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
