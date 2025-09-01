import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """Create a connection to the Supabase database"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Parse the connection string
    import re
    pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)'
    match = re.match(pattern, db_url)
    
    if not match:
        raise ValueError("Invalid DATABASE_URL format")
        
    user, password, host, port, dbname = match.groups()
    
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn

def setup_database():
    """Set up the database schema and initial data"""
    try:
        # Read the SQL file
        with open('supabase_setup.sql', 'r') as file:
            sql_commands = file.read()
        
        # Split the SQL commands by semicolon and filter out empty statements
        commands = [cmd.strip() for cmd in sql_commands.split(';') if cmd.strip()]
        
        # Execute each command
        with get_connection() as conn:
            with conn.cursor() as cur:
                for command in commands:
                    try:
                        if command:  # Skip empty commands
                            cur.execute(command)
                            print(f"Executed: {command[:50]}..." if len(command) > 50 else f"Executed: {command}")
                    except Exception as e:
                        print(f"Error executing command: {e}")
                        print(f"Command was: {command[:200]}..." if len(command) > 200 else f"Command was: {command}")
                        
        print("\nDatabase setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == "__main__":
    print("Setting up database schema and initial data...")
    setup_database()
