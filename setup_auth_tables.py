import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def create_auth_tables():
    """Create necessary authentication tables in Supabase"""
    try:
        # Create users table
        result = supabase.rpc('pg_catalog.pg_tables_are_missing', {'schema_name': 'auth', 'table_name': 'users'}).execute()
        
        if result.data:
            print("Creating auth.users table...")
            # This will be automatically created by Supabase Auth
            print("auth.users table should be automatically managed by Supabase Auth")
        
        # Create profiles table (extended user data)
        create_profiles = """
        CREATE TABLE IF NOT EXISTS public.profiles (
            id UUID REFERENCES auth.users ON DELETE CASCADE,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user')),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id)
        );
        """
        
        supabase.rpc('pg_execute', {'query': create_profiles}).execute()
        print("Created profiles table")
        
        # Enable RLS on profiles
        supabase.rpc('pg_execute', {'query': 'ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;'}).execute()
        
        # Create policies for profiles
        policies = [
            """
            CREATE POLICY "Public profiles are viewable by everyone." 
            ON public.profiles FOR SELECT 
            USING (true);
            """,
            """
            CREATE POLICY "Users can insert their own profile." 
            ON public.profiles FOR INSERT 
            WITH CHECK (auth.uid() = id);
            """,
            """
            CREATE POLICY "Users can update own profile." 
            ON public.profiles FOR UPDATE 
            USING (auth.uid() = id);
            """
        ]
        
        for policy in policies:
            try:
                supabase.rpc('pg_execute', {'query': policy}).execute()
            except Exception as e:
                print(f"Error creating policy (might already exist): {e}")
        
        print("Authentication tables and policies set up successfully!")
        
    except Exception as e:
        print(f"Error setting up auth tables: {e}")

def create_trigger_for_new_user():
    """Create a trigger to automatically create a profile when a new user signs up"""
    try:
        trigger_function = """
        CREATE OR REPLACE FUNCTION public.handle_new_user()
        RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO public.profiles (id, email, role)
            VALUES (
                NEW.id,
                NEW.email,
                CASE 
                    WHEN NEW.email = 'admin@hranalytics.com' THEN 'admin'
                    ELSE 'user'
                END
            )
            ON CONFLICT (id) DO NOTHING;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        """
        
        # Drop trigger if it exists
        try:
            supabase.rpc('pg_execute', 
                {'query': 'DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;'}).execute()
        except:
            pass
            
        supabase.rpc('pg_execute', {'query': trigger_function}).execute()
        
        create_trigger = """
        CREATE TRIGGER on_auth_user_created
            AFTER INSERT ON auth.users
            FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
        """
        supabase.rpc('pg_execute', {'query': create_trigger}).execute()
        
        print("Created trigger for new user registration")
        
    except Exception as e:
        print(f"Error creating user trigger: {e}")

def create_admin_user():
    """Create an initial admin user if it doesn't exist"""
    try:
        # Check if admin exists
        admin_email = "admin@hranalytics.com"
        admin_password = "admin123"
        
        # Try to sign in admin (to check if exists)
        try:
            result = supabase.auth.sign_in_with_password({
                "email": admin_email,
                "password": admin_password
            })
            print("Admin user already exists")
            return
        except:
            pass  # User doesn't exist, will create
        
        # Create admin user
        result = supabase.auth.sign_up({
            "email": admin_email,
            "password": admin_password,
            "options": {
                "data": {
                    "role": "admin"
                }
            }
        })
        
        # Update the role in profiles table
        if hasattr(result, 'user') and hasattr(result.user, 'id'):
            supabase.table('profiles')\
                .update({'role': 'admin'})\
                .eq('email', admin_email)\
                .execute()
            
            print("Admin user created successfully!")
            print(f"Email: {admin_email}")
            print(f"Password: {admin_password}")
        else:
            print("Error creating admin user")
            
    except Exception as e:
        print(f"Error creating admin user: {e}")

if __name__ == "__main__":
    print("Setting up authentication tables...")
    create_auth_tables()
    create_trigger_for_new_user()
    create_admin_user()
    print("\nSetup complete! You can now run the application with authentication.")
    print("Use the following credentials to log in as admin:")
    print("Email: admin@hranalytics.com")
    print("Password: admin123")
