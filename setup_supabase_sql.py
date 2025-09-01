"""
Setup Supabase database with SQL commands
Run these SQL commands in your Supabase SQL Editor
"""

def generate_sql_commands():
    """Generate SQL commands to create tables"""
    
    sql_commands = """
-- HR Performance Analytics Database Setup
-- Run these commands in Supabase SQL Editor

-- 1. Create employees table
CREATE TABLE IF NOT EXISTS public.employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(100) NOT NULL,
    position VARCHAR(100) NOT NULL,
    hire_date DATE NOT NULL,
    salary INTEGER NOT NULL,
    performance_score DECIMAL(3,2),
    age INTEGER,
    gender VARCHAR(20),
    education VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create performance_reviews table
CREATE TABLE IF NOT EXISTS public.performance_reviews (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES public.employees(id),
    review_date DATE NOT NULL,
    performance_score DECIMAL(3,2),
    goals_met INTEGER,
    feedback TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Insert sample employee data
INSERT INTO public.employees (name, department, position, hire_date, salary, performance_score, age, gender, education) VALUES
('John Smith', 'Engineering', 'Senior Developer', '2022-01-15', 85000, 8.5, 32, 'Male', 'Bachelor''s'),
('Sarah Johnson', 'Marketing', 'Marketing Manager', '2021-03-10', 75000, 9.2, 29, 'Female', 'Master''s'),
('Mike Chen', 'Sales', 'Sales Lead', '2020-07-22', 70000, 8.8, 35, 'Male', 'Bachelor''s'),
('Lisa Rodriguez', 'HR', 'HR Specialist', '2023-02-01', 60000, 8.1, 27, 'Female', 'Bachelor''s'),
('David Wilson', 'Finance', 'Financial Analyst', '2022-09-15', 65000, 7.9, 31, 'Male', 'Master''s'),
('Emma Davis', 'Engineering', 'Junior Developer', '2023-06-01', 55000, 7.5, 25, 'Female', 'Bachelor''s'),
('Robert Brown', 'Sales', 'Account Manager', '2021-11-20', 68000, 8.3, 33, 'Male', 'Bachelor''s'),
('Jennifer Lee', 'Marketing', 'Content Specialist', '2022-04-10', 52000, 8.0, 28, 'Female', 'Bachelor''s'),
('Alex Garcia', 'Operations', 'Operations Manager', '2020-02-15', 78000, 8.7, 36, 'Male', 'Master''s'),
('Maria Gonzalez', 'HR', 'Recruiter', '2023-01-05', 58000, 8.2, 30, 'Female', 'Bachelor''s');

-- 4. Insert sample performance review data
INSERT INTO public.performance_reviews (employee_id, review_date, performance_score, goals_met, feedback) VALUES
(1, '2024-01-15', 8.5, 85, 'Excellent technical skills and team collaboration'),
(2, '2024-01-15', 9.2, 92, 'Outstanding marketing campaign results'),
(3, '2024-01-15', 8.8, 88, 'Consistently exceeds sales targets'),
(4, '2024-01-15', 8.1, 81, 'Strong HR processes and employee satisfaction'),
(5, '2024-01-15', 7.9, 79, 'Good financial analysis and reporting'),
(6, '2024-01-15', 7.5, 75, 'Promising junior developer with growth potential'),
(7, '2024-01-15', 8.3, 83, 'Reliable account management and client relations'),
(8, '2024-01-15', 8.0, 80, 'Creative content and strong engagement metrics'),
(9, '2024-01-15', 8.7, 87, 'Efficient operations and process improvements'),
(10, '2024-01-15', 8.2, 82, 'Successful recruitment and candidate experience');

-- 5. Enable Row Level Security (RLS) - Optional
ALTER TABLE public.employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.performance_reviews ENABLE ROW LEVEL SECURITY;

-- 6. Create policies for public access (adjust as needed)
CREATE POLICY "Enable read access for all users" ON public.employees FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.employees FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON public.employees FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON public.employees FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON public.performance_reviews FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON public.performance_reviews FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON public.performance_reviews FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON public.performance_reviews FOR DELETE USING (true);
"""
    
    return sql_commands

def main():
    """Main function to display setup instructions"""
    print("🗄️ Supabase Database Setup")
    print("=" * 50)
    print("📋 Follow these steps:")
    print("1. Go to your Supabase project dashboard")
    print("2. Navigate to SQL Editor")
    print("3. Copy and paste the SQL commands below")
    print("4. Run the commands")
    print("5. Verify tables are created")
    print("\n" + "="*50)
    print("SQL COMMANDS:")
    print("="*50)
    
    sql = generate_sql_commands()
    print(sql)
    
    # Save to file
    with open("supabase_setup.sql", "w", encoding="utf-8") as f:
        f.write(sql)
    
    print("\n" + "="*50)
    print("✅ SQL commands saved to 'supabase_setup.sql'")
    print("📝 Copy these commands to Supabase SQL Editor")

if __name__ == "__main__":
    main()
