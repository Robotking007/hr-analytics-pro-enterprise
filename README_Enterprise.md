# HR Analytics Pro - Enterprise Edition

🏢 **Complete Enterprise HR Analytics Platform with Authentication, Modern UI, and Comprehensive Features**

## 🚀 Features

### 🔐 Authentication & Security
- JWT-based authentication via Supabase
- Role-based access control (admin/user roles)
- Protected routes and session management
- Automatic sign out functionality
- Secure password hashing

### 🎨 Modern UI/UX
- Dark/light theme support with glassmorphism effects
- Responsive design for all screen sizes
- Smooth animations and transitions
- Beautiful gradient designs
- Toast notifications for user feedback
- Collapsible sidebar navigation
- Loading states and spinners
- Hover effects and interactive elements

### 📊 Comprehensive Dashboard
- **Key Performance Statistics Cards:**
  - Total employees count (1,247)
  - Average performance (87.3%)
  - Model accuracy (94.1%)
  - Bias score (0.82)

- **Quick Actions Panel:**
  - Add new employee
  - Run prediction model
  - Start bias audit
  - Export analytics
  - Upload files

- **Recent Insights Feed:**
  - High-risk turnover predictions
  - Performance model updates
  - Bias audit completions
  - Priority-based badge system

- **System Status Monitoring:**
  - Performance model health
  - Data pipeline status
  - Bias monitor activity
  - Real-time status badges

### 👥 Employee Management
- **Employee Statistics Dashboard**
- **Advanced Employee Directory:**
  - Searchable employee table
  - Filter by department/role
  - Performance visualization bars
  - Risk level badges (Low/Medium/High)
  - Employee details view
- **Add Employee Dialog** with form validation

### 📈 Analytics & Reporting
- **KPI Dashboard:**
  - Employee satisfaction (4.2/5.0)
  - Retention rate (94.2%)
  - Productivity index (127)
  - Training completion (89%)

- **Advanced Analytics:**
  - Department Performance Analysis
  - Performance Trends
  - Engagement Metrics
  - Goal Achievement Tracking

- **Export Functionality:**
  - Analytics report generation
  - Download capabilities

### 🤖 AI Predictions
- **Model Status Cards:**
  - Turnover prediction model
  - Performance forecasting
  - Career progression model
  - Risk assessment model

- **High-Risk Employee Analysis:**
  - Risk scoring (0-100)
  - Risk factors identification
  - Detailed employee dialogs

- **Interactive Model Execution:**
  - Start prediction runs
  - Real-time progress indicators

### ⚖️ Bias Audit & Fairness
- **Overall Fairness Metrics:**
  - Hiring fairness (82%)
  - Promotion equity (91%)
  - Compensation parity (76%)
  - Performance review fairness (88%)

- **Comprehensive Analysis:**
  - Detected Bias Tab
  - Demographics Tab
  - Recommendations Tab
  - Audit History

### 🔍 Data Explorer
- **Dataset Overview**
- **Advanced Data Tools:**
  - Browse Data: Interactive table with search/filter
  - Custom Query: SQL query builder interface
  - Analytics: Department distribution charts
  - Data Quality: Completeness/accuracy metrics

### ⚙️ Settings & Administration
- **Tabbed Settings Interface:**
  - Account: Profile information and status
  - Notifications: Toggle for various notification types
  - Security: Security features with critical indicators
  - Preferences: Theme, language, timezone settings
  - Integrations: Third-party service connections
  - System: Data management and system information

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd hr-performance-analytics-pro
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file with:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
JWT_SECRET=your_jwt_secret_key
```

4. **Run the application:**
```bash
streamlit run main_enterprise.py
```

## 🔑 Demo Credentials

### Admin Account
- **Email:** admin@hranalytics.com
- **Password:** admin123

### User Account
- **Email:** user@hranalytics.com
- **Password:** user123

## 📁 Project Structure

```
hr-performance-analytics-pro/
├── main_enterprise.py          # Main application entry point
├── auth.py                     # Authentication system
├── ui_components.py            # Modern UI components
├── streamlit_offline.py        # Original enhanced version
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
└── README_Enterprise.md        # This file
```

## 🔧 Technical Features

### Database Integration
- Supabase backend with real-time synchronization
- Row Level Security (RLS) policies
- Fallback to demo data for offline mode

### Component Architecture
- Reusable UI components with glassmorphism effects
- Modular authentication system
- Theme management system

### Performance
- Efficient data caching
- Optimized re-renders
- Real-time data updates

## 🚀 Usage

1. **Start the application** and navigate to the login page
2. **Sign in** with demo credentials or create a new account
3. **Explore the dashboard** with real-time KPIs and insights
4. **Manage employees** through the advanced directory
5. **Run AI predictions** for performance and risk analysis
6. **Conduct bias audits** for fairness compliance
7. **Analyze data** through the comprehensive explorer
8. **Configure settings** for personalized experience

## 🔒 Security Features

- JWT token-based authentication
- Password hashing with SHA-256
- Session management with automatic expiry
- Role-based access control
- Protected routes for sensitive data

## 🎨 UI/UX Features

- **Glassmorphism Design:** Modern glass-like effects with backdrop blur
- **Responsive Layout:** Adapts to all screen sizes
- **Theme System:** Light and dark modes with smooth transitions
- **Interactive Elements:** Hover effects, animations, and transitions
- **Professional Styling:** Enterprise-grade visual design

## 📊 Analytics Capabilities

- Real-time performance monitoring
- Predictive analytics with ML models
- Bias detection and fairness analysis
- Comprehensive reporting and exports
- Interactive data visualization

## 🤖 AI & ML Features

- Multiple prediction models (Random Forest, Gradient Boosting, Linear Regression)
- Feature importance analysis
- Risk assessment algorithms
- Performance forecasting
- Bias detection algorithms

## 🔧 Administration

- User management (admin role required)
- System configuration
- Data management tools
- Audit trail tracking
- Performance monitoring

## 📈 Future Enhancements

- Advanced ML model training interface
- Real-time notifications system
- Mobile application support
- Advanced reporting templates
- Integration with external HR systems

## 🆘 Support

For support and questions, please refer to the documentation or contact the development team.

---

**HR Analytics Pro - Enterprise Edition** - Transforming HR through intelligent analytics and modern technology.
