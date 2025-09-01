"""
Authentication module for HR Analytics Pro
Handles JWT-based authentication, role-based access control, and session management
"""
import streamlit as st
import jwt
import hashlib
import time
import qrcode
import io
import base64
from datetime import datetime, timedelta
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import secrets
import pyotp

load_dotenv()

class AuthManager:
    def __init__(self):
        self.supabase = self._init_supabase()
        self.jwt_secret = os.getenv("JWT_SECRET", "hr-analytics-secret-key")
        
    def _init_supabase(self):
        """Initialize Supabase client for authentication"""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if url and key:
                return create_client(url, key)
        except Exception as e:
            st.error(f"Failed to initialize Supabase: {e}")
        return None
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def generate_jwt_token(self, user_data: dict) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': user_data['id'],
            'email': user_data['email'],
            'role': user_data.get('role', 'user'),
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}
    
    def sign_up(self, email: str, password: str, full_name: str, role: str = 'user') -> dict:
        """Register new user using Supabase Auth"""
        try:
            if self.supabase:
                # Use Supabase Auth to create user
                result = self.supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {
                            "full_name": full_name,
                            "role": role
                        }
                    }
                })
                
                if result.user:
                    return {'success': True, 'user': {
                        'id': result.user.id,
                        'email': result.user.email,
                        'role': role,
                        'full_name': full_name
                    }}
                else:
                    return {'error': 'Failed to create user'}
            else:
                # Fallback for offline mode
                return {'success': True, 'user': {'id': 1, 'email': email, 'role': role, 'full_name': full_name}}
                
        except Exception as e:
            return {'error': f'Registration failed: {str(e)}'}
    
    def sign_in(self, email: str, password: str, mfa_code: str = None) -> dict:
        """Authenticate user login using Supabase Auth with MFA support"""
        try:
            if self.supabase:
                # Use Supabase Auth to sign in
                result = self.supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                
                if result.user:
                    # Get user metadata for role
                    user_metadata = result.user.user_metadata or {}
                    role = user_metadata.get('role', 'user')
                    full_name = user_metadata.get('full_name', email.split('@')[0])
                    
                    user_data = {
                        'id': result.user.id,
                        'email': result.user.email,
                        'role': role,
                        'full_name': full_name
                    }
                    
                    token = self.generate_jwt_token(user_data)
                    return {'success': True, 'user': user_data, 'token': token}
                else:
                    return {'error': 'Invalid credentials'}
            else:
                # Fallback for demo mode
                if email == "admin@hranalytics.com" and password == "admin123":
                    user = {'id': 1, 'email': email, 'role': 'admin', 'full_name': 'Admin User'}
                    token = self.generate_jwt_token(user)
                    return {'success': True, 'user': user, 'token': token}
                elif email == "user@hranalytics.com" and password == "user123":
                    user = {'id': 2, 'email': email, 'role': 'user', 'full_name': 'Regular User'}
                    token = self.generate_jwt_token(user)
                    return {'success': True, 'user': user, 'token': token}
                else:
                    return {'error': 'Invalid credentials'}
                    
        except Exception as e:
            # Check if MFA is required
            error_msg = str(e)
            if 'MFA' in error_msg or 'mfa' in error_msg.lower():
                return {'mfa_required': True, 'error': 'MFA verification required', 'factor_id': getattr(e, 'factor_id', None)}
            else:
                return {'error': f'Login failed: {error_msg}'}
    
    def enroll_mfa(self, factor_type: str = "totp") -> dict:
        """Enroll user for Multi-Factor Authentication"""
        try:
            if not self.supabase:
                return {'error': 'Supabase not available'}
            
            result = self.supabase.auth.mfa.enroll({
                "factor_type": factor_type
            })
            
            if result.get('error'):
                return {'error': f'MFA enrollment failed: {result["error"]}'}
            
            data = result.get('data', {})
            
            if factor_type == "totp":
                # Generate QR code for TOTP
                qr_code_url = data.get('totp', {}).get('qr_code')
                secret = data.get('totp', {}).get('secret')
                
                if qr_code_url:
                    return {
                        'success': True,
                        'qr_code': qr_code_url,
                        'secret': secret,
                        'factor_id': data.get('id'),
                        'factor_type': factor_type
                    }
            
            return {
                'success': True,
                'factor_id': data.get('id'),
                'factor_type': factor_type,
                'data': data
            }
            
        except Exception as e:
            return {'error': f'MFA enrollment failed: {str(e)}'}
    
    def verify_mfa(self, factor_id: str, code: str) -> dict:
        """Verify MFA code during authentication"""
        try:
            if not self.supabase:
                return {'error': 'Supabase not available'}
            
            result = self.supabase.auth.mfa.verify({
                "factor_id": factor_id,
                "code": code
            })
            
            if result.get('error'):
                return {'error': f'MFA verification failed: {result["error"]}'}
            
            session = result.get('data', {}).get('session')
            if session:
                user = session.get('user')
                if user:
                    user_metadata = user.get('user_metadata', {})
                    user_data = {
                        'id': user['id'],
                        'email': user['email'],
                        'role': user_metadata.get('role', 'user'),
                        'full_name': user_metadata.get('full_name', user['email'].split('@')[0])
                    }
                    token = self.generate_jwt_token(user_data)
                    return {'success': True, 'user': user_data, 'token': token}
            
            return {'error': 'MFA verification failed'}
            
        except Exception as e:
            return {'error': f'MFA verification failed: {str(e)}'}
    
    def list_mfa_factors(self) -> dict:
        """List enrolled MFA factors for current user"""
        try:
            if not self.supabase:
                return {'error': 'Supabase not available'}
            
            result = self.supabase.auth.mfa.list_factors()
            
            if result.get('error'):
                return {'error': f'Failed to list MFA factors: {result["error"]}'}
            
            factors = result.get('data', [])
            return {'success': True, 'factors': factors}
            
        except Exception as e:
            return {'error': f'Failed to list MFA factors: {str(e)}'}
    
    def remove_mfa_factor(self, factor_id: str) -> dict:
        """Remove/unenroll an MFA factor"""
        try:
            if not self.supabase:
                return {'error': 'Supabase not available'}
            
            result = self.supabase.auth.mfa.unenroll({
                "factor_id": factor_id
            })
            
            if result.get('error'):
                return {'error': f'Failed to remove MFA factor: {result["error"]}'}
            
            return {'success': True, 'message': 'MFA factor removed successfully'}
            
        except Exception as e:
            return {'error': f'Failed to remove MFA factor: {str(e)}'}
    
    def generate_qr_code(self, secret: str, email: str) -> str:
        """Generate QR code for TOTP setup"""
        try:
            # Create TOTP URI
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=email,
                issuer_name="HR Analytics Pro"
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            # Create QR code image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            st.error(f"Failed to generate QR code: {str(e)}")
            return None
    
    def verify_totp_code(self, secret: str, code: str) -> bool:
        """Verify TOTP code locally"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except Exception:
            return False
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated"""
        if 'auth_token' in st.session_state:
            token_data = self.verify_jwt_token(st.session_state.auth_token)
            if 'error' not in token_data:
                st.session_state.user_data = token_data
                return True
            else:
                # Token expired or invalid
                self.sign_out()
        return False
    
    def sign_out(self):
        """Sign out user and clear session"""
        keys_to_clear = ['auth_token', 'user_data', 'user_role', 'user_email']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def require_auth(self, required_role: str = None):
        """Decorator to require authentication for pages"""
        if not self.check_authentication():
            st.error("🔒 Authentication required. Please sign in.")
            self.show_auth_form()
            st.stop()
        
        if required_role and st.session_state.user_data.get('role') != required_role:
            st.error("🚫 Insufficient permissions for this page.")
            st.stop()
    
    def show_auth_form(self):
        """Display authentication form"""
        st.markdown("""
        <div style="max-width: 400px; margin: 50px auto; padding: 30px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);">
            <h2 style="color: white; text-align: center; margin-bottom: 30px;">
                🏢 HR Analytics Pro
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔐 Sign In", "📝 Sign Up", "🔒 MFA Setup"])
        
        with tab1:
            # Check if MFA verification is needed
            if st.session_state.get('mfa_required'):
                self.show_mfa_verification_form()
            else:
                with st.form("signin_form"):
                    st.subheader("Welcome Back")
                    email = st.text_input("Email", placeholder="Enter your email")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                            if email and password:
                                result = self.sign_in(email, password)
                                if result.get('success'):
                                    st.session_state.auth_token = result['token']
                                    st.session_state.user_data = result['user']
                                    st.success("✅ Signed in successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                elif result.get('mfa_required'):
                                    st.session_state.mfa_required = True
                                    st.session_state.mfa_factor_id = result.get('factor_id')
                                    st.session_state.pending_email = email
                                    st.info("🔐 MFA verification required. Please enter your authentication code.")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('error', 'Login failed')}")
                            else:
                                st.error("Please fill in all fields")
                    
                    with col2:
                        if st.form_submit_button("Demo Login", use_container_width=True):
                            result = self.sign_in("admin@hranalytics.com", "admin123")
                            if result.get('success'):
                                st.session_state.auth_token = result['token']
                                st.session_state.user_data = result['user']
                                st.success("✅ Demo login successful!")
                                time.sleep(1)
                                st.rerun()
        
        with tab2:
            with st.form("signup_form"):
                st.subheader("Create Account")
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email", key="signup_email")
                password = st.text_input("Password", type="password", placeholder="Create a password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                role = st.selectbox("Role", ["user", "admin"], help="Select your role")
                
                if st.form_submit_button("Create Account", type="primary", use_container_width=True):
                    if all([full_name, email, password, confirm_password]):
                        if password == confirm_password:
                            result = self.sign_up(email, password, full_name, role)
                            if result.get('success'):
                                st.success("✅ Account created successfully! Please sign in.")
                            else:
                                st.error(f"❌ {result.get('error', 'Registration failed')}")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.error("Please fill in all fields")
        
        with tab3:
            self.show_mfa_setup_form()
        
        # Demo credentials info
        with st.expander("🔑 Demo Credentials"):
            st.info("""
            **Admin Account:**
            - Email: admin@hranalytics.com
            - Password: admin123
            
            **User Account:**
            - Email: user@hranalytics.com  
            - Password: user123
            """)
    
    def show_mfa_verification_form(self):
        """Display MFA verification form"""
        st.subheader("🔐 Multi-Factor Authentication")
        st.info("Please enter your authentication code to complete sign-in.")
        
        with st.form("mfa_verification_form"):
            mfa_code = st.text_input(
                "Authentication Code", 
                placeholder="Enter 6-digit code from your authenticator app",
                max_chars=6,
                help="Enter the 6-digit code from your authenticator app (Google Authenticator, Authy, etc.)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Verify Code", type="primary", use_container_width=True):
                    if mfa_code and len(mfa_code) == 6:
                        factor_id = st.session_state.get('mfa_factor_id')
                        if factor_id:
                            result = self.verify_mfa(factor_id, mfa_code)
                            if result.get('success'):
                                st.session_state.auth_token = result['token']
                                st.session_state.user_data = result['user']
                                # Clear MFA session state
                                for key in ['mfa_required', 'mfa_factor_id', 'pending_email']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.success("✅ MFA verification successful!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('error', 'MFA verification failed')}")
                        else:
                            st.error("❌ MFA session expired. Please sign in again.")
                    else:
                        st.error("Please enter a valid 6-digit code")
            
            with col2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    # Clear MFA session state
                    for key in ['mfa_required', 'mfa_factor_id', 'pending_email']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
    
    def show_mfa_setup_form(self):
        """Display MFA setup form"""
        st.subheader("🔒 Multi-Factor Authentication Setup")
        
        if not self.check_authentication():
            st.warning("⚠️ Please sign in first to set up MFA.")
            return
        
        # List existing MFA factors
        factors_result = self.list_mfa_factors()
        if factors_result.get('success'):
            factors = factors_result.get('factors', [])
            if factors:
                st.success(f"✅ You have {len(factors)} MFA factor(s) enrolled.")
                
                with st.expander("📱 Manage Existing MFA Factors"):
                    for factor in factors:
                        factor_type = factor.get('factor_type', 'Unknown')
                        factor_id = factor.get('id')
                        created_at = factor.get('created_at', '')
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{factor_type.upper()}** - Created: {created_at[:10] if created_at else 'Unknown'}")
                        with col2:
                            if st.button(f"Remove", key=f"remove_{factor_id}", type="secondary"):
                                result = self.remove_mfa_factor(factor_id)
                                if result.get('success'):
                                    st.success("✅ MFA factor removed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {result.get('error', 'Failed to remove MFA factor')}")
        
        # MFA enrollment section
        st.subheader("📲 Enroll New MFA Factor")
        
        factor_type = st.selectbox(
            "Authentication Method",
            ["totp", "email_otp"],
            format_func=lambda x: {
                "totp": "📱 Authenticator App (TOTP)",
                "email_otp": "📧 Email OTP",
            }.get(x, x),
            help="Choose your preferred authentication method"
        )
        
        if st.button("🔐 Setup MFA", type="primary"):
            with st.spinner("Setting up MFA..."):
                result = self.enroll_mfa(factor_type)
                
                if result.get('success'):
                    if factor_type == "totp":
                        st.success("✅ TOTP MFA enrollment successful!")
                        
                        # Display QR code
                        qr_code = result.get('qr_code')
                        secret = result.get('secret')
                        
                        if qr_code:
                            st.subheader("📱 Scan QR Code")
                            st.image(qr_code, caption="Scan this QR code with your authenticator app")
                        
                        if secret:
                            st.subheader("🔑 Manual Entry")
                            st.code(secret, language=None)
                            st.info("If you can't scan the QR code, manually enter this secret key in your authenticator app.")
                        
                        # Test verification
                        st.subheader("🧪 Test Your Setup")
                        with st.form("test_totp_form"):
                            test_code = st.text_input(
                                "Test Code", 
                                placeholder="Enter code from your authenticator app",
                                max_chars=6
                            )
                            if st.form_submit_button("Test Code"):
                                if secret and self.verify_totp_code(secret, test_code):
                                    st.success("✅ TOTP setup verified successfully!")
                                else:
                                    st.error("❌ Invalid code. Please try again.")
                    
                    elif factor_type == "email_otp":
                        st.success("✅ Email OTP MFA enrollment successful!")
                        st.info("📧 You will receive OTP codes via email during sign-in.")
                
                else:
                    st.error(f"❌ {result.get('error', 'MFA enrollment failed')}")

# Global auth manager instance
auth_manager = AuthManager()
