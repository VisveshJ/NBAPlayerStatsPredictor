"""
Google OAuth authentication manager using official Google libraries.
Optimized for Streamlit Cloud followng the 'Minimal Pattern' for stability.
"""

import streamlit as st
import os
import requests
from typing import Optional, Dict, Any
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences

# Constants for OAuth
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

class AuthManager:
    """Manages Google OAuth using the official Google-auth-oauthlib flow."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json",
        cookie_name: str = "nba_predictor_auth",
        cookie_key: str = None,
        redirect_uri: str = "http://localhost:8501",
        cookie_expiry_days: int = 30,
    ):
        self._db: SQLiteBackend = get_database()
        self.credentials_path = credentials_path
        
        # Determine redirect_uri
        # On Cloud, we use the secret. Locally, we use localhost.
        self.redirect_uri = redirect_uri
        try:
            if "OAUTH_REDIRECT_URI" in st.secrets:
                self.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
        except Exception:
            pass

        # Load configuration from secrets or local file
        self.client_config = self._load_client_config()

    def _load_client_config(self) -> Optional[Dict[str, Any]]:
        """Load client configuration in the format Google libraries expect."""
        # Preferred: Lead from st.secrets (Cloud)
        try:
            if "google_auth" in st.secrets:
                raw_secrets = dict(st.secrets["google_auth"])
                # Ensure the 'web' wrapper exists
                if "web" in raw_secrets:
                    config = {"web": dict(raw_secrets["web"])}
                else:
                    config = {"web": raw_secrets}
                
                # Add default URIs if missing (required by google-auth-oauthlib)
                web = config["web"]
                if "auth_uri" not in web:
                    web["auth_uri"] = "https://accounts.google.com/o/oauth2/auth"
                if "token_uri" not in web:
                    web["token_uri"] = "https://oauth2.googleapis.com/token"
                if "auth_provider_x509_cert_url" not in web:
                    web["auth_provider_x509_cert_url"] = "https://www.googleapis.com/oauth2/v1/certs"
                
                return config
        except Exception:
            pass
            
        # Fallback: Load from local file
        if os.path.exists(self.credentials_path):
            import json
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        
        return None

    def _get_flow(self) -> Optional[Flow]:
        """Create a Google OAuth Flow instance."""
        if not self.client_config:
            return None
            
        try:
            # Create the flow from the dictionary config (no file needed!)
            flow = Flow.from_client_config(
                self.client_config,
                scopes=SCOPES,
                redirect_uri=self.redirect_uri
            )
            return flow
        except Exception as e:
            st.error(f"Flow creation failed: {e}")
            return None

    def check_authentication(self) -> bool:
        """
        Check if user is authenticated. 
        Handles the redirect 'code' exchange automatically.
        """
        # If already connected in session, we are good
        if st.session_state.get("connected", False):
            return True

        # Check for 'code' in query params (backward compatibility for older Streamlit)
        try:
            if hasattr(st, "query_params"):
                all_params = st.query_params
            else:
                all_params = st.experimental_get_query_params()
        except Exception:
            all_params = {}

        if "code" in all_params:
            # Handle list-style params from experimental_get_query_params
            code = all_params["code"]
            if isinstance(code, list):
                code = code[0]
                
            flow = self._get_flow()
            if flow:
                try:
                    # Exchange the code for a token
                    flow.fetch_token(code=code)
                    credentials = flow.credentials
                    
                    # Verify the ID Token and get user info
                    request = google_requests.Request()
                    id_info = id_token.verify_oauth2_token(
                        credentials.id_token, request, self.client_config["web"]["client_id"]
                    )
                    
                    # Store in session state
                    st.session_state["connected"] = True
                    st.session_state["user_info"] = {
                        "name": id_info.get("name"),
                        "email": id_info.get("email"),
                        "picture": id_info.get("picture"),
                    }
                    st.session_state["oauth_id"] = id_info.get("email") # Use email as ID
                    
                    # Sync to DB
                    self._sync_user_to_db()
                    
                    # Clear the code from URL
                    if hasattr(st, "query_params"):
                        st.query_params.clear()
                    else:
                        st.experimental_set_query_params()
                    return True
                except Exception as e:
                    st.error(f"Authentication failed during code exchange: {e}")
                    return False
        
        return st.session_state.get("demo_logged_in", False)

    def show_login_button(self) -> None:
        """Render the Google Login button or Demo login."""
        if self.is_authenticated():
            return

        flow = self._get_flow()
        if flow:
            try:
                # Generate the Authorization URL
                auth_url, _ = flow.authorization_url(prompt="consent")
                
                # Display standard branding
                st.markdown(f'<div style="text-align:center;"><a href="{auth_url}" target="_self" style="background-color: white; color: #757575; padding: 10px 24px; border-radius: 4px; text-decoration: none; font-family: Roboto; font-weight: 500; border: 1px solid #dadce0; display: inline-flex; align-items: center; gap: 10px;"><img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" width="20px"> Sign in with Google</a></div>', unsafe_allow_html=True)
                
                # Hidden diagnostics
                with st.expander("🔍 Auth Debug Info"):
                    st.write(f"Redirect URI: `{self.redirect_uri}`")
                    st.write(f"Client ID: `{self.client_config['web'].get('client_id')[:10]}...`")
            except Exception as e:
                st.error(f"Login button error: {e}")
                self._show_demo_login()
        else:
            self._show_demo_login()

    def _sync_user_to_db(self) -> None:
        """Sync authenticated user info to database."""
        user_info = st.session_state.get("user_info", {})
        oauth_id = st.session_state.get("oauth_id")
        
        if not oauth_id:
            return
        
        existing_user = self._db.get_user(oauth_id)
        if existing_user is None:
            new_user = UserPreferences(
                user_id=oauth_id,
                email=user_info.get("email", ""),
                name=user_info.get("name", "Unknown"),
                picture_url=user_info.get("picture"),
            )
            self._db.create_or_update_user(new_user)

    def _show_demo_login(self) -> None:
        """Fallback demo login."""
        st.info("� OAuth not configured or unavailable. Use demo login.")
        demo_name = st.text_input("Name:", key="demo_name_input")
        demo_email = st.text_input("Email:", key="demo_email_input")
        
        if st.button("🚀 Demo Login", type="primary"):
            if demo_name and demo_email:
                st.session_state["demo_logged_in"] = True
                st.session_state["connected"] = True
                st.session_state["oauth_id"] = f"demo_{demo_email}"
                st.session_state["user_info"] = {"name": demo_name, "email": demo_email}
                self._sync_user_to_db()
                st.rerun()

    def logout(self) -> None:
        """Log out and clear session."""
        for key in ["demo_logged_in", "connected", "oauth_id", "user_info"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    def is_authenticated(self) -> bool:
        return st.session_state.get("connected", False)

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get("user_info", {})

    def get_favorite_players(self) -> list:
        oid = st.session_state.get("oauth_id")
        return self._db.get_favorite_players(oid) if oid else []

    def get_favorite_teams(self) -> list:
        oid = st.session_state.get("oauth_id")
        return self._db.get_favorite_teams(oid) if oid else []

    def add_favorite_player(self, player_name: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.add_favorite_player(oid, player_name) if oid else False

    def remove_favorite_player(self, player_name: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.remove_favorite_player(oid, player_name) if oid else False

    def add_favorite_team(self, team_abbrev: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.add_favorite_team(oid, team_abbrev) if oid else False

    def remove_favorite_team(self, team_abbrev: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.remove_favorite_team(oid, team_abbrev) if oid else False

# Singleton
_auth_manager: Optional[AuthManager] = None

def get_auth_manager(**kwargs) -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(**kwargs)
    return _auth_manager
