"""
Google OAuth authentication wrapper using streamlit-google-auth.
Robust implementation handling dictionary-to-file conversion for Streamlit Cloud.
"""

import streamlit as st
from typing import Optional, Dict, Any
import os
import json

# Import the library
try:
    from streamlit_google_auth import Authenticate
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences


class AuthManager:
    """Manages Google OAuth authentication using streamlit-google-auth."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json",
        cookie_name: str = "nba_predictor_auth",
        cookie_key: str = None,
        redirect_uri: str = "http://localhost:8501",
        cookie_expiry_days: int = 30,
    ):
        self.credentials_path = credentials_path
        self.cookie_name = cookie_name
        self.cookie_key = cookie_key or os.environ.get("OAUTH_COOKIE_KEY", "default-secret-key-change-me")
        
        # Priority for redirect_uri:
        # 1. Force use of OAUTH_REDIRECT_URI from secrets if it exists (for Cloud)
        # 2. Argument passed to constructor
        # 3. Default localhost
        if "OAUTH_REDIRECT_URI" in st.secrets:
            self.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
        else:
            self.redirect_uri = redirect_uri
            
        self.cookie_expiry_days = cookie_expiry_days
        self._authenticator = None
        self._db: SQLiteBackend = get_database()
        
    def _get_authenticator(self) -> Optional["Authenticate"]:
        """Get or create the authenticator instance."""
        if not GOOGLE_AUTH_AVAILABLE:
            return None
        
        if self._authenticator is None:
            config_path = self.credentials_path
            
            # Check if local file exists
            if not os.path.exists(config_path):
                # If not, try loading from Secrets and creating a persistent temp file
                if "google_auth" in st.secrets:
                    try:
                        # Deep convert secrets to dict
                        def deep_dict(obj):
                            if hasattr(obj, "to_dict"):
                                return deep_dict(obj.to_dict())
                            if isinstance(obj, dict):
                                return {k: deep_dict(v) for k, v in obj.items()}
                            if isinstance(obj, list):
                                return [deep_dict(v) for v in obj]
                            return obj

                        raw_secrets = deep_dict(st.secrets["google_auth"])
                        
                        # MANDATORY: Wrap in 'web' if missing
                        if "web" not in raw_secrets and "installed" not in raw_secrets:
                            auth_payload = {"web": raw_secrets}
                        else:
                            auth_payload = raw_secrets
                        
                        # Ensure redirect_uris is a list
                        key = "web" if "web" in auth_payload else "installed"
                        if "redirect_uris" in auth_payload[key]:
                            if isinstance(auth_payload[key]["redirect_uris"], str):
                                auth_payload[key]["redirect_uris"] = [auth_payload[key]["redirect_uris"]]
                        else:
                            auth_payload[key]["redirect_uris"] = []
                            
                        # Ensure current redirect_uri is included exactly as sent to Google
                        if self.redirect_uri not in auth_payload[key]["redirect_uris"]:
                            auth_payload[key]["redirect_uris"].append(self.redirect_uri)

                        # Use a fixed path in /tmp for persistence across script reruns
                        # This is important for the OAuth callback phase
                        persistent_path = "/tmp/google_credentials_st.json"
                        with open(persistent_path, 'w') as f:
                            json.dump(auth_payload, f)
                        config_path = persistent_path
                        
                    except Exception as e:
                        st.sidebar.error(f"Error preparing secrets: {e}")
                        return None
                else:
                    return None
            
            try:
                self._authenticator = Authenticate(
                    secret_credentials_path=config_path,
                    cookie_name=self.cookie_name,
                    cookie_key=self.cookie_key,
                    redirect_uri=self.redirect_uri,
                    cookie_expiry_days=self.cookie_expiry_days,
                )
            except Exception as e:
                st.sidebar.error(f"Init Error: {e}")
                return None
        
        return self._authenticator

    def check_authentication(self) -> bool:
        """Check if user is authenticated."""
        auth = self._get_authenticator()
        if auth is None:
            return st.session_state.get("demo_logged_in", False)
        
        try:
            # Initialize library's required keys
            for key in ["connected", "user_info", "oauth_id"]:
                if key not in st.session_state:
                    st.session_state[key] = False if key == "connected" else None

            # Run the library's check
            auth.check_authentification()
        except Exception as e:
            # Silently fail check (common during login flow)
            return False
        
        if st.session_state.get("connected", False):
            self._sync_user_to_db()
            return True
        return False

    def show_login_button(self) -> None:
        """Render the Google Login button or Demo login."""
        auth = self._get_authenticator()
        
        if not self.is_authenticated():
            st.caption(f"🔧 **Redirect URI Path:** `{self.redirect_uri}`")

        if auth is None:
            self._show_demo_login()
            return
        
        try:
            auth.login()
        except Exception as e:
            st.error(f"📉 **Login Handshake Failed:** {e}")
            if st.button("Try Demo Mode"):
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
        """Show demo login for development."""
        st.markdown("### 🔐 Login")
        st.info("⚠️ Demo Mode / OAuth Loading...")
        
        # Diagnostics
        with st.expander("🔍 System Status"):
            st.write(f"Library: {GOOGLE_AUTH_AVAILABLE}")
            st.write(f"Secrets: {'google_auth' in st.secrets}")
            st.write(f"Redirect: {self.redirect_uri}")
        
        demo_name = st.text_input("Enter your name:", key="demo_name_input")
        demo_email = st.text_input("Enter your email:", key="demo_email_input")
        
        if st.button("🚀 Demo Login", type="primary"):
            if demo_name and demo_email:
                st.session_state["demo_logged_in"] = True
                st.session_state["connected"] = True
                st.session_state["oauth_id"] = f"demo_{demo_email}"
                st.session_state["user_info"] = {"name": demo_name, "email": demo_email}
                self._sync_user_to_db()
                st.rerun()

    def logout(self) -> None:
        auth = self._get_authenticator()
        if auth:
            auth.logout()
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

# Singleton
_auth_manager: Optional[AuthManager] = None

def get_auth_manager(**kwargs) -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(**kwargs)
    return _auth_manager
