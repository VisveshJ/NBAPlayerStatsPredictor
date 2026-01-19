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
        
        # Safely check secrets to avoid error if no secrets.toml exists locally
        redirect_from_secrets = None
        try:
            if "OAUTH_REDIRECT_URI" in st.secrets:
                redirect_from_secrets = st.secrets["OAUTH_REDIRECT_URI"]
        except Exception:
            pass

        if redirect_from_secrets:
            self.redirect_uri = redirect_from_secrets
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
            
            # Use local file if it exists, otherwise use secrets
            auth_in_secrets = False
            try:
                auth_in_secrets = "google_auth" in st.secrets
            except Exception:
                pass

            if not os.path.exists(config_path) and auth_in_secrets:
                try:
                    def deep_dict(obj):
                        if hasattr(obj, "to_dict"):
                            return deep_dict(obj.to_dict())
                        if isinstance(obj, dict):
                            return {k: deep_dict(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [deep_dict(v) for v in obj]
                        return obj

                    raw_secrets = deep_dict(st.secrets["google_auth"])
                    
                    # Normalize structure to the 'web' format required by the Google library
                    if "web" in raw_secrets:
                        auth_payload = {"web": raw_secrets["web"]}
                    elif "installed" in raw_secrets:
                        auth_payload = {"installed": raw_secrets["installed"]}
                    else:
                        auth_payload = {"web": raw_secrets}
                    
                    key = "web" if "web" in auth_payload else "installed"
                    
                    # Normalize project_id if it exists
                    if "project_id" in auth_payload[key]:
                        # Some Google libraries fail if this isn't strictly lowercase
                        auth_payload[key]["project_id"] = str(auth_payload[key]["project_id"]).lower()
                    
                    # SYNC redirect_uris: Append the current one if missing, but keep others
                    current_uris = auth_payload[key].get("redirect_uris", [])
                    if isinstance(current_uris, str):
                        current_uris = [current_uris]
                    
                    if self.redirect_uri not in current_uris:
                        current_uris.append(self.redirect_uri)
                    
                    auth_payload[key]["redirect_uris"] = current_uris

                    # Persistence: Save to /tmp
                    persistent_path = "/tmp/google_credentials_st.json"
                    with open(persistent_path, 'w') as f:
                        json.dump(auth_payload, f)
                    config_path = persistent_path
                    
                except Exception as e:
                    st.sidebar.error(f"Error preparing secrets: {e}")
                    return None
            
            # Fallback check if file still missing
            if not os.path.exists(config_path):
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
            # Ensure session keys exist
            for key in ["connected", "user_info", "oauth_id"]:
                if key not in st.session_state:
                    st.session_state[key] = False if key == "connected" else None

            auth.check_authentification()
        except:
            st.session_state["connected"] = False
            return False
        
        if st.session_state.get("connected", False):
            self._sync_user_to_db()
            return True
        return False

    def show_login_button(self) -> None:
        """Render the Google Login button or Demo login."""
        auth = self._get_authenticator()
        
        if not self.is_authenticated():
            st.caption(f"🔧 **Handshake URI:** `{self.redirect_uri}`")

        if auth is None:
            self._show_demo_login()
            return
        
        try:
            auth.login()
        except Exception as e:
            st.error(f"📉 **OAuth Error:** {e}")
            st.info("💡 **Quick Fix:** Ensure the Redirect URI above exactly matches your Google Cloud Console.")
            if st.button("Use Demo Login"):
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
        """Show demo login window."""
        st.markdown("### 🔐 Login")
        
        # Troubleshooter
        with st.expander("🔍 Debugging"):
            auth_found = False
            try:
                auth_found = 'google_auth' in st.secrets
            except Exception:
                pass
            st.write(f"Library: {GOOGLE_AUTH_AVAILABLE}")
            st.write(f"Secrets Found: {auth_found}")
            st.write(f"Active URI: {self.redirect_uri}")
        
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
