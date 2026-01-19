"""
Final Precision Google OAuth Manager for NBA Stats Predictor.
Forced production matching and sanitized state handling for Streamlit Cloud.
"""

import streamlit as st
import os
import json
from typing import Optional, Dict, Any
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences

# Use only the essential scopes to avoid 'Unverified App' blocks in production
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

class AuthManager:
    """Minimized, high-precision OAuth handler."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json",
        redirect_uri: Optional[str] = None,
        **kwargs
    ):
        self._db: SQLiteBackend = get_database()
        self.credentials_path = credentials_path
        
        # Determine URI: Smart detection
        # 1. Try to get the redirect URI from Streamlit Secrets (Cloud priority)
        try:
            if "OAUTH_REDIRECT_URI" in st.secrets:
                self.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
            else:
                self.redirect_uri = "https://nbaplayerpredictor.streamlit.app"
        except:
            self.redirect_uri = "https://nbaplayerpredictor.streamlit.app"
            
        # 2. If we are running locally, override with localhost
        # We detect local by checking if the passed arg or current environment suggests it
        is_local = False
        if redirect_uri and ("localhost" in redirect_uri or "127.0.0.1" in redirect_uri):
            is_local = True
        
        if is_local:
            self.redirect_uri = "http://localhost:8501"
            
        # Ensure NO trailing slash
        self.redirect_uri = self.redirect_uri.rstrip("/")

    def _get_config(self) -> Optional[Dict[str, Any]]:
        """Sanitize secrets with strict matching."""
        try:
            if "google_auth" in st.secrets:
                raw = dict(st.secrets["google_auth"])
                inner = raw.get("web", raw)
                return {
                    "web": {
                        "client_id": str(inner.get("client_id", "")).strip(),
                        "client_secret": str(inner.get("client_secret", "")).strip(),
                        "project_id": str(inner.get("project_id", "nbaappproject")).lower(),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }
        except: pass
        
        if os.path.exists(self.credentials_path):
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        return None

    def _get_flow(self) -> Optional[Flow]:
        config = self._get_config()
        if not config: return None
        try:
            return Flow.from_client_config(config, scopes=SCOPES, redirect_uri=self.redirect_uri)
        except: return None

    def check_authentication(self) -> bool:
        """Passive and active check for authentication."""
        # Check for logout trigger
        if st.session_state.get("logout_now", False):
            self.logout()
            return False

        # Existing session check
        if st.session_state.get("connected", False):
            return True

        # Handle Query Params (Callback)
        try:
            params = st.query_params
        except:
            params = st.experimental_get_query_params()

        if "code" in params:
            flow = self._get_flow()
            if not flow: return False
            try:
                code = params["code"]
                if isinstance(code, list): code = code[0]
                
                # Exchange code for tokens
                flow.fetch_token(code=code)
                creds = flow.credentials
                
                # Fetch identity from the fresh token
                request = google_requests.Request()
                config = self._get_config()
                client_id = config["web"]["client_id"] if config else None
                
                id_info = id_token.verify_oauth2_token(creds.id_token, request, client_id)
                
                # STRICT: Ensure we are writing to the session fresh
                new_email = id_info.get("email")
                
                st.session_state["connected"] = True
                st.session_state["user_info"] = {
                    "name": id_info.get("name"),
                    "email": new_email,
                    "picture": id_info.get("picture"),
                }
                st.session_state["oauth_id"] = new_email
                
                # Sync fresh user data to DB
                self._sync_user_to_db()
                
                # Clear handshake data from URL
                st.query_params.clear()
                st.rerun()
                return True
            except Exception as e:
                st.error(f"Authentication Sync Failed: {e}")
                return False
                
        return st.session_state.get("demo_logged_in", False)

    def show_login_button(self) -> None:
        if self.is_authenticated(): return

        flow = self._get_flow()
        if not flow:
            self._show_demo_login()
            return

        try:
            # Reverting to st.link_button which handles security headers correctly (fixes 403)
            auth_url, _ = flow.authorization_url(prompt="select_account")
            
            st.markdown("### 🏀 NBA Predictor Login")
            st.write("Join to save your favorites and get AI insights.")
            
            # Use native link button
            st.link_button("🚀 Sign in with Google", auth_url, type="primary", use_container_width=True)
        except:
            self._show_demo_login()

    def _sync_user_to_db(self) -> None:
        user_info = st.session_state.get("user_info", {})
        oid = st.session_state.get("oauth_id")
        if oid:
            existing = self._db.get_user(oid)
            if not existing:
                new_user = UserPreferences(
                    user_id=oid,
                    email=user_info.get("email", ""),
                    name=user_info.get("name", "Unknown"),
                    picture_url=user_info.get("picture")
                )
                self._db.create_or_update_user(new_user)

    def _show_demo_login(self) -> None:
        st.subheader("Guest Login")
        name = st.text_input("Name:", key="d_n")
        email = st.text_input("Email:", key="d_e")
        if st.button("Log in as Guest"):
            st.session_state["demo_logged_in"] = True
            st.session_state["connected"] = True
            st.session_state["oauth_id"] = f"demo_{email}"
            st.session_state["user_info"] = {"name": name, "email": email}
            self._sync_user_to_db()
            st.rerun()

    def logout(self) -> None:
        """Total wipe of all session and authentication data."""
        keys_to_clear = [
            "demo_logged_in", "connected", "oauth_id", "user_info", 
            "logout_now"
        ]
        for k in keys_to_clear:
            if k in st.session_state: 
                del st.session_state[k]
        
        # Also clear query params just in case
        st.query_params.clear()
        st.rerun()

    def is_authenticated(self) -> bool:
        return st.session_state.get("connected", False)

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get("user_info", {})

    def get_current_user(self):
        """Compatibility method for app.py that expects attribute access."""
        info = self.get_user_info()
        if not info:
            return None
        # Return a simple object that allows .email access
        class User:
            def __init__(self, d):
                self.email = d.get("email")
                self.name = d.get("name")
                self.picture = d.get("picture")
        return User(info)

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

# In Streamlit, singleton managers should be cached per session to avoid data leaks
def get_auth_manager(**kwargs):
    if "auth_manager_instance" not in st.session_state:
        st.session_state["auth_manager_instance"] = AuthManager(**kwargs)
    return st.session_state["auth_manager_instance"]
