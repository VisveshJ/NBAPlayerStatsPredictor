"""
Same-tab Google OAuth Manager for Streamlit Cloud.
Forced same-window login using JavaScript redirect to break out of iframes.
Includes all database methods required by app.py.
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

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

class AuthManager:
    """Minimized, same-tab OAuth handler."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json",
        **kwargs
    ):
        self._db: SQLiteBackend = get_database()
        self.credentials_path = credentials_path
        
        # 🔒 Strict Redirect URI (Must match Google Console EXACTLY)
        self.redirect_uri = st.secrets.get("OAUTH_REDIRECT_URI", "https://nbaplayerpredictor.streamlit.app").rstrip("/")

    # ---------- Config & Flow Construction ----------

    def _get_config(self) -> Optional[Dict[str, Any]]:
        """Safely load and sanitize client configuration from Secrets."""
        if "google_auth" in st.secrets:
            raw = dict(st.secrets["google_auth"])
            inner = raw.get("web", raw)
            
            return {
                "web": {
                    "client_id": str(inner.get("client_id", "")).strip(),
                    "client_secret": str(inner.get("client_secret", "")).strip(),
                    "project_id": str(inner.get("project_id", "")).strip().lower(),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
        
        if os.path.exists(self.credentials_path):
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        return None

    def _get_flow(self) -> Optional[Flow]:
        config = self._get_config()
        if not config:
            return None
        return Flow.from_client_config(
            config, 
            scopes=SCOPES, 
            redirect_uri=self.redirect_uri
        )

    # ---------- Authentication Life Cycle ----------

    def check_authentication(self) -> bool:
        """Passive check/callback handler executed at the top of app.py."""
        if st.session_state.get("connected", False):
            return True

        # Process the 'code' from query parameters (the OAuth callback)
        params = st.query_params
        if "code" in params:
            flow = self._get_flow()
            if not flow: return False
            
            try:
                code = params["code"]
                if isinstance(code, list): code = code[0]
                
                # Exchange code for tokens
                flow.fetch_token(code=code)
                creds = flow.credentials
                
                # Validate user identity
                request = google_requests.Request()
                id_info = id_token.verify_oauth2_token(
                    creds.id_token, 
                    request, 
                    self._get_config()["web"]["client_id"]
                )
                
                # Populate session state
                email = id_info.get("email")
                st.session_state["connected"] = True
                st.session_state["user_info"] = {
                    "name": id_info.get("name"),
                    "email": email,
                    "picture": id_info.get("picture"),
                }
                st.session_state["oauth_id"] = email
                
                self._sync_user_to_db()
                
                # Cleanup URL and refresh app state
                st.query_params.clear()
                st.rerun()
                return True
            except Exception as e:
                st.error(f"Login failed: {e}")
                return False
                
        return st.session_state.get("demo_logged_in", False)

    def start_login(self) -> None:
        """JS-based redirect to force 'Same Tab' even inside Streamlit Cloud's iframe."""
        flow = self._get_flow()
        if not flow:
            st.error("OAuth Configuration Missing!")
            return

        auth_url, _ = flow.authorization_url(prompt="select_account")
        
        # window.top.location.href is required to break out of the Streamlit iframe
        st.markdown(f"""
            <script>
                window.top.location.href = "{auth_url}";
            </script>
            <div style="text-align: center; padding: 20px;">
                <p>Redirecting to Google Secure Login...</p>
                <a href="{auth_url}" target="_self" style="color: #FF6B35;">Click here if not redirected automatically</a>
            </div>
            """, unsafe_allow_html=True)
        st.stop()

    def show_login_button(self) -> None:
        """Render the login UI."""
        if self.is_authenticated():
            return

        st.markdown("### 🏀 NBA Predictor Login")
        st.write("Sign in to save your favorite players and unlock AI stats.")

        if st.button("🚀 Sign in with Google", type="primary", use_container_width=True):
            self.start_login()

    def logout(self) -> None:
        """Clear session and redirect."""
        for k in ["connected", "oauth_id", "user_info", "demo_logged_in"]:
            st.session_state.pop(k, None)
        st.query_params.clear()
        st.rerun()

    # ---------- Shared Application Logic (Required by app.py) ----------

    def is_authenticated(self) -> bool:
        return st.session_state.get("connected", False)

    def get_user_info(self) -> dict:
        return st.session_state.get("user_info", {})

    def get_current_user(self):
        """Helper for app.py that expects attribute-based access (current_user.email)."""
        info = self.get_user_info()
        if not info: return None
        class User:
            def __init__(self, d):
                self.email = d.get("email")
                self.name = d.get("name")
                self.picture = d.get("picture")
        return User(info)

    def _sync_user_to_db(self) -> None:
        oid = st.session_state.get("oauth_id")
        info = self.get_user_info()
        if oid:
            if not self._db.get_user(oid):
                self._db.create_or_update_user(UserPreferences(
                    user_id=oid,
                    email=info.get("email", ""),
                    name=info.get("name", "Unknown"),
                    picture_url=info.get("picture")
                ))

    # ---------- Favorites Management (Required by app.py) ----------

    def get_favorite_players(self) -> list:
        oid = st.session_state.get("oauth_id")
        return self._db.get_favorite_players(oid) if oid else []

    def get_favorite_teams(self) -> list:
        oid = st.session_state.get("oauth_id")
        return self._db.get_favorite_teams(oid) if oid else []

    def add_favorite_player(self, name: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.add_favorite_player(oid, name) if oid else False

    def remove_favorite_player(self, name: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.remove_favorite_player(oid, name) if oid else False

    def add_favorite_team(self, abbrev: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.add_favorite_team(oid, abbrev) if oid else False

    def remove_favorite_team(self, abbrev: str) -> bool:
        oid = st.session_state.get("oauth_id")
        return self._db.remove_favorite_team(oid, abbrev) if oid else False

    def reorder_favorite_player(self, player_name: str, direction: str) -> bool:
        """Handles moving players up/down in the favorites list."""
        oid = st.session_state.get("oauth_id")
        if not oid: return False
        
        current_list = self.get_favorite_players()
        if player_name not in current_list: return False
        
        idx = current_list.index(player_name)
        if direction == "up" and idx > 0:
            current_list[idx], current_list[idx-1] = current_list[idx-1], current_list[idx]
        elif direction == "down" and idx < len(current_list) - 1:
            current_list[idx], current_list[idx+1] = current_list[idx+1], current_list[idx]
        else:
            return False
            
        return self._db.set_favorite_players(oid, current_list)

def get_auth_manager():
    """Singleton instance manager."""
    if "auth_manager_instance" not in st.session_state:
        st.session_state["auth_manager_instance"] = AuthManager()
    return st.session_state["auth_manager_instance"]
