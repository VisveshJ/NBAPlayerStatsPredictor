"""
Google OAuth authentication wrapper using st-google-auth.
Modern, file-less implementation designed for Streamlit Cloud.
"""

import streamlit as st
from typing import Optional, Dict, Any
import os

# Import the modern library
try:
    from st_google_auth import st_google_auth
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences


class AuthManager:
    """Manages Google OAuth authentication using st-google-auth."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json", # Kept for signature compatibility
        cookie_name: str = "nba_predictor_auth",
        cookie_key: str = None,
        redirect_uri: str = None,
        cookie_expiry_days: int = 30,
    ):
        self._db: SQLiteBackend = get_database()
        
        # Load Client ID and Secret from secrets
        self.client_id = None
        self.client_secret = None
        
        if "google_auth" in st.secrets:
            auth_secrets = st.secrets["google_auth"]
            # Check for nested structure or flat structure
            if "web" in auth_secrets:
                self.client_id = auth_secrets["web"].get("client_id")
                self.client_secret = auth_secrets["web"].get("client_secret")
            else:
                self.client_id = auth_secrets.get("client_id")
                self.client_secret = auth_secrets.get("client_secret")
                
    def check_authentication(self) -> bool:
        """
        Check if user is authenticated.
        This handles the OAuth handshake seamlessly.
        """
        if not GOOGLE_AUTH_AVAILABLE:
            return st.session_state.get("demo_logged_in", False)
            
        if not self.client_id or not self.client_secret:
            # Fallback to demo if secrets are missing
            return st.session_state.get("demo_logged_in", False)
            
        # Use st-google-auth to handle login
        # This will automatically render the button if needed and handle the callback
        user_info = st_google_auth(
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        
        if user_info:
            st.session_state["connected"] = True
            st.session_state["user_info"] = user_info
            # Use email as oauth_id for consistency
            st.session_state["oauth_id"] = user_info.get("email")
            self._sync_user_to_db()
            return True
            
        return st.session_state.get("demo_logged_in", False)

    def show_login_button(self) -> None:
        """
        No-op in st-google-auth because check_authentication handles rendering.
        We provide a fallback demo login if secrets are missing.
        """
        if not self.client_id or not self.client_secret:
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
                name=user_info.get("name", user_info.get("email", "Unknown")),
                picture_url=user_info.get("picture"),
            )
            self._db.create_or_update_user(new_user)
        else:
            existing_user.name = user_info.get("name", existing_user.name)
            existing_user.picture_url = user_info.get("picture", existing_user.picture_url)
            self._db.create_or_update_user(existing_user)

    def _show_demo_login(self) -> None:
        """Show demo login for development."""
        st.markdown("### 🔐 Login")
        st.info("⚠️ Demo Mode: Google OAuth not configured.")
        
        demo_name = st.text_input("Enter your name:", key="demo_name_input")
        demo_email = st.text_input("Enter your email:", key="demo_email_input")
        
        if st.button("🚀 Demo Login", type="primary"):
            if demo_name and demo_email:
                st.session_state["demo_logged_in"] = True
                st.session_state["connected"] = True
                st.session_state["oauth_id"] = f"demo_{demo_email}"
                st.session_state["user_info"] = {
                    "name": demo_name,
                    "email": demo_email,
                    "picture": None,
                }
                self._sync_user_to_db()
                st.rerun()

    def logout(self) -> None:
        """Log out the current user."""
        # Simple session clear for st-google-auth
        for key in ["demo_logged_in", "connected", "oauth_id", "user_info"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    def is_authenticated(self) -> bool:
        return st.session_state.get("connected", False)

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        return st.session_state.get("user_info", {})

    def get_favorite_players(self) -> list:
        oauth_id = st.session_state.get("oauth_id")
        return self._db.get_favorite_players(oauth_id) if oauth_id else []

    def get_favorite_teams(self) -> list:
        oauth_id = st.session_state.get("oauth_id")
        return self._db.get_favorite_teams(oauth_id) if oauth_id else []
    
    # ... Wrapper methods for DB interactions ...
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
