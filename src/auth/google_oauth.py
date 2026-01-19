"""
Expert-level Google OAuth Manager.
Handles Streamlit Cloud proxy mismatches, state persistence, and strict token exchange.
"""

import streamlit as st
import os
import json
import requests
from typing import Optional, Dict, Any
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences

# Core OAuth Scopes
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

class AuthManager:
    """Production-grade OAuth handler for Streamlit Cloud."""
    
    def __init__(
        self,
        credentials_path: str = "google_credentials.json",
        cookie_name: str = "nba_predictor_auth",
        cookie_key: str = None,
        redirect_uri: str = "http://localhost:8501",
        **kwargs
    ):
        self._db: SQLiteBackend = get_database()
        self.credentials_path = credentials_path
        
        # 1. Determine Public Redirect URI
        self.redirect_uri = redirect_uri
        try:
            # Prioritize Streamlit Secrets for Cloud
            if "OAUTH_REDIRECT_URI" in st.secrets:
                self.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
        except:
            pass
        
        # Ensure no trailing slash for the base lib config
        if self.redirect_uri.endswith("/"):
            self.redirect_uri = self.redirect_uri[:-1]

    def _get_config(self) -> Optional[Dict[str, Any]]:
        """Safely load and sanitize client configuration."""
        config = None
        
        # Try Secrets first
        try:
            if "google_auth" in st.secrets:
                raw = dict(st.secrets["google_auth"])
                inner = raw.get("web", raw)
                
                # SANITIZE: Strip hidden spaces/newlines from IDs and Secrets
                config = {
                    "web": {
                        "client_id": str(inner.get("client_id", "")).strip(),
                        "client_secret": str(inner.get("client_secret", "")).strip(),
                        "project_id": str(inner.get("project_id", "")).strip().lower(),
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    }
                }
        except:
            pass
            
        # Fallback to local file
        if not config and os.path.exists(self.credentials_path):
            with open(self.credentials_path, 'r') as f:
                config = json.load(f)
                
        return config

    def _get_flow(self, current_url: str = None) -> Optional[Flow]:
        """Initialize the OAuth flow with strict URI matching."""
        config = self._get_config()
        if not config:
            return None
            
        try:
            # Use the detected current URL or the configured redirect URI
            target_uri = current_url or self.redirect_uri
            
            return Flow.from_client_config(
                config,
                scopes=SCOPES,
                redirect_uri=target_uri
            )
        except Exception as e:
            st.error(f"Failed to initialize OAuth Flow: {e}")
            return None

    def check_authentication(self) -> bool:
        """Passive and active check for authentication."""
        if st.session_state.get("connected", False):
            return True

        # Handle Redirect Callback
        try:
            # Streamlit 1.30+ query_params handling
            params = st.query_params
        except:
            params = st.experimental_get_query_params()

        if "code" in params:
            # We strictly use the base URL for the exchange to match the proxy
            flow = self._get_flow()
            if not flow:
                return False
                
            try:
                # Exchange code for tokens
                code = params["code"]
                if isinstance(code, list): code = code[0]
                
                flow.fetch_token(code=code)
                creds = flow.credentials
                
                # Validate user identity
                request = google_requests.Request()
                id_info = id_token.verify_oauth2_token(
                    creds.id_token, request, flow.client_id
                )
                
                # Populate session
                st.session_state["connected"] = True
                st.session_state["user_info"] = {
                    "name": id_info.get("name"),
                    "email": id_info.get("email"),
                    "picture": id_info.get("picture"),
                }
                st.session_state["oauth_id"] = id_info.get("email")
                
                self._sync_user_to_db()
                
                # Cleanup URL
                st.query_params.clear()
                st.rerun()
                return True
            except Exception as e:
                st.error("📉 **OAuth Handshake Failed**")
                st.code(f"Technical Reason: {str(e)}")
                st.info("💡 Tip: If this is a redirect_uri_mismatch, ensure your Google Console has the HTTPS protocol enabled.")
                return False
                
        return st.session_state.get("demo_logged_in", False)

    def show_login_button(self) -> None:
        """Render the login UI."""
        if self.is_authenticated():
            return

        flow = self._get_flow()
        if not flow:
            self._show_demo_login()
            return

        try:
            auth_url, _ = flow.authorization_url(prompt="select_account")
            
            st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <a href="{auth_url}" target="_self" style="
                        background-color: white; 
                        color: #3c4043; 
                        padding: 12px 24px; 
                        border-radius: 4px; 
                        text-decoration: none; 
                        font-family: 'Google Sans',Roboto,Arial,sans-serif;
                        font-weight: 500; 
                        border: 1px solid #dadce0; 
                        display: inline-flex; 
                        align-items: center; 
                        gap: 12px;
                        box-shadow: 0 1px 3px rgba(60,64,67,0.3);
                        transition: background-color 0.2s;
                    ">
                        <img src="https://fonts.gstatic.com/s/i/productlogos/googleg/v6/24px.svg" width="20px">
                        Sign in with Google
                    </a>
                    <p style="margin-top: 15px; color: #9CA3AF; font-size: 0.8rem;">
                        Secure authentication via Google Cloud Project: <code>{flow.client_id[:15]}...</code>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("�️ Connection Manager"):
                st.write(f"📡 **Outbound Redirect:** `{self.redirect_uri}`")
                st.write(f"🔑 **Client ID Status:** {'Detected' if flow.client_id else 'Missing'}")
                if st.button("Switch to Demo Mode"):
                    self._show_demo_login()
                    
        except Exception as e:
            st.error(f"UI Error: {e}")
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
        """Demo login for bypass."""
        st.markdown("---")
        st.subheader("🚀 Demo Access")
        name = st.text_input("Username:", key="demo_u")
        email = st.text_input("Email:", key="demo_e")
        if st.button("Enter as Guest"):
            st.session_state["demo_logged_in"] = True
            st.session_state["connected"] = True
            st.session_state["oauth_id"] = f"demo_{email}"
            st.session_state["user_info"] = {"name": name, "email": email}
            self._sync_user_to_db()
            st.rerun()

    def logout(self) -> None:
        """Reset session."""
        keys = ["demo_logged_in", "connected", "oauth_id", "user_info"]
        for k in keys:
            if k in st.session_state: del st.session_state[k]
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
