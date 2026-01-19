"""
Google OAuth authentication wrapper for Streamlit.
Provides easy-to-use functions for login/logout and user management.
"""

import streamlit as st
from typing import Optional, Dict, Any
import os

# Try to import streamlit-google-auth
try:
    from streamlit_google_auth import Authenticate
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from src.database.sqlite_backend import get_database, SQLiteBackend
from src.database.base import UserPreferences


class AuthManager:
    """Manages Google OAuth authentication and user sessions."""
    
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
        # 1. Passed argument
        # 2. st.secrets["OAUTH_REDIRECT_URI"]
        # 3. Environment variable
        # 4. Default localhost
        if redirect_uri == "http://localhost:8501":
            if "OAUTH_REDIRECT_URI" in st.secrets:
                self.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
            else:
                self.redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", redirect_uri)
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
            
            # Check if credentials file exists
            if not os.path.exists(config_path):
                # Try to load from st.secrets if on Streamlit Cloud
                # Priority: google_auth.web -> google_auth
                auth_payload = None
                
                if "google_auth" in st.secrets:
                    try:
                        import json
                        import tempfile
                        
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
                        
                        # Handle different nesting styles
                        if "web" in raw_secrets:
                            auth_payload = {"web": raw_secrets["web"]}
                        elif "installed" in raw_secrets:
                            auth_payload = {"installed": raw_secrets["installed"]}
                        elif "client_id" in raw_secrets:
                            # User provided flat secrets under [google_auth]
                            auth_payload = {"web": raw_secrets}
                        else:
                            st.sidebar.error("❌ 'google_auth' found but missing 'client_id' or 'web' section.")
                            return None
                        
                        # Normalize redirect_uris to be a list
                        key = "web" if "web" in auth_payload else "installed"
                        if "redirect_uris" in auth_payload[key]:
                            if isinstance(auth_payload[key]["redirect_uris"], str):
                                auth_payload[key]["redirect_uris"] = [auth_payload[key]["redirect_uris"]]
                        else:
                            auth_payload[key]["redirect_uris"] = []
                        
                        # Inject current redirect_uri if missing
                        if self.redirect_uri not in auth_payload[key]["redirect_uris"]:
                            auth_payload[key]["redirect_uris"].append(self.redirect_uri)

                        # Diagnostics (Masked)
                        masked_payload = deep_dict(auth_payload)
                        m_key = "web" if "web" in masked_payload else "installed"
                        if "client_secret" in masked_payload[m_key]:
                            masked_payload[m_key]["client_secret"] = "********"
                        if "client_id" in masked_payload[m_key]:
                            cid = masked_payload[m_key]["client_id"]
                            masked_payload[m_key]["client_id"] = f"{cid[:10]}...{cid[-10:]}" if len(cid) > 20 else "********"
                        
                        st.sidebar.caption("🔐 OAuth Config Loaded")
                        with st.sidebar.expander("Show Final JSON (Masked)"):
                            st.json(masked_payload)

                        # Write to temp file
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
                        json.dump(auth_payload, tmp_file)
                        tmp_file.close()
                        config_path = tmp_file.name
                        
                    except Exception as e:
                        st.sidebar.error(f"Error preparing OAuth secrets: {e}")
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
                st.sidebar.error(f"Authenticate Init Error: {e}")
                return None
        
        return self._authenticator
    
    def check_authentication(self) -> bool:
        """
        Check if user is authenticated. 
        Returns True if authenticated, False otherwise.
        """
        # Ensure session keys
        for key in ["connected", "user_info", "oauth_id"]:
            if key not in st.session_state:
                st.session_state[key] = False if key == "connected" else ({} if key == "user_info" else None)
        
        auth = self._get_authenticator()
        if auth is None:
            return st.session_state.get("demo_logged_in", False)
        
        try:
            auth.check_authentification()
        except:
            st.session_state["connected"] = False
            return False
        
        if st.session_state.get("connected", False):
            self._sync_user_to_db()
            return True
        return False

    def show_login_button(self) -> None:
        """Display the Google login button."""
        auth = self._get_authenticator()
        
        if not self.is_authenticated():
            st.caption(f"🔧 **Redirect URI:** `{self.redirect_uri}`")

        if auth is None:
            self._show_demo_login()
            return
        
        try:
            auth.login()
        except Exception as e:
            st.error("📉 **OAuth Connection Failed**")
            st.warning(f"Error Detail: {e}")
            st.info("💡 **Tip:** Verify that your Redirect URI matches what you configured in the Google Cloud Console.")
            if st.button("Try Demo Mode Instead"):
                self._show_demo_login()
    
    def _show_demo_login(self) -> None:
        """Show demo login for development without Google OAuth."""
        st.markdown("### 🔐 Login")
        st.info("⚠️ Demo Mode: Google OAuth not configured. Using demo login.")
        
        # Add debug info to help user configure Google Console
        st.caption(f"🔧 **Debug Info:** Redirect URI configured as: `{self.redirect_uri}`")
        
        demo_name = st.text_input("Enter your name:", key="demo_name_input")
        demo_email = st.text_input("Enter your email:", key="demo_email_input")
        
        if st.button("🚀 Demo Login", type="primary"):
            if demo_name and demo_email:
                # Create demo session
                st.session_state["demo_logged_in"] = True
                st.session_state["connected"] = True
                st.session_state["oauth_id"] = f"demo_{demo_email}"
                st.session_state["user_info"] = {
                    "name": demo_name,
                    "email": demo_email,
                    "picture": None,
                }
                
                # Sync to database
                self._sync_user_to_db()
                st.rerun()
            else:
                st.error("Please enter both name and email.")
    
    def logout(self) -> None:
        """Log out the current user."""
        auth = self._get_authenticator()
        
        if auth is not None:
            auth.logout()
        
        # Clear demo session too
        for key in ["demo_logged_in", "connected", "oauth_id", "user_info"]:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()
    
    def get_current_user(self) -> Optional[UserPreferences]:
        """Get the current logged-in user's preferences from database."""
        oauth_id = st.session_state.get("oauth_id")
        
        if not oauth_id:
            return None
        
        return self._db.get_user(oauth_id)
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get basic user info from session (not from DB)."""
        if not self.is_authenticated():
            return None
        
        return st.session_state.get("user_info", {})
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return st.session_state.get("connected", False)
    
    def add_favorite_player(self, player_name: str) -> bool:
        """Add a player to current user's favorites."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
        return self._db.add_favorite_player(oauth_id, player_name)
    
    def remove_favorite_player(self, player_name: str) -> bool:
        """Remove a player from current user's favorites."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
        return self._db.remove_favorite_player(oauth_id, player_name)
    
    def add_favorite_team(self, team_abbrev: str) -> bool:
        """Add a team to current user's favorites."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
        return self._db.add_favorite_team(oauth_id, team_abbrev)
    
    def remove_favorite_team(self, team_abbrev: str) -> bool:
        """Remove a team from current user's favorites."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
        return self._db.remove_favorite_team(oauth_id, team_abbrev)
    
    def get_favorite_players(self) -> list:
        """Get current user's favorite players."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return []
        return self._db.get_favorite_players(oauth_id)
    
    def get_favorite_teams(self) -> list:
        """Get current user's favorite teams."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return []
        return self._db.get_favorite_teams(oauth_id)
    
    def reorder_favorite_player(self, player_name: str, direction: str) -> bool:
        """Move a player up or down in the favorites list."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
        
        players = self.get_favorite_players()
        if player_name not in players:
            return False
        
        idx = players.index(player_name)
        if direction == "up" and idx > 0:
            players[idx], players[idx-1] = players[idx-1], players[idx]
        elif direction == "down" and idx < len(players) - 1:
            players[idx], players[idx+1] = players[idx+1], players[idx]
        else:
            return False
            
        return self._db.set_favorite_players(oauth_id, players)
        
    def reorder_favorite_team(self, team_abbrev: str, direction: str) -> bool:
        """Move a team up or down in the favorites list."""
        oauth_id = st.session_state.get("oauth_id")
        if not oauth_id:
            return False
            
        teams = self.get_favorite_teams()
        if team_abbrev not in teams:
            return False
            
        idx = teams.index(team_abbrev)
        if direction == "up" and idx > 0:
            teams[idx], teams[idx-1] = teams[idx-1], teams[idx]
        elif direction == "down" and idx < len(teams) - 1:
            teams[idx], teams[idx+1] = teams[idx+1], teams[idx]
        else:
            return False
            
        return self._db.set_favorite_teams(oauth_id, teams)


# Singleton instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager(**kwargs) -> AuthManager:
    """Get or create the auth manager singleton."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager(**kwargs)
    return _auth_manager
