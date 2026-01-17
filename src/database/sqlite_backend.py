"""
SQLite implementation of the database backend.
Stores user preferences in a local SQLite database.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager

from .base import DatabaseBackend, UserPreferences


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend for user preferences."""
    
    def __init__(self, db_path: str = "user_data.db"):
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        self.initialize()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize(self) -> None:
        """Create the users table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    picture_url TEXT,
                    favorite_players TEXT DEFAULT '[]',
                    favorite_teams TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)
    
    def get_user(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences by user ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return UserPreferences(
                user_id=row["user_id"],
                email=row["email"],
                name=row["name"],
                picture_url=row["picture_url"],
                favorite_players=json.loads(row["favorite_players"]),
                favorite_teams=json.loads(row["favorite_teams"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
    
    def get_user_by_email(self, email: str) -> Optional[UserPreferences]:
        """Get user preferences by email."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return UserPreferences(
                user_id=row["user_id"],
                email=row["email"],
                name=row["name"],
                picture_url=row["picture_url"],
                favorite_players=json.loads(row["favorite_players"]),
                favorite_teams=json.loads(row["favorite_teams"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
    
    def create_or_update_user(self, user: UserPreferences) -> UserPreferences:
        """Create a new user or update existing user preferences."""
        now = datetime.now()
        user.updated_at = now
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # First check if user exists by user_id
            existing = self.get_user(user.user_id)
            
            # If not found by user_id, check by email (handles demo -> Google OAuth transition)
            if existing is None:
                existing_by_email = self.get_user_by_email(user.email)
                if existing_by_email:
                    # User exists with this email but different user_id
                    # Delete old record and create new one with new user_id
                    cursor.execute("DELETE FROM users WHERE email = ?", (user.email,))
                    existing = None  # Force creation of new user
            
            if existing:
                # Update existing user
                cursor.execute("""
                    UPDATE users SET
                        email = ?,
                        name = ?,
                        picture_url = ?,
                        favorite_players = ?,
                        favorite_teams = ?,
                        updated_at = ?
                    WHERE user_id = ?
                """, (
                    user.email,
                    user.name,
                    user.picture_url,
                    json.dumps(user.favorite_players),
                    json.dumps(user.favorite_teams),
                    now.isoformat(),
                    user.user_id,
                ))
            else:
                # Create new user
                user.created_at = now
                cursor.execute("""
                    INSERT INTO users (
                        user_id, email, name, picture_url,
                        favorite_players, favorite_teams,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id,
                    user.email,
                    user.name,
                    user.picture_url,
                    json.dumps(user.favorite_players),
                    json.dumps(user.favorite_teams),
                    user.created_at.isoformat(),
                    user.updated_at.isoformat(),
                ))
        
        return user
    
    def add_favorite_player(self, user_id: str, player_name: str) -> bool:
        """Add a player to user's favorites."""
        user = self.get_user(user_id)
        if user is None:
            return False
        
        if player_name in user.favorite_players:
            return False
        
        user.favorite_players.append(player_name)
        self.create_or_update_user(user)
        return True
    
    def remove_favorite_player(self, user_id: str, player_name: str) -> bool:
        """Remove a player from user's favorites."""
        user = self.get_user(user_id)
        if user is None:
            return False
        
        if player_name not in user.favorite_players:
            return False
        
        user.favorite_players.remove(player_name)
        self.create_or_update_user(user)
        return True
    
    def add_favorite_team(self, user_id: str, team_abbrev: str) -> bool:
        """Add a team to user's favorites."""
        user = self.get_user(user_id)
        if user is None:
            return False
        
        if team_abbrev in user.favorite_teams:
            return False
        
        user.favorite_teams.append(team_abbrev)
        self.create_or_update_user(user)
        return True
    
    def remove_favorite_team(self, user_id: str, team_abbrev: str) -> bool:
        """Remove a team from user's favorites."""
        user = self.get_user(user_id)
        if user is None:
            return False
        
        if team_abbrev not in user.favorite_teams:
            return False
        
        user.favorite_teams.remove(team_abbrev)
        self.create_or_update_user(user)
        return True
    
    def get_favorite_players(self, user_id: str) -> List[str]:
        """Get list of user's favorite players."""
        user = self.get_user(user_id)
        return user.favorite_players if user else []
    
    def get_favorite_teams(self, user_id: str) -> List[str]:
        """Get list of user's favorite teams."""
        user = self.get_user(user_id)
        return user.favorite_teams if user else []
    
    def set_favorite_players(self, user_id: str, players: List[str]) -> bool:
        """Set the entire list of favorite players (used for reordering)."""
        user = self.get_user(user_id)
        if user is None:
            return False
        user.favorite_players = players
        self.create_or_update_user(user)
        return True
    
    def set_favorite_teams(self, user_id: str, teams: List[str]) -> bool:
        """Set the entire list of favorite teams (used for reordering)."""
        user = self.get_user(user_id)
        if user is None:
            return False
        user.favorite_teams = teams
        self.create_or_update_user(user)
        return True
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# Singleton instance for easy access
_db_instance: Optional[SQLiteBackend] = None


def get_database(db_path: str = "user_data.db") -> SQLiteBackend:
    """Get or create the database singleton instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteBackend(db_path)
    return _db_instance
