"""
Database abstraction layer for user preferences.
Provides a base class that can be extended for different database backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserPreferences:
    """User preferences data model."""
    user_id: str
    email: str
    name: str
    picture_url: Optional[str] = None
    favorite_players: List[str] = None
    favorite_teams: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.favorite_players is None:
            self.favorite_players = []
        if self.favorite_teams is None:
            self.favorite_teams = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "picture_url": self.picture_url,
            "favorite_players": self.favorite_players,
            "favorite_teams": self.favorite_teams,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database (create tables, etc.)."""
        pass
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences by user ID."""
        pass
    
    @abstractmethod
    def create_or_update_user(self, user: UserPreferences) -> UserPreferences:
        """Create a new user or update existing user preferences."""
        pass
    
    @abstractmethod
    def add_favorite_player(self, user_id: str, player_name: str) -> bool:
        """Add a player to user's favorites. Returns True if added, False if already exists."""
        pass
    
    @abstractmethod
    def remove_favorite_player(self, user_id: str, player_name: str) -> bool:
        """Remove a player from user's favorites. Returns True if removed."""
        pass
    
    @abstractmethod
    def add_favorite_team(self, user_id: str, team_abbrev: str) -> bool:
        """Add a team to user's favorites. Returns True if added, False if already exists."""
        pass
    
    @abstractmethod
    def remove_favorite_team(self, user_id: str, team_abbrev: str) -> bool:
        """Remove a team from user's favorites. Returns True if removed."""
        pass
    
    @abstractmethod
    def get_favorite_players(self, user_id: str) -> List[str]:
        """Get list of user's favorite players."""
        pass
    
    @abstractmethod
    def get_favorite_teams(self, user_id: str) -> List[str]:
        """Get list of user's favorite teams."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass
