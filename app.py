"""
NBA Live Stats Predictor - Main Application Entry Point
Features Google OAuth authentication, personalized user experience, and full HMM predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import time
import warnings
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguestandings, teamgamelog
import matplotlib.pyplot as plt
import json
import os
import pytz
from datetime import datetime, timezone, timedelta

# Import our custom modules
from src.auth.google_oauth import get_auth_manager
from src.ui.components import (
    apply_dark_theme,
    render_user_profile_sidebar,
    render_player_card,
    render_team_card,
    render_empty_state,
    render_section_header,
    render_welcome_header,
    render_login_page,
)
from src.logic.model import (
    train_hmm_with_drtg, predict_with_drtg, 
    ensure_minimum_transitions, calculate_player_consistency
)

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NBA Live Stats Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
apply_dark_theme()

# ==================== AUTHENTICATION ====================
auth = get_auth_manager()

# Check authentication status
is_authenticated = auth.check_authentication()


# Team abbreviation mapping used across the app
TEAM_ABBREV_MAP = {
    'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', # Specific matches first
    'Los Angeles': 'LAL', 'Golden State': 'GSW', 'Phoenix': 'PHX', 'Denver': 'DEN',
    'Memphis': 'MEM', 'Sacramento': 'SAC', 'Dallas': 'DAL', 'New Orleans': 'NOP',
    'LA': 'LAC', 'Minnesota': 'MIN', 'Oklahoma City': 'OKC', 'Portland': 'POR',
    'Utah': 'UTA', 'San Antonio': 'SAS', 'Houston': 'HOU',
    'Boston': 'BOS', 'Milwaukee': 'MIL', 'Philadelphia': 'PHI', 'Cleveland': 'CLE',
    'New York': 'NYK', 'Brooklyn': 'BKN', 'Miami': 'MIA', 'Atlanta': 'ATL',
    'Chicago': 'CHI', 'Toronto': 'TOR', 'Indiana': 'IND', 'Washington': 'WAS',
    'Orlando': 'ORL', 'Charlotte': 'CHA', 'Detroit': 'DET'
}

# Mapping of abbreviations to full team names for UI display
TEAM_NAME_MAP = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

def get_team_abbrev(city):
    """Get team abbreviation from city name."""
    for key in TEAM_ABBREV_MAP:
        if key in city:
            return TEAM_ABBREV_MAP[key]
    return city[:3].upper()


def format_pct(val):
    """Format percentage value: show '100.0' for 100%, otherwise 'X.X%'."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        val_f = float(val)
        # Use a small epsilon for floating point comparison
        if abs(val_f - 100.0) < 0.001:
            return "100.0"
        return f"{val_f:.1f}%"
    except (ValueError, TypeError):
        return str(val)

def get_streak_color(streak_text):
    """Determine color for streak text (W in green, L in red)."""
    if not streak_text or streak_text == 'N/A':
        return "#FAFAFA"
    if 'W' in str(streak_text).upper():
        return "#10B981"  # Green
    if 'L' in str(streak_text).upper():
        return "#EF4444"  # Red
    return "#FAFAFA"

def get_record_color(record_text):
    """Determine color for record text (Winning in green, Losing in red)."""
    if not record_text or record_text == 'N/A' or '-' not in str(record_text):
        return "#FAFAFA"
    try:
        parts = str(record_text).split('-')
        if len(parts) >= 2:
            w = int(parts[0])
            l = int(parts[1])
            if w > l:
                return "#10B981"  # Green
            if l > w:
                return "#EF4444"  # Red
    except:
        pass
    return "#FAFAFA"


def get_local_now():
    """Get the current time in the user's selected timezone."""
    tz_name = st.session_state.get('user_timezone', 'US/Pacific')
    try:
        return datetime.now(pytz.timezone(tz_name))
    except:
        return datetime.now(pytz.timezone('US/Pacific'))


# ==================== NAVIGATION HELPERS ====================
def nav_to_compare(p1, p2):
    """Callback to navigate to Compare Players page and load two specific players."""
    st.session_state.current_page = "Compare Players"
    st.session_state.nav_radio = "Compare Players"
    st.session_state.compare_player1_search = p1
    st.session_state.compare_player2_search = p2
    st.session_state.compare_select1 = p1
    st.session_state.compare_select2 = p2
    st.session_state.trigger_compare = True

def nav_to_player_stats(player_name):
    """Callback to navigate to Player Stats page for a specific player."""
    st.session_state.current_page = "Player Stats"
    st.session_state.nav_radio = "Player Stats"
    st.session_state.player_stats_search = player_name

def nav_to_predictions(player_name):
    """Callback to navigate to Predictions page for a specific player."""
    st.session_state.current_page = "Predictions"
    st.session_state.nav_radio = "Predictions"
    st.session_state.redirect_to_predictions = player_name
    st.session_state.auto_load_player = player_name

# ==================== DATA FETCHING FUNCTIONS ====================
@st.cache_data(ttl=3600)
def get_current_defensive_ratings(season="2025-26"):
    """Fetch current defensive ratings for all NBA teams with retry logic."""
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries with exponential backoff
            if attempt > 0:
                time.sleep(2 * attempt)  # 2s, 4s delay
            
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame"
            )
            
            df = team_stats.get_data_frames()[0]
            all_teams = teams.get_teams()
            valid_nba_teams = {t['abbreviation'] for t in all_teams}
            
            team_name_to_abbrev = {}
            for t in all_teams:
                team_name_to_abbrev[t['full_name']] = t['abbreviation']
                team_name_to_abbrev[t['nickname']] = t['abbreviation']
                team_name_to_abbrev[t['abbreviation']] = t['abbreviation']
            
            team_name_to_abbrev['LA Clippers'] = 'LAC'
            team_def_ratings = {}
            team_col = None
            for col_name in ['TEAM_ABBREVIATION', 'TEAM_NAME', 'Team']:
                if col_name in df.columns:
                    team_col = col_name
                    break
            
            if team_col and 'DEF_RATING' in df.columns:
                for _, row in df.iterrows():
                    team_value = row[team_col]
                    
                    if team_value in valid_nba_teams:
                        team_abbrev = team_value
                    else:
                        team_abbrev = team_name_to_abbrev.get(team_value)
                    
                    if team_abbrev and team_abbrev in valid_nba_teams:
                        team_def_ratings[team_abbrev] = round(row['DEF_RATING'], 1)
            
            if len(team_def_ratings) < 30:
                league_avg = np.mean(list(team_def_ratings.values())) if team_def_ratings else 112.0
                for team in valid_nba_teams:
                    if team not in team_def_ratings:
                        team_def_ratings[team] = round(league_avg, 1)
            
            return team_def_ratings
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                continue  # Try again
    
    # All retries failed
    st.error(f"Error fetching defensive ratings after {max_retries} attempts: {str(last_error)}")
    return {}


@st.cache_data(ttl=3600)
def get_team_ratings_with_ranks(season="2025-26"):
    """Get offensive and defensive ratings with league rankings for all teams."""
    try:
        time.sleep(0.6)
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame"
        )
        
        df = team_stats.get_data_frames()[0]
        
        # Find team abbreviation column
        all_teams = teams.get_teams()
        team_name_to_abbrev = {}
        for t in all_teams:
            team_name_to_abbrev[t['full_name']] = t['abbreviation']
            team_name_to_abbrev[t['nickname']] = t['abbreviation']
        team_name_to_abbrev['LA Clippers'] = 'LAC'
        
        # Build ratings dict with rankings
        result = {}
        
        # Sort by offensive rating (descending - higher is better) and assign ranks
        df_sorted_off = df.sort_values('OFF_RATING', ascending=False).reset_index(drop=True)
        df_sorted_off['OFF_RANK'] = range(1, len(df_sorted_off) + 1)
        
        # Sort by defensive rating (ascending - lower is better) and assign ranks
        df_sorted_def = df.sort_values('DEF_RATING', ascending=True).reset_index(drop=True)
        df_sorted_def['DEF_RANK'] = range(1, len(df_sorted_def) + 1)
        
        # Merge ranks back
        for _, row in df.iterrows():
            team_name = row.get('TEAM_NAME', '')
            abbrev = team_name_to_abbrev.get(team_name)
            if not abbrev:
                continue
                
            off_rtg = round(row['OFF_RATING'], 1)
            def_rtg = round(row['DEF_RATING'], 1)
            
            # Find ranks
            off_rank = df_sorted_off[df_sorted_off['TEAM_NAME'] == team_name]['OFF_RANK'].values
            def_rank = df_sorted_def[df_sorted_def['TEAM_NAME'] == team_name]['DEF_RANK'].values
            
            result[abbrev] = {
                'off_rtg': off_rtg,
                'def_rtg': def_rtg,
                'off_rank': int(off_rank[0]) if len(off_rank) > 0 else 0,
                'def_rank': int(def_rank[0]) if len(def_rank) > 0 else 0
            }
        
        return result
    except Exception as e:
        return {}



def get_league_standings(season="2025-26"):
    """Fetch current NBA standings for all teams."""
    try:
        time.sleep(0.6)
        standings = leaguestandings.LeagueStandings(
            season=season,
            season_type="Regular Season"
        )
        
        df = standings.get_data_frames()[0]
        
        # Get relevant columns
        standings_data = []
        for _, row in df.iterrows():
            team_data = {
                'TeamID': row.get('TeamID'),
                'TeamAbbrev': get_team_abbrev(row.get('TeamCity', '')),
                'TeamName': row.get('TeamName', ''),
                'TeamCity': row.get('TeamCity', ''),
            }
            # Fix Clippers name for display consistency
            if team_data['TeamCity'] == 'LA' and team_data['TeamName'] == 'Clippers':
                team_data['TeamCity'] = 'Los Angeles Clippers'
                team_data['TeamName'] = 'Clippers'

            team_data.update({
                'Conference': row.get('Conference', ''),
                'Division': row.get('Division', ''),
                'ConferenceRecord': row.get('ConferenceRecord', ''),
                'DivisionRecord': row.get('DivisionRecord', ''),
                'DivisionRank': row.get('DivisionRank', 0),
                'PlayoffRank': row.get('PlayoffRank', 0),
                'Wins': row.get('WINS', 0),
                'Losses': row.get('LOSSES', 0),
                'WinPct': row.get('WinPCT', 0),
                'Record': f"{row.get('WINS', 0)}-{row.get('LOSSES', 0)}",
                'HOME': row.get('HOME', ''),
                'ROAD': row.get('ROAD', ''),
                'L10': row.get('L10', ''),
                'strCurrentStreak': str(row.get('strCurrentStreak', '')).replace(' ', ''),
                'PointsPG': row.get('PointsPG', 0),
                'OppPointsPG': row.get('OppPointsPG', 0),
                'GB': row.get('ConferenceGamesBack', '-'),
            })
            standings_data.append(team_data)
        
        return pd.DataFrame(standings_data)
    except Exception as e:
        st.error(f"Error fetching standings: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_nba_schedule():
    """Fetch full NBA schedule from CDN."""
    import requests
    from datetime import datetime
    
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        league_schedule = data.get('leagueSchedule', {})
        game_dates = league_schedule.get('gameDates', [])
        
        for game_date in game_dates:
            date_str = game_date.get('gameDate', '')
            for game in game_date.get('games', []):
                games.append({
                    'game_date': date_str,
                    'game_id': game.get('gameId'),
                    'home_team': game.get('homeTeam', {}).get('teamTricode', ''),
                    'away_team': game.get('awayTeam', {}).get('teamTricode', ''),
                    'home_team_name': game.get('homeTeam', {}).get('teamName', ''),
                    'away_team_name': game.get('awayTeam', {}).get('teamName', ''),
                    'game_status': game.get('gameStatus', 1),  # 1=scheduled, 2=in progress, 3=final
                    'game_time_utc': game.get('gameDateTimeUTC', ''),  # UTC time
                    'broadcasters': game.get('broadcasters', {})
                })
        
        return games
    except Exception as e:
        return []


@st.cache_data(ttl=60)  # Short TTL for live scores
def get_todays_scoreboard():
    """Fetch today's scoreboard with live/final scores."""
    import requests
    
    try:
        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        scoreboard = {}
        games = data.get('scoreboard', {}).get('games', [])
        
        for game in games:
            game_id = game.get('gameId')
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            scoreboard[game_id] = {
                'game_status': game.get('gameStatus', 1),  # 1=scheduled, 2=in progress, 3=final
                'game_status_text': game.get('gameStatusText', ''),
                'home_score': home_team.get('score', 0),
                'away_score': away_team.get('score', 0),
                'home_team': home_team.get('teamTricode', ''),
                'away_team': away_team.get('teamTricode', ''),
                'period': game.get('period', 0),
                'game_clock': game.get('gameClock', ''),
            }
        
        return scoreboard
    except Exception as e:
        return {}

def get_team_upcoming_games(team_abbrev, schedule, standings_df, num_games=5):
    """Get upcoming games for a specific team."""
    from datetime import datetime
    
    if not schedule:
        return []
    
    today = get_local_now().date()
    
    # Build playoff rank lookup from standings
    team_ranks = {}
    for _, row in standings_df.iterrows():
        # Map city to abbreviation
        city = row.get('TeamCity', '')
        abbrev = get_team_abbrev(city)
        if abbrev:
            team_ranks[abbrev] = {
                'rank': int(row.get('PlayoffRank', 0)),
                'name': row.get('TeamName', ''),
                'conference': row.get('Conference', ''),
                'record': row.get('Record', '')
            }
    
    upcoming = []
    for game in schedule:
        # Parse game date
        try:
            game_date_str = game['game_date']
            # Format: "01/14/2026 12:00:00 AM" or similar
            game_date = datetime.strptime(game_date_str.split(' ')[0], '%m/%d/%Y').date()
        except:
            continue
        
        # Only include future games (not played yet)
        if game_date < today:
            continue
        if game['game_status'] == 3:  # Already finished
            continue
        
        # Check if team is playing
        is_home = game['home_team'] == team_abbrev
        is_away = game['away_team'] == team_abbrev
        
        if is_home or is_away:
            opponent = game['away_team'] if is_home else game['home_team']
            opponent_info = team_ranks.get(opponent, {'rank': 0, 'name': opponent, 'conference': '', 'record': ''})
            
            upcoming.append({
                'date': game_date.strftime('%b %d'),
                'opponent': opponent,
                'opponent_name': opponent_info['name'],
                'opponent_rank': opponent_info['rank'],
                'opponent_conference': opponent_info['conference'],
                'opponent_record': opponent_info['record'],
                'is_home': is_home,
            })
            
            if len(upcoming) >= num_games:
                break
    
    return upcoming


def get_todays_games(schedule, standings_df, tz=None):
    """Get games scheduled for today with team seeds."""
    from datetime import datetime
    
    if not schedule:
        return []
    
    today = get_local_now().date()
    
    # Build playoff rank lookup from standings
    team_ranks = {}
    for _, row in standings_df.iterrows():
        abbrev = get_team_abbrev(row.get('TeamCity', ''))
        if abbrev:
            team_ranks[abbrev] = {
                'rank': int(row.get('PlayoffRank', 0)),
                'name': row.get('TeamName', ''),
                'conference': row.get('Conference', ''),
                'record': row.get('Record', ''),
                'streak': row.get('strCurrentStreak', '')
            }
    
    todays_games = []
    for game in schedule:
        try:
            game_date_str = game['game_date']
            game_date = datetime.strptime(game_date_str.split(' ')[0], '%m/%d/%Y').date()
        except:
            continue
        
        if game_date == today:
            home_team = game['home_team']
            away_team = game['away_team']
            home_info = team_ranks.get(home_team, {'rank': 0, 'name': home_team, 'conference': '', 'streak': ''})
            away_info = team_ranks.get(away_team, {'rank': 0, 'name': away_team, 'conference': '', 'streak': ''})
            
            # Parse game time and convert to local timezone
            game_time_local = ""
            try:
                game_time_utc = game.get('game_time_utc', '')
                if game_time_utc:
                    # Parse UTC time
                    from datetime import timezone
                    utc_dt = datetime.strptime(game_time_utc, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                    # Convert to local timezone
                    target_tz = tz if tz else pytz.timezone(st.session_state.get('user_timezone', 'US/Pacific'))
                    local_dt = utc_dt.astimezone(target_tz)
                    game_time_local = local_dt.strftime('%I:%M %p').lstrip('0')
            except:
                pass
            
            # Extract channel/broadcaster
            broadcasters = game.get('broadcasters', {})
            channel = ""
            if broadcasters:
                natl = broadcasters.get('nationalBroadcasters', [])
                if natl:
                    channel = natl[0].get('broadcasterAbbreviation', '')
                else:
                    home_tv = broadcasters.get('homeTvBroadcasters', [])
                    if home_tv:
                        channel = home_tv[0].get('broadcasterAbbreviation', '')
                    else:
                        away_tv = broadcasters.get('awayTvBroadcasters', [])
                        if away_tv:
                            channel = away_tv[0].get('broadcasterAbbreviation', '')

            todays_games.append({
                'game_id': game.get('game_id'),
                'home_team': home_team,
                'away_team': away_team,
                'home_name': home_info['name'],
                'away_name': away_info['name'],
                'home_rank': home_info['rank'],
                'away_rank': away_info['rank'],
                'home_conference': home_info['conference'],
                'away_conference': away_info['conference'],
                'home_record': home_info.get('record', ''),
                'away_record': away_info.get('record', ''),
                'home_streak': home_info.get('streak', ''),
                'away_streak': away_info.get('streak', ''),
                'game_status': game.get('game_status', 1),
                'game_time': game_time_local,
                'game_time_sort': game.get('game_time_utc', ''),  # For sorting
                'channel': channel
            })
    
    # Sort games by time (earliest to latest)
    todays_games.sort(key=lambda x: x.get('game_time_sort', 'Z'))
    
    return todays_games


@st.cache_data(ttl=3600)
def get_team_roster(team_abbrev):
    """Fetch roster for a team."""
    from nba_api.stats.endpoints import commonteamroster
    
    try:
        # Get team ID from abbreviation
        all_teams = teams.get_teams()
        team = [t for t in all_teams if t['abbreviation'] == team_abbrev]
        
        if not team:
            return None
        
        team_id = team[0]['id']
        
        time.sleep(0.6)
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        
        df = roster.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        # Return relevant columns
        roster_data = df[['PLAYER', 'NUM', 'POSITION', 'HEIGHT', 'WEIGHT', 'AGE']].copy()
        roster_data.columns = ['Player', '#', 'Pos', 'Ht', 'Wt', 'Age']
        
        return roster_data
        
    except Exception as e:
        return None


@st.cache_data(ttl=1800)
def get_team_game_log(team_abbrev, season="2025-26", num_games=5):
    """Fetch recent game log for a team with PLUS_MINUS for score calculation."""
    try:
        # Get team ID from abbreviation
        all_teams = teams.get_teams()
        team = [t for t in all_teams if t['abbreviation'] == team_abbrev]
        
        if not team:
            return None
        
        team_id = team[0]['id']
        
        time.sleep(0.6)
        # Use leaguegamefinder instead of teamgamelog to get PLUS_MINUS
        from nba_api.stats.endpoints import leaguegamefinder
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Regular Season"
        )
        
        df = gamefinder.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        # Rename GAME_DATE to match expected format
        df['GAME_DATE'] = df['GAME_DATE']
        
        # Return last N games
        return df.head(num_games)
    except Exception as e:
        return None


def get_team_logo_url(team_abbrev):
    """Get the NBA team logo URL from official NBA CDN."""
    # NBA team IDs for logo URLs
    team_ids = {
        "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
        "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
        "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
        "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
        "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
        "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
        "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
        "UTA": 1610612762, "WAS": 1610612764
    }
    
    team_id = team_ids.get(team_abbrev)
    if team_id:
        return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    return None


def render_team_logo_html(logo_url, size=44, padding=8):
    """Return HTML for a clean team logo without background circle."""
    if not logo_url:
        return ""
    
    return f"""
    <div style="display: flex; align-items: center; justify-content: center; padding-top: {padding}px;">
        <img src="{logo_url}" style="width: {size}px; height: {size}px; filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));">
    </div>
    """


import unicodedata

def normalize_name(name):
    """Normalize name to handle accents (e.g. Dončić -> Doncic)."""
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

def get_player_photo_url(player_name):
    """Get the official NBA player headshot URL from CDN."""
    try:
        all_players = players.get_players()
        # Normalize both input and database names for comparison
        norm_input = normalize_name(player_name).lower()
        player = [p for p in all_players if normalize_name(p['full_name']).lower() == norm_input]
        
        if player:
            player_id = player[0]['id']
            return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    except:
        pass
    return None

@st.cache_data(ttl=2000) # Increased TTL to force cache invalidation on deployment and distinct from previous
def get_player_game_log(player_name, season="2025-26"):
    """Fetch game log for a player from the current season."""
    all_players = players.get_players()
    norm_input = normalize_name(player_name).lower()
    player = [p for p in all_players if normalize_name(p['full_name']).lower() == norm_input]
    
    if not player:
        return None, None
    
    player_id = str(player[0]['id'])
    
    try:
        time.sleep(0.6)
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            return None, None
        
        player_team = df['MATCHUP'].iloc[0][:3]
        
        df['Opponent'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
        df = df.rename(columns={
            'PTS': 'Points', 'AST': 'Assists', 'REB': 'Rebounds',
            'STL': 'Steals', 'BLK': 'Blocks', 'TOV': 'Turnovers',
            'FGM': 'FGM', 'FGA': 'FGA', 'FG_PCT': 'FG%',
            'FG3M': '3PM', 'FG3A': '3PA', 'FG3_PCT': '3P%',
            'FTM': 'FTM', 'FTA': 'FTA', 'FT_PCT': 'FT%',
            'MIN': 'MIN', 'PF': 'PF', 'WL': 'W/L'
        })
        
        df['TS%'] = df.apply(lambda row: 
            round(row['Points'] / (2 * (row['FGA'] + 0.44 * row['FTA'])) * 100, 1) 
            if (row['FGA'] + 0.44 * row['FTA']) > 0 else 0, 
            axis=1
        )
        
        # Create combined columns for display
        df['FG'] = df.apply(lambda row: f"{row['FGM']}/{row['FGA']}", axis=1)
        df['3P'] = df.apply(lambda row: f"{row['3PM']}/{row['3PA']}", axis=1)
        df['FT'] = df.apply(lambda row: f"{row['FTM']}/{row['FTA']}", axis=1)
        
        df = df.iloc[::-1].reset_index(drop=True)
        return df, player_team
        
    except Exception as e:
        st.error(f"Error fetching game log: {str(e)}")
        return None, None


@st.cache_data(ttl=3600)
def get_bulk_player_stats(season="2025-26"):
    """Fetch stats for ALL players in one go to optimize loading."""
    from nba_api.stats.endpoints import leaguedashplayerstats
    try:
        # PerGame for averages
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame"
        )
        df = stats.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching bulk stats: {e}")
        return None


@st.cache_data(ttl=86400)
def abbreviate_position(pos, player_name=None):
    """Abbreviate position names for UI compact display."""
    # Special case for Victor Wembanyama
    if player_name and "Wembanyama" in player_name:
        return "C"
        
    if not pos:
        return ""
    pos = pos.strip()
    
    # Specific type mappings (if endpoint provides them)
    if "Point Guard" in pos or pos == "PG": return "PG"
    if "Shooting Guard" in pos or pos == "SG": return "SG"
    if "Small Forward" in pos or pos == "SF": return "SF"
    if "Power Forward" in pos or pos == "PF": return "PF"
    
    # Standard mappings with preference for G/F and F/C order per user request
    mapping = {
        "Guard": "G",
        "Forward": "F",
        "Center": "C",
        "Guard-Forward": "G/F",
        "Forward-Guard": "G/F", # Force G/F
        "Forward-Center": "F/C",
        "Center-Forward": "F/C", # Force F/C
        "Guard-Center": "G/C",
        "Center-Guard": "G/C"
    }
    
    if pos in mapping:
        return mapping[pos]
    
    # Handle dash-separated values from API with re-ordering
    if "-" in pos:
        parts = [p.strip() for p in pos.split("-")]
        abbrev_parts = [mapping.get(p, p) for p in parts]
        
        # Sort/Reorder to ensure G/F and F/C order
        if "G" in abbrev_parts and "F" in abbrev_parts:
            return "G/F"
        if "F" in abbrev_parts and "C" in abbrev_parts:
            return "F/C"
            
        return "/".join(abbrev_parts)
        
    return pos


def fetch_player_bio(player_name):
    """Get player bio info including height, weight, and draft year."""
    from nba_api.stats.endpoints import commonplayerinfo
    
    try:
        all_players = players.get_players()
        norm_input = normalize_name(player_name).lower()
        player = [p for p in all_players if normalize_name(p['full_name']).lower() == norm_input]
        
        if not player:
            return None
        
        player_id = str(player[0]['id'])
        
        time.sleep(0.6)
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = player_info.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        row = df.iloc[0]
        age = row.get('AGE')
        if not age and 'BIRTHDATE' in row:
            try:
                from datetime import datetime
                import pandas as pd
                birthdate = pd.to_datetime(row['BIRTHDATE'])
                today = get_local_now()
                age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            except:
                age = "N/A"
        
        return {
            'player_id': player_id,
            'height': row.get('HEIGHT', ''),
            'weight': row.get('WEIGHT', ''),
            'draft_year': row.get('DRAFT_YEAR', ''),
            'draft_round': row.get('DRAFT_ROUND', ''),
            'draft_number': row.get('DRAFT_NUMBER', ''),
            'position': row.get('POSITION', ''),
            'jersey': row.get('JERSEY', ''),
            'country': row.get('COUNTRY', ''),
            'team_abbrev': row.get('TEAM_ABBREVIATION', ''),
            'age': age
        }
    except Exception as e:
        # st.error(f"Bio Error for {player_name}: {e}") # Uncomment for debug
        print(f"Bio Error for {player_name}: {e}")
        return None


@st.cache_data(ttl=86400)
def get_active_players_list(season="2025-26"):
    """Get list of players who have actually played games this season."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals"
        )
        
        df = player_stats.get_data_frames()[0]
        active_players = df[df['GP'] > 0]['PLAYER_NAME'].unique().tolist()
        
        return sorted(active_players)
    except Exception:
        return None


def search_players(query, season="2025-26"):
    """Search for players by name."""
    active_players_list = get_active_players_list(season)
    
    if active_players_list:
        if not query:
            return active_players_list[:30]
        
        query_lower = query.lower()
        matches = [name for name in active_players_list 
                   if query_lower in name.lower()]
        return matches[:20]
    else:
        all_players = players.get_players()
        active_players = [p for p in all_players if p.get('is_active', True)]
        
        query_lower = query.lower()
        matches = [p['full_name'] for p in active_players 
                   if query_lower in p['full_name'].lower()]
        
        return matches[:20]

# ==================== AROUND THE NBA DATA FUNCTIONS ====================

@st.cache_data(ttl=1800)  # 30 minute cache
def get_nba_injuries():
    """Fetch injury data from ESPN NBA injuries page."""
    import requests
    from bs4 import BeautifulSoup
    import re
    
    try:
        url = "https://www.espn.com/nba/injuries"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        injuries = []
        
        # ESPN structures data with team sections
        # Each team has a header with team name, followed by a table with player injuries
        
        # Find all content sections that contain team injury data
        # Look for elements with the pattern: team header -> injury table
        
        # Get all Table__Title elements (team names)
        team_headers = soup.find_all('div', class_='Table__Title')
        
        for team_header in team_headers:
            team_name = team_header.get_text(strip=True)
            if not team_name:
                continue
            
            # Find the next sibling/parent table after this team header
            # Navigate up to find the wrapper, then find the table within
            parent = team_header.parent
            table = None
            
            # Try to find table within parent container
            while parent and not table:
                table = parent.find('table')
                if not table:
                    # Try next sibling of parent
                    next_elem = parent.find_next_sibling()
                    if next_elem:
                        table = next_elem.find('table') if hasattr(next_elem, 'find') else None
                    parent = parent.parent if hasattr(parent, 'parent') else None
            
            if not table:
                continue
            
            # Parse rows from this table
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    # Get player name
                    player_cell = cells[0]
                    player_link = player_cell.find('a')
                    player_name = player_link.get_text(strip=True) if player_link else player_cell.get_text(strip=True)
                    
                    # Skip header rows
                    if not player_name or player_name.upper() == "NAME":
                        continue
                    
                    # Get injury date
                    injury_date = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    
                    # Get status
                    status = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    
                    # Get injury description
                    description = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    
                    injuries.append({
                        'team': team_name,
                        'player': player_name,
                        'date': injury_date,
                        'status': status,
                        'description': description
                    })
        
        return injuries
    except Exception as e:
        print(f"Error fetching injuries: {e}")
        return []


def get_team_streaks(standings_df):
    """Calculate hot and cold teams based on standings data."""
    if standings_df is None or standings_df.empty:
        return {'hot': [], 'cold': []}
    
    hot_teams = []
    cold_teams = []
    
    for _, row in standings_df.iterrows():
        team_abbrev = get_team_abbrev(row.get('TeamCity', ''))
        team_name = f"{row.get('TeamCity', '')} {row.get('TeamName', '')}"
        record = row.get('Record', '')
        l10 = row.get('L10', '')
        streak = row.get('strCurrentStreak', '')
        conference = row.get('Conference', '')
        rank = int(row.get('PlayoffRank', 0))
        
        # Parse L10 record (e.g., "7-3" means 7 wins, 3 losses)
        l10_wins = 0
        l10_losses = 0
        try:
            if '-' in str(l10):
                parts = str(l10).split('-')
                l10_wins = int(parts[0])
                l10_losses = int(parts[1])
        except:
            pass
        
        # Parse streak (e.g., "W 5" or "L 3")
        streak_type = ""
        streak_count = 0
        try:
            if streak:
                parts = str(streak).split()
                if len(parts) >= 2:
                    streak_type = parts[0]  # "W" or "L"
                    streak_count = int(parts[1])
        except:
            pass
        
        team_data = {
            'abbrev': team_abbrev,
            'name': team_name,
            'record': record,
            'l10': l10,
            'l10_wins': l10_wins,
            'streak': streak,
            'streak_type': streak_type,
            'streak_count': streak_count,
            'conference': conference,
            'rank': rank
        }
        
        # Hot team criteria: 7+ wins in L10 OR 5+ game winning streak
        if l10_wins >= 7 or (streak_type == 'W' and streak_count >= 5):
            hot_teams.append(team_data)
        
        # Cold team criteria: 3 or fewer wins in L10 OR 5+ game losing streak
        if l10_wins <= 3 or (streak_type == 'L' and streak_count >= 5):
            cold_teams.append(team_data)
    
    # Sort hot teams by streak count (descending), then by L10 wins
    hot_teams.sort(key=lambda x: (x['streak_count'] if x['streak_type'] == 'W' else 0, x['l10_wins']), reverse=True)
    
    # Sort cold teams by streak count (descending), then by L10 losses (most losses first)
    cold_teams.sort(key=lambda x: (x['streak_count'] if x['streak_type'] == 'L' else 0, 10 - x['l10_wins']), reverse=True)
    
    return {'hot': hot_teams, 'cold': cold_teams}


@st.cache_data(ttl=900)  # 15 minute cache
def get_nba_news():
    """Fetch latest NBA news headlines from NBA.com."""
    import requests
    from bs4 import BeautifulSoup
    
    try:
        url = "https://www.nba.com/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        
        # Look for headlines and timestamps
        articles = soup.find_all(['div', 'a'], class_=lambda x: x and ('ArticleTile' in x or 'HeroArticle' in x) if x else False)
        
        for article in articles:
            # Look for headline within this container
            link_elem = article.find('a', class_=lambda x: x and 'HeadlineLink' in x if x else False) or \
                        (article if article.name == 'a' and 'HeadlineLink' in (article.get('class', [])) else None)
            
            if not link_elem:
                # Try finding any link that might be the headline
                link_elem = article.find('a') if article.name != 'a' else article
            
            if link_elem:
                title_elem = link_elem.find(['h1', 'h2', 'h3', 'h4', 'span', 'p'])
                title = title_elem.get_text(strip=True) if title_elem else link_elem.get_text(strip=True)
                link = link_elem.get('href', '')
                
                # Look for timestamp
                time_elem = article.find(class_=lambda x: x and 'Timestamp' in x if x else False)
                timestamp = time_elem.get_text(strip=True) if time_elem else ""
                
                if link and not link.startswith('http'):
                    link = 'https://www.nba.com' + link
                    
                if title and len(title) > 10:
                    # Avoid duplicates
                    if not any(item['title'] == title for item in news_items):
                        news_items.append({
                            'title': title,
                            'link': link,
                            'time': timestamp
                        })
        
        # If no items found with specific classes, try fallback
        if not news_items:
            headlines = soup.find_all(['h3', 'h4'])
            for h in headlines:
                parent_link = h.find_parent('a')
                if parent_link:
                    title = h.get_text(strip=True)
                    link = parent_link.get('href', '')
                    if link and not link.startswith('http'):
                        link = 'https://www.nba.com' + link
                    if title and len(title) > 20:
                        news_items.append({'title': title, 'link': link, 'time': ''})
        
        return news_items[:10]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


@st.cache_data(ttl=3600*24*7)  # 7-day cache for MVP ladder (updates weekly)
def get_mvp_ladder():
    """Fetch the latest Kia MVP Ladder rankings from NBA.com with dynamic URL calculation."""
    import requests
    from bs4 import BeautifulSoup
    import re
    from datetime import datetime, timedelta
    
    # Fallback data (Jan 16, 2026 article)
    fallback_players = [
        {'rank': '1', 'name': 'Shai Gilgeous-Alexander', 'team': 'Oklahoma City Thunder', 'team_abbrev': 'OKC', 'stats': '31.9 PPG, 4.5 RPG, 6.4 APG', 'games_played': '40', 'player_id': '1628983'},
        {'rank': '2', 'name': 'Nikola Jokić', 'team': 'Denver Nuggets', 'team_abbrev': 'DEN', 'stats': '29.6 PPG, 12.2 RPG, 11.0 APG', 'games_played': '39', 'player_id': '203999'},
        {'rank': '3', 'name': 'Luka Dončić', 'team': 'Los Angeles Lakers', 'team_abbrev': 'LAL', 'stats': '33.4 PPG, 7.9 RPG, 8.8 APG', 'games_played': '38', 'player_id': '1629029'},
        {'rank': '4', 'name': 'Victor Wembanyama', 'team': 'San Antonio Spurs', 'team_abbrev': 'SAS', 'stats': '23.9 PPG, 10.9 RPG, 3.0 APG', 'games_played': '37', 'player_id': '1641705'},
        {'rank': '5', 'name': 'Jaylen Brown', 'team': 'Boston Celtics', 'team_abbrev': 'BOS', 'stats': '28.2 PPG, 3.2 RPG, 6.1 APG', 'games_played': '41', 'player_id': '1627759'},
        {'rank': '6', 'name': 'Cade Cunningham', 'team': 'Detroit Pistons', 'team_abbrev': 'DET', 'stats': 'N/A', 'games_played': 'N/A', 'player_id': '1630595'},
        {'rank': '7', 'name': 'Anthony Edwards', 'team': 'Minnesota Timberwolves', 'team_abbrev': 'MIN', 'stats': 'N/A', 'games_played': 'N/A', 'player_id': '1630162'},
        {'rank': '8', 'name': 'Tyrese Maxey', 'team': 'Philadelphia 76ers', 'team_abbrev': 'PHI', 'stats': 'N/A', 'games_played': 'N/A', 'player_id': '1630178'},
        {'rank': '9', 'name': 'Jalen Brunson', 'team': 'New York Knicks', 'team_abbrev': 'NYK', 'stats': 'N/A', 'games_played': 'N/A', 'player_id': '1628973'},
        {'rank': '10', 'name': 'Jamal Murray', 'team': 'Denver Nuggets', 'team_abbrev': 'DEN', 'stats': 'N/A', 'games_played': 'N/A', 'player_id': '1627750'}
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        # Dynamically calculate URL based on current date
        # Articles are published weekly on Thursdays (day 3)
        today = get_local_now()
        # Find most recent Thursday (or today if Thursday)
        days_since_thursday = (today.weekday() - 3) % 7
        recent_thursday = today - timedelta(days=days_since_thursday)
        
        # Try URLs for the past few weeks to find the latest article
        article_url = None
        for weeks_back in range(4):  # Check up to 4 weeks back
            check_date = recent_thursday - timedelta(weeks=weeks_back)
            month_abbrev = check_date.strftime('%b').lower()
            day = check_date.day
            year = check_date.year
            url = f"https://www.nba.com/news/kia-mvp-ladder-{month_abbrev}-{day}-{year}"
            
            try:
                resp = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
                if resp.status_code == 200:
                    article_url = url
                    break
            except:
                continue
        
        if not article_url:
            return fallback_players, "January 16, 2026"
            
        # Scrape the article
        art_resp = requests.get(article_url, headers=headers, timeout=10)
        art_resp.raise_for_status()
        art_soup = BeautifulSoup(art_resp.text, 'html.parser')
        
        players = []
        
        # Parse TOP 5 from h2/h3/h4 headings (e.g., "1. Shai Gilgeous-Alexander, Oklahoma City Thunder")
        rankings = art_soup.find_all(['h2', 'h3', 'h4'])
        for r in rankings:
            text = r.get_text(strip=True)
            # Match pattern like "1. Player Name, Team Name"
            match = re.match(r'^(\d+)\.?\s+(.+)', text)
            if match and int(match.group(1)) <= 5:
                rank = match.group(1)
                rest = match.group(2)
                
                # Split name and team
                if ',' in rest:
                    parts = rest.split(',', 1)
                    name = parts[0].strip()
                    team = parts[1].strip() if len(parts) > 1 else "N/A"
                else:
                    name = rest
                    team = "N/A"
                
                # Get stats from following paragraph
                stats = "N/A"
                current = r.find_next()
                while current and current.name not in ['h2', 'h3', 'h4']:
                    if current.name == 'p':
                        p_text = current.get_text(strip=True)
                        if "stats:" in p_text.lower():
                            stats = p_text.split('stats:')[-1].split('|')[0].strip()
                            break
                    current = current.find_next()
                
                team_abbrev = get_team_abbrev(team)
                
                players.append({
                    'rank': rank,
                    'name': name,
                    'team': team,
                    'team_abbrev': team_abbrev,
                    'stats': stats,
                    'games_played': 'N/A',
                    'player_id': None
                })
        
        # Parse "The Next 5" section (contained in a <p> with <strong> tags for ranks)
        next5_header = art_soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and 'next 5' in tag.get_text().lower())
        if next5_header:
            next_p = next5_header.find_next('p')
            if next_p:
                # The paragraph contains: <strong>6.</strong> Player, Team ↔️<br>...
                p_html = str(next_p)
                # Split by <br> or <br/>
                lines = re.split(r'<br\s*/?\s*>', p_html)
                for line in lines:
                    # Extract rank from <strong>N.</strong>
                    rank_match = re.search(r'<strong>(\d+)\.?</strong>\s*(.+)', line, re.IGNORECASE)
                    if rank_match:
                        rank = rank_match.group(1)
                        rest = rank_match.group(2)
                        # Remove HTML tags and emojis, keep only text
                        rest = re.sub(r'<[^>]+>', '', rest)
                        rest = re.sub(r'[↔️⬆️⬇️↗️↘️]', '', rest).strip()
                        # Clean up any potential invisible characters
                        rest = re.sub(r'\s+', ' ', rest).strip()
                        
                        if ',' in rest:
                            parts = rest.split(',', 1)
                            name = parts[0].strip()
                            team = parts[1].strip() if len(parts) > 1 else "N/A"
                        else:
                            name = rest
                            team = "N/A"
                        
                        team_abbrev = get_team_abbrev(team)
                        
                        players.append({
                            'rank': rank,
                            'name': name,
                            'team': team,
                            'team_abbrev': team_abbrev,
                            'stats': 'N/A',
                            'games_played': 'N/A',
                            'player_id': None
                        })
        
        # Sort by rank and ensure we have 10
        players.sort(key=lambda x: int(x['rank']))
        
        # Use the date from the successful article
        as_of_date = check_date.strftime("%B %d, %Y") if check_date else "January 16, 2026"

        if len(players) >= 10:
            return players[:10], as_of_date
        else:
            return fallback_players, "January 16, 2026"
            
    except Exception as e:
        print(f"Error fetching MVP ladder: {e}")
        return fallback_players, "January 16, 2026"


def get_rookie_ladder():
    """Fetch the latest Kia Rookie Ladder rankings from NBA.com with dynamic URL calculation."""
    import requests
    from bs4 import BeautifulSoup
    import re
    from datetime import datetime, timedelta
    
    # Updated fallback data based on Jan 21, 2026 article
    fallback_players = [
        {'rank': '1', 'name': 'Kon Knueppel', 'team': 'Charlotte Hornets', 'team_abbrev': 'CHA', 'stats': '19 ppg, 5.3 rpg, 3.5 apg', 'draft_pick': '4', 'games_played': 'N/A', 'player_id': None},
        {'rank': '2', 'name': 'Cooper Flagg', 'team': 'Dallas Mavericks', 'team_abbrev': 'DAL', 'stats': '18.8 ppg, 6.3 rpg, 4.1 apg', 'draft_pick': '1', 'games_played': 'N/A', 'player_id': None},
        {'rank': '3', 'name': 'VJ Edgecombe', 'team': 'Philadelphia 76ers', 'team_abbrev': 'PHI', 'stats': '15.8 ppg, 5.3 rpg, 4.2 apg', 'draft_pick': '3', 'games_played': 'N/A', 'player_id': None},
        {'rank': '4', 'name': 'Derik Queen', 'team': 'New Orleans Pelicans', 'team_abbrev': 'NOP', 'stats': '12.6 ppg, 7.5 rpg, 4.3 apg', 'draft_pick': '13', 'games_played': 'N/A', 'player_id': None},
        {'rank': '5', 'name': 'Cedric Coward', 'team': 'Memphis Grizzlies', 'team_abbrev': 'MEM', 'stats': '14 ppg, 6.5 rpg, 2.9 apg', 'draft_pick': '11', 'games_played': 'N/A', 'player_id': None},
        {'rank': '6', 'name': 'Maxime Raynaud', 'team': 'Sacramento Kings', 'team_abbrev': 'SAC', 'stats': '10.1 ppg, 6.6 rpg, 1.1 apg', 'draft_pick': '42', 'games_played': 'N/A', 'player_id': None},
        {'rank': '7', 'name': 'Egor Demin', 'team': 'Brooklyn Nets', 'team_abbrev': 'BKN', 'stats': '10.4 ppg, 3 rpg, 3.4 apg', 'draft_pick': '8', 'games_played': 'N/A', 'player_id': None},
        {'rank': '8', 'name': 'Caleb Love', 'team': 'Portland Trail Blazers', 'team_abbrev': 'POR', 'stats': '11.1 ppg, 2.7 rpg, 2.6 apg', 'draft_pick': 'Undrafted', 'games_played': 'N/A', 'player_id': None},
        {'rank': '9', 'name': 'Jeremiah Fears', 'team': 'New Orleans Pelicans', 'team_abbrev': 'NOP', 'stats': '13.9 ppg, 3.7 rpg, 3.2 apg', 'draft_pick': '7', 'games_played': 'N/A', 'player_id': None},
        {'rank': '10', 'name': 'Dylan Harper', 'team': 'San Antonio Spurs', 'team_abbrev': 'SAS', 'stats': '10.6 ppg, 3.2 rpg, 3.6 apg', 'draft_pick': '2', 'games_played': 'N/A', 'player_id': None},
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        # Dynamically calculate URL based on current date
        # Rookie Ladder articles are published weekly on Tuesdays (weekday 1)
        today = get_local_now()
        days_since_tuesday = (today.weekday() - 1) % 7
        recent_tuesday = today - timedelta(days=days_since_tuesday)
        
        # Try URLs for the past few weeks to find the latest article
        article_url = None
        check_date = None
        for weeks_back in range(4):  # Check up to 4 weeks back
            check_date = recent_tuesday - timedelta(weeks=weeks_back)
            month_abbrev = check_date.strftime('%b').lower()
            day = check_date.day
            year = check_date.year
            url = f"https://www.nba.com/news/kia-rookie-ladder-{month_abbrev}-{day}-{year}"
            
            try:
                resp = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
                if resp.status_code == 200:
                    article_url = url
                    break
            except:
                continue
        
        if not article_url:
            return fallback_players, "January 21, 2026"
            
        # Scrape the article
        art_resp = requests.get(article_url, headers=headers, timeout=10)
        art_resp.raise_for_status()
        art_soup = BeautifulSoup(art_resp.text, 'html.parser')
        
        # Get all text content for parsing
        full_text = art_soup.get_text()
        
        players = []
        
        # Parse using regex on the full text
        # Pattern: "1. Player Name, Team Name" followed by stats on next lines
        # Top 5 and Next 5 have slightly different formats
        
        # Match pattern like "1. Kon Knueppel, Charlotte Hornets"
        player_pattern = re.compile(r'(\d+)\.\s+([A-Za-zÀ-ÿ\s\'\-\.]+),\s+([A-Za-z\s]+(?:76ers|Hornets|Mavericks|Lakers|Heat|Nets|Kings|Grizzlies|Pelicans|Spurs|Trail Blazers|Warriors|Cavaliers|Celtics|Bucks|Knicks|Suns|Thunder|Timberwolves|Nuggets|Clippers|Rockets|Pacers|Hawks|Magic|Pistons|Raptors|Bulls|Jazz|Wizards))')
        
        # Find all player mentions
        matches = player_pattern.findall(full_text)
        
        for match in matches:
            rank = match[0]
            name = match[1].strip()
            team = match[2].strip()
            
            # Only process ranks 1-10
            if not rank.isdigit() or int(rank) < 1 or int(rank) > 10:
                continue
            
            # Skip if player already added (avoid duplicates)
            if any(p['rank'] == rank for p in players):
                continue
            
            # Find stats and draft pick in the text following the player name
            # Search for "Season stats:" and "Draft pick:" after the player mention
            player_section_start = full_text.find(f"{rank}. {name}")
            if player_section_start == -1:
                continue
                
            # Get text chunk after player name (up to next player or 1000 chars)
            next_player_match = re.search(r'\n\d+\.\s+[A-Za-z]', full_text[player_section_start + 50:])
            if next_player_match:
                section_end = player_section_start + 50 + next_player_match.start()
            else:
                section_end = player_section_start + 1000
            
            section = full_text[player_section_start:section_end]
            
            # Extract Season stats
            stats = "N/A"
            stats_match = re.search(r'Season stats:\s*([\d\.]+\s*ppg,\s*[\d\.]+\s*rpg,\s*[\d\.]+\s*apg)', section, re.IGNORECASE)
            if stats_match:
                stats = stats_match.group(1).strip()
            
            # Extract Draft pick
            draft_pick = "N/A"
            pick_match = re.search(r'Draft pick:\s*(?:No\.\s*)?(\d+|Undrafted)', section, re.IGNORECASE)
            if pick_match:
                draft_pick = pick_match.group(1)
            
            team_abbrev = get_team_abbrev(team)
            
            players.append({
                'rank': rank,
                'name': name,
                'team': team,
                'team_abbrev': team_abbrev,
                'stats': stats,
                'draft_pick': draft_pick,
                'games_played': 'N/A',
                'player_id': None
            })
        
        # Sort by rank
        players.sort(key=lambda x: int(x['rank']))
        
        # Use the date from the successful article
        as_of_date = check_date.strftime("%B %d, %Y") if check_date else "January 21, 2026"

        if len(players) >= 5:
            return players[:10], as_of_date
        else:
            # Use fallback if parsing failed
            return fallback_players, "January 21, 2026"
            
    except Exception as e:
        print(f"Error fetching Rookie ladder: {e}")
        return fallback_players, "January 21, 2026"


def get_today_game_slate():
    """Get today's games from the full schedule."""
    from datetime import datetime
    import pytz
    
    # Get current date in ET (NBA standard)
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    date_today = now_et.strftime("%Y-%m-%d")
    
    full_schedule = get_nba_schedule()
    today_games = [g for g in full_schedule if g['game_date'].startswith(date_today)]
    
    return today_games


# ==================== HELPER FUNCTIONS ====================

def parse_minutes(min_str):
    """Parse minutes from various formats."""
    if pd.isna(min_str):
        return 0
    try:
        if ':' in str(min_str):
            parts = str(min_str).split(':')
            return int(parts[0])
        else:
            return float(min_str)
    except:
        return 0


def clear_player_data():
    """Clear player data from session state."""
    st.session_state.player_data = None
    st.session_state.player_team = None
    st.session_state.selected_player = None


# ==================== SIDEBAR ====================
st.sidebar.title("NBA Stats Predictor")
st.sidebar.markdown("---")

# Fixed season for current season
season = "2025-26"

# Initialize page in session state if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

if is_authenticated:
    nav_options = ["Home", "Predictions", "Player Stats", "Compare Players", "Around the NBA", "Standings", "Awards", "Favorites", "About"]
else:
    # Restrict to Home and About when not logged in
    nav_options = ["Home", "About"]

# Check if we have pending navigation (from buttons like "View" or upcoming games)
# This must be checked BEFORE the radio widget is rendered
pending_nav_target = st.session_state.get('pending_nav_target')

if pending_nav_target and pending_nav_target in nav_options:
    # Set the radio button's state BEFORE it's rendered
    st.session_state['nav_radio'] = pending_nav_target
    st.session_state.current_page = pending_nav_target
    del st.session_state['pending_nav_target']


# Get current page index  
current_idx = nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0

page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=current_idx,
    key="nav_radio",
    label_visibility="visible"
)

# Update session state with selected page
if st.session_state.current_page != page:
    # If the user changed the page MANUALLY via sidebar, they probably want the default tab
    if page == "Favorites":
        st.session_state['favorites_requested_tab'] = "Favorite Players"
    st.session_state.current_page = page

# Timezone Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
timezone_options = [
    "US/Pacific", "US/Mountain", "US/Central", "US/Eastern", 
    "US/Alaska", "US/Hawaii", "UTC"
]
selected_tz_name = st.sidebar.selectbox(
    "Your Timezone",
    timezone_options,
    index=timezone_options.index(st.session_state.get('user_timezone', 'US/Pacific')),
    key="user_timezone_select",
    help="Adjusts game times and dates to your local area"
)
if selected_tz_name != st.session_state.get('user_timezone'):
    st.session_state['user_timezone'] = selected_tz_name
    st.rerun()

user_tz = pytz.timezone(st.session_state.get('user_timezone', 'US/Pacific'))

# Show user profile and logout in sidebar
if is_authenticated:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Your Account")
    user_info = auth.get_user_info()
    if user_info:
        name = user_info.get("name", "User")
        email = user_info.get("email", "")
        picture_url = user_info.get("picture")
        
        if picture_url:
            st.sidebar.image(picture_url, width=60)
        st.sidebar.write(f"**{name}**")
        st.sidebar.caption(email)
        
        if st.sidebar.button("Logout", type="primary", use_container_width=True):
            auth.logout()
else:
    st.sidebar.markdown("---")
    st.sidebar.info("Login to save favorites!")

# Refresh Data button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and fetch fresh stats from NBA API"):
    st.cache_data.clear()
    st.toast("✅ Cache cleared! Data will refresh on next load.")
    st.rerun()


# ==================== INITIALIZE SESSION STATE ====================
if 'player_data' not in st.session_state:
    st.session_state.player_data = None
if 'player_team' not in st.session_state:
    st.session_state.player_team = None
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'auto_load_player' not in st.session_state:
    st.session_state.auto_load_player = None
if 'stats_player_data' not in st.session_state:
    st.session_state.stats_player_data = None
if 'stats_player_team' not in st.session_state:
    st.session_state.stats_player_team = None
if 'stats_selected_player' not in st.session_state:
    st.session_state.stats_selected_player = None
if 'stats_last_search' not in st.session_state:
    st.session_state.stats_last_search = ""


# ==================== HOME PAGE ====================
if page == "Home":
    if not is_authenticated:
        # Show login page for unauthenticated users
        render_login_page()
        
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #9CA3AF;'>Sign in to unlock all features</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            auth.show_login_button()
        
        st.markdown("---")
        st.warning("🔒 **Full Access Restricted**: High-performance predictions, player stats, and historical comparisons are reserved for signed-in users.")
    
    else:
        # Personalized home page for authenticated users
        user_info = auth.get_user_info()
        user_name = user_info.get("name", "User") if user_info else "User"
        
        render_welcome_header(user_name)
        
        # Create two-column layout for favorites
        col1, col2 = st.columns(2)
        
        # ===== FAVORITE PLAYERS SECTION =====
        with col1:
            render_section_header("Your Favorite Players", "")
            
            favorite_players = auth.get_favorite_players()
            
            if favorite_players:
                for player_name in favorite_players[:6]:
                    col_img, col_logo, col_info, col_actions = st.columns([0.7, 0.7, 3.4, 1.5])
                    
                    # Fetch bio if not in cache
                    if 'player_bio_cache' not in st.session_state:
                        st.session_state['player_bio_cache'] = {}
                    
                    if player_name not in st.session_state['player_bio_cache']:
                        st.session_state['player_bio_cache'][player_name] = fetch_player_bio(player_name)
                    
                    bio = st.session_state['player_bio_cache'][player_name]
                    
                    with col_img:
                        if bio and bio.get('player_id'):
                            headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{bio['player_id']}.png"
                            st.image(headshot_url, width=65)
                        else:
                            st.markdown("<div style='font-size: 2.5rem; text-align: center; margin-top: 10px;'>👤</div>", unsafe_allow_html=True)

                    with col_logo:
                        # Team Logo
                        team_abbrev = bio.get('team_abbrev') if bio else None
                        if team_abbrev:
                            logo = get_team_logo_url(team_abbrev)
                            if logo:
                                st.image(logo, width=65)
                            else:
                                st.write(team_abbrev)
                    
                    with col_info:
                        # Name + Position
                        pos_label = ""
                        if bio and bio.get('position'):
                            abbrev_pos = abbreviate_position(bio['position'], player_name)
                            pos_label = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>({abbrev_pos})</span>"
                        st.markdown(f"<h4 style='margin: 0; font-size: 1.1rem;'>{player_name}{pos_label}</h4>", unsafe_allow_html=True)
                        
                        # Bio Stats
                        if bio:
                            st.markdown(f"""
                            <div style="display: flex; gap: 10px; flex-wrap: wrap; color: #9CA3AF; font-size: 0.75rem; margin-top: 4px;">
                                <span>HT: <strong style="color: #FAFAFA;">{bio.get('height', '-')}</strong></span>
                                <span>WT: <strong style="color: #FAFAFA;">{bio.get('weight', '-')}</strong></span>
                                <span>Age: <strong style="color: #FAFAFA;">{bio.get('age', '-')}</strong></span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_actions:
                        st.button("Stats", key=f"home_stats_{player_name}", use_container_width=True, 
                                  on_click=nav_to_player_stats, args=(player_name,))
                        st.button("Analyze", key=f"home_predict_{player_name}", use_container_width=True, type="primary",
                                  on_click=nav_to_predictions, args=(player_name,))
            else:
                render_empty_state(
                    "No favorite players yet! Go to Favorites to add some.",
                    ""
                )
        
        # ===== FAVORITE TEAMS SECTION =====
        with col2:
            render_section_header("Your Favorite Teams", "")
            
            favorite_teams = auth.get_favorite_teams()
            team_def_ratings = get_current_defensive_ratings(season)
            
            if favorite_teams:
                for team_abbrev in favorite_teams[:6]:
                    rating = team_def_ratings.get(team_abbrev, "N/A")
                    logo = get_team_logo_url(team_abbrev)
                    
                    l_col, t_col, b_col = st.columns([0.2, 0.4, 0.4])
                    with l_col:
                        if logo:
                            st.image(logo, width=50)
                    with t_col:
                        st.markdown(f"<div style='font-size: 1.1rem; font-weight: bold; margin-top: 8px;'>{team_abbrev}</div>", unsafe_allow_html=True)
                    with b_col:
                        if st.button("Stats", key=f"home_team_stats_{team_abbrev}", type="secondary", use_container_width=True):
                            st.session_state['pending_nav_target'] = "Favorites"
                            st.session_state['favorites_requested_tab'] = "Favorite Teams"
                            st.rerun()
            else:
                render_empty_state(
                    "No favorite teams yet! Add teams from the Live Predictions page.",
                    ""
                )
        
        st.markdown("---")
        
        # ===== QUICK PREDICTIONS SECTION =====
        render_section_header("Quick Predictions", "")
        
        player_search = st.text_input(
            "Search for a player:",
            placeholder="e.g., LeBron James, Stephen Curry...",
            key="home_player_search"
        )
        
        if player_search:
            matching = search_players(player_search, season)
            if matching:
                st.write("**Matching players:**")
                for player in matching[:5]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {player}")
                    with col2:
                        if st.button("Predict", key=f"quick_{player}"):
                            st.session_state["redirect_to_predictions"] = player
                            st.session_state["auto_load_player"] = player
                            st.session_state['pending_nav_target'] = "Predictions"
                            st.rerun()


# ==================== LIVE PREDICTIONS PAGE ====================
elif page == "Predictions":
    st.title("Player Performance Predictions")
    st.markdown("Predict any NBA player's next game stats using real-time data")
    
    # Fetch defensive ratings and schedule data
    with st.spinner("Fetching latest data..."):
        team_def_ratings = get_current_defensive_ratings(season)
        nba_schedule = get_nba_schedule()
        standings_df = get_league_standings(season)
    
    if not team_def_ratings:
        st.error("Could not fetch defensive ratings. Please try again later.")
        st.stop()
    
    # Today's Games Section
    st.markdown("### Today's Games")
    # Pass the user_tz defined in sidebar
    todays_games = get_todays_games(nba_schedule, standings_df, tz=user_tz)
    
    # Fetch live/final scores
    scoreboard = get_todays_scoreboard()
    
    if todays_games:
        from datetime import datetime
        tz_label = st.session_state.get('user_timezone', 'US/Pacific').split('/')[-1].replace('_', ' ')
        st.caption(f"**{get_local_now().strftime('%A, %B %d, %Y')}** • {len(todays_games)} game(s) • _All times in {tz_label}_")
        
        # Display games in a grid
        cols_per_row = 3
        for i in range(0, len(todays_games), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(todays_games):
                    game = todays_games[i + j]
                    with col:
                        # Get team logos
                        away_logo = get_team_logo_url(game['away_team'])
                        home_logo = get_team_logo_url(game['home_team'])
                        channel_display = game.get('channel', '')
                        
                        # Add channel display to top right if available
                        channel_html = f"<div style='position: absolute; top: 12px; right: 12px; color: #9CA3AF; font-size: 0.72rem; font-weight: 600;'>{channel_display}</div>" if channel_display else ""
                        
                        # Format seeds and conference
                        away_seed = f"#{game['away_rank']}" if game['away_rank'] else ""
                        home_seed = f"#{game['home_rank']}" if game['home_rank'] else ""
                        def fmt_cs(conf, streak):
                            if not conf: return ""
                            c = conf.replace("ern", "")
                            s = f"({streak.replace(' ', '')})" if streak else ""
                            return f"{c} {s}"
                            
                        away_conf = fmt_cs(game.get('away_conference'), game.get('away_streak'))
                        home_conf = fmt_cs(game.get('home_conference'), game.get('home_streak'))
                        
                        game_time = game.get('game_time', 'TBD')
                        if not game_time:
                            game_time = 'TBD'
                        
                        # Get records
                        away_record = game.get('away_record', '')
                        home_record = game.get('home_record', '')
                        
                        # Get live score from scoreboard
                        game_id = game.get('game_id')
                        live_data = scoreboard.get(game_id, {})
                        game_status = live_data.get('game_status', 1)
                        status_text = live_data.get('game_status_text', '')
                        home_score = live_data.get('home_score', 0)
                        away_score = live_data.get('away_score', 0)
                        
                        # Determine display based on game status
                        if game_status == 3:  # Final
                            status_display = f"<span style='color: #10B981;'>FINAL</span>"
                            away_score_display = f"<span style='font-size: 1.3rem; font-weight: bold; color: {'#10B981' if away_score > home_score else '#FAFAFA'};'>{away_score}</span>"
                            home_score_display = f"<span style='font-size: 1.3rem; font-weight: bold; color: {'#10B981' if home_score > away_score else '#FAFAFA'};'>{home_score}</span>"
                        elif game_status == 2:  # In Progress
                            status_display = f"<span style='color: #F59E0B;'>{status_text}</span>"
                            away_score_display = f"<span style='font-size: 1.3rem; font-weight: bold;'>{away_score}</span>"
                            home_score_display = f"<span style='font-size: 1.3rem; font-weight: bold;'>{home_score}</span>"
                        else:  # Scheduled - show 'Home' label for home team
                            status_display = f"<span style='color: #FF6B35;'>{game_time}</span>"
                            away_score_display = ""
                            home_score_display = "<span style='color: #9CA3AF; font-size: 0.75rem;'>Home</span>"
                        
                        # Create a more visual card with logos and scores
                        channel_display = game.get('channel', '')
                        box_score_url = f"https://www.nba.com/game/{game_id}" if game_id else "https://www.nba.com/games"
                        box_score_html = f"<div style='position: absolute; top: 12px; left: 12px; font-size: 0.72rem; font-weight: 700;'><a href='{box_score_url}' target='_blank' style='color: #9CA3AF; text-decoration: none; border: 1px solid #374151; padding: 2px 6px; border-radius: 4px;'>BOX SCORE</a></div>"
                        channel_html = f"<div style='position: absolute; top: 12px; right: 12px; color: #9CA3AF; font-size: 0.72rem; font-weight: 600;'>{channel_display}</div>" if channel_display else ""
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1F2937 0%, #111827 100%); 
                                    border-radius: 10px; padding: 15px; text-align: center; 
                                    border: 1px solid #374151; margin-bottom: 10px; position: relative;">
                            {box_score_html}
                            {channel_html}
                            <div style="font-size: 0.8rem; font-weight: bold; margin-bottom: 8px;">{status_display}</div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <img src="{away_logo}" width="38" height="38" style="filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['away_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{away_record}, {away_seed} {away_conf}</div>
                                    </div>
                                </div>
                                <div>{away_score_display}</div>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 10px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <img src="{home_logo}" width="38" height="38" style="filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['home_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{home_record}, {home_seed} {home_conf}</div>
                                    </div>
                                </div>
                                <div>{home_score_display}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No games scheduled for today.")
    
    st.markdown("---")
    
    # Check for redirect from home page or favorites
    initial_search = st.session_state.pop("redirect_to_predictions", "")
    auto_load = st.session_state.pop("auto_load_player", None)
    
    # If we have an auto-load request, fetch data immediately
    if auto_load:
        initial_search = auto_load
        with st.spinner(f"Fetching {auto_load}'s game log..."):
            player_df, player_team = get_player_game_log(auto_load, season)
        
        if player_df is not None and len(player_df) > 0:
            st.session_state.player_data = player_df
            st.session_state.player_team = player_team
            st.session_state.selected_player = auto_load
            st.session_state.last_search = auto_load
    
    # Player search
    st.markdown("### Select Player")
    st.caption("Only showing players active in current season")
    
    player_search = st.text_input(
        "Search player name:", 
        placeholder="e.g., LeBron James",
        value=initial_search,
        key="prediction_player_search"
    )
    
    # Track search changes
    current_search = player_search.strip()
    if current_search != st.session_state.last_search:
        if (st.session_state.last_search != "" and current_search != st.session_state.last_search) or \
           (st.session_state.player_data is not None and current_search == ""):
            clear_player_data()
        st.session_state.last_search = current_search
    
    if player_search:
        matching_players = search_players(player_search, season)
        if matching_players:
            selected_player = st.selectbox("Select from matches:", matching_players, key="player_select")
            
            if st.button("Load Player Data", type="primary", key="load_player"):
                with st.spinner(f"Fetching {selected_player}'s game log..."):
                    player_df, player_team = get_player_game_log(selected_player, season)
                
                if player_df is None or len(player_df) == 0:
                    st.error(f"No games found for {selected_player} in {season} season.")
                else:
                    st.session_state.player_data = player_df
                    st.session_state.player_team = player_team
                    st.session_state.selected_player = selected_player
                    st.success(f"Loaded {len(player_df)} games for {selected_player} (Team: {player_team})")
                    st.rerun()
        else:
            st.warning("No active players found with that name.")
            clear_player_data()
    else:
        if st.session_state.player_data is not None:
            clear_player_data()
    
    # Show player data if loaded
    if st.session_state.player_data is not None:
        selected_player = st.session_state.selected_player
        player_team = st.session_state.player_team
        player_df = st.session_state.player_data
        
        # Fetch schedule and standings for upcoming games
        nba_schedule = get_nba_schedule()
        standings_df = get_league_standings(season)
        
        # Get player bio first for better branding
        bio = fetch_player_bio(selected_player)
        if bio and bio.get('team_abbrev'):
            player_team = bio['team_abbrev']

        # Show player photo and team logo centered (Standardized layout) - Shifted slightly right for balance
        spacer1, photo_col, logo_col, spacer2 = st.columns([1.35, 0.8, 0.8, 1.05])
        with photo_col:
            player_photo = get_player_photo_url(selected_player)
            if player_photo:
                st.image(player_photo, width=150)
        with logo_col:
            team_logo = get_team_logo_url(player_team)
            if team_logo:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                    <img src="{team_logo}" style="width: 120px; height: 120px; filter: drop-shadow(0px 4px 8px rgba(0,0,0,0.5));">
                </div>
                """, unsafe_allow_html=True)
        
        # Get team seed and full name
        team_seed_suffix = ""
        team_full_name = player_team
        if standings_df is not None:
            for _, row in standings_df.iterrows():
                if get_team_abbrev(row['TeamCity']) == player_team:
                    team_rec = row['Record']
                    team_rank = row['PlayoffRank']
                    team_conf = row['Conference']
                    team_seed_suffix = f" (#{int(team_rank)})"
                    team_full_name = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                    team_display = f"{team_full_name} <span style='color: #9CA3AF; font-size: 1.1rem;'>({team_rec} | #{team_rank} in {team_conf})</span>"
                    break
        
        # Player name, position and team centered
        pos_label = ""
        if bio and bio.get('position'):
            abbrev_pos = abbreviate_position(bio['position'], selected_player)
            pos_label = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 1.5rem;'>({abbrev_pos})</span>"
            
        st.markdown(f"""
            <div style='text-align: center;'>
                <h3 style='margin-bottom: 0px;'>{selected_player}{pos_label}</h3>
                <div style='color: #9CA3AF; font-size: 1.1rem; margin-bottom: 10px;'>{team_display}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Format Bio Info with muted labels and bold values (Awards card style)
        if bio:
            height = bio.get('height', '-')
            weight = bio.get('weight', '-')
            age = bio.get('age', '-')
            draft_year = bio.get('draft_year', '-')
            draft_round = bio.get('draft_round', '')
            draft_num = bio.get('draft_number', '')
            draft_display = draft_year
            if draft_round and draft_num and draft_year != 'Undrafted':
                draft_display = f"{draft_year} R{draft_round}, #{draft_num}"
            
            bio_html = f"""<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem; margin-bottom: 12px;">
<span style="color: #9CA3AF;">HT:</span> <span style="color: #FAFAFA; font-weight: bold;">{height}</span> • 
<span style="color: #9CA3AF;">WT:</span> <span style="color: #FAFAFA; font-weight: bold;">{weight} lbs</span> • 
<span style="color: #9CA3AF;">Age:</span> <span style="color: #FAFAFA; font-weight: bold;">{age}</span> • 
<span style="color: #9CA3AF;">Draft:</span> <span style="color: #FAFAFA; font-weight: bold;">{draft_display}</span>
</div>"""
            st.markdown(bio_html, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem; margin-bottom: 12px;'>Team: {player_team}</div>", unsafe_allow_html=True)
        
        
        # Favorite buttons closer together (only for authenticated users)
        if is_authenticated:
            spacer1, c1, c2, spacer2 = st.columns([1, 1.5, 1.5, 1])
            with c1:
                if st.button(f"⭐ Favorite {selected_player}", use_container_width=True):
                    if auth.add_favorite_player(selected_player):
                        st.toast(f"Added {selected_player} to favorites!")
                    else:
                        st.toast(f"{selected_player} is already in favorites!")
            
            with c2:
                if st.button(f"⭐ Favorite {player_team}{team_seed_suffix}", use_container_width=True):
                    if auth.add_favorite_team(player_team):
                        st.toast(f"Added {player_team} to favorite teams!")
                    else:
                        st.toast(f"{player_team} is already being watched!")

        # Show recent games right after loading - ADD MINUTES
        st.markdown("### Recent Performance")
        
        # Define columns to display with minutes, W/L, and score
        recent_cols = [
            'GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 
            'Steals', 'Blocks', 'Turnovers', 'PF', 'FG', '3P', 'FT', 'TS%'
        ]
        
        # Get actual team game scores by fetching team game log
        if 'Score' not in player_df.columns:
            # Find all unique teams the player has played for this season explicitly from their matchups
            # Matchup format is usually "ATL vs ..." or "ATL @ ..."
            unique_teams = set()
            if 'MATCHUP' in player_df.columns:
                for match in player_df['MATCHUP']:
                    if len(match) >= 3:
                        unique_teams.add(match[:3])
            
            # If for some reason we couldn't parse matchups, fallback to current team
            if not unique_teams:
                unique_teams.add(player_team)
            
            score_lookup = {}
            
            # Fetch game data for ALL teams
            for team_abbrev in unique_teams:
                team_game_data = get_team_game_log(team_abbrev, season, num_games=82)
                
                if team_game_data is not None and len(team_game_data) > 0:
                    for _, trow in team_game_data.iterrows():
                        game_id = str(trow.get('GAME_ID', ''))
                        if 'PTS' in trow and 'PLUS_MINUS' in trow:
                            team_pts = int(trow['PTS'])
                            opp_pts = int(trow['PTS'] - trow['PLUS_MINUS'])
                            score_lookup[game_id] = f"{team_pts} - {opp_pts}"
            
            if score_lookup:
                # Apply to player df using GAME_ID
                player_df['Score'] = player_df.apply(
                    lambda row: score_lookup.get(str(row.get('Game_ID', row.get('GAME_ID', ''))), 'N/A'),
                    axis=1
                )
            else:
                player_df['Score'] = 'N/A'
        
        # Filter to only include columns that exist
        available_cols = [col for col in recent_cols if col in player_df.columns]
        
        # Get last 5 games (most recent at top)
        recent_games = player_df.tail(5)[available_cols].iloc[::-1].copy()
        
        # Format TS% column
        if 'TS%' in recent_games.columns:
            recent_games['TS%'] = recent_games['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        
        # Format PF as integer (convert to string to preserve after concat)
        if 'PF' in recent_games.columns:
            recent_games['PF'] = recent_games['PF'].fillna(0).astype(int).astype(str)
        
        # Calculate averages for the last 5 games
        last_5_df = player_df.tail(5).copy()
        
        # Calculate averages (use whole numbers for minutes)
        if 'MIN' in last_5_df.columns:
            # Convert minutes to numeric (handle MM:SS format if present)
            def parse_minutes(min_str):
                if pd.isna(min_str):
                    return 0
                try:
                    if ':' in str(min_str):
                        parts = str(min_str).split(':')
                        return int(parts[0])  # Just take the minutes part
                    else:
                        return float(min_str)
                except:
                    return 0
            
            last_5_df['MIN_NUM'] = last_5_df['MIN'].apply(parse_minutes)
            avg_minutes = round(last_5_df['MIN_NUM'].mean(), 1)
        else:
            avg_minutes = "N/A"
        
        # Calculate averages for numeric columns
        avg_points = round(last_5_df['Points'].mean(), 1)
        avg_rebounds = round(last_5_df['Rebounds'].mean(), 1)
        avg_assists = round(last_5_df['Assists'].mean(), 1)
        avg_steals = round(last_5_df['Steals'].mean(), 1)
        avg_blocks = round(last_5_df['Blocks'].mean(), 1)
        avg_turnovers = round(last_5_df['Turnovers'].mean(), 1)
        
        # Calculate shooting averages
        if 'FGM' in last_5_df.columns and 'FGA' in last_5_df.columns:
            avg_fgm = round(last_5_df['FGM'].mean(), 1)
            avg_fga = round(last_5_df['FGA'].mean(), 1)
            total_fgm = last_5_df['FGM'].sum()
            total_fga = last_5_df['FGA'].sum()
            fg_pct = round((total_fgm / total_fga * 100), 1) if total_fga > 0 else 0
            avg_fg = f"{avg_fgm:.1f}/{avg_fga:.1f}"
        else:
            avg_fg = "N/A"
            fg_pct = "N/A"
        
        # Three pointers
        if '3PM' in last_5_df.columns and '3PA' in last_5_df.columns:
            avg_3pm = round(last_5_df['3PM'].mean(), 1)
            avg_3pa = round(last_5_df['3PA'].mean(), 1)
            total_3pm = last_5_df['3PM'].sum()
            total_3pa = last_5_df['3PA'].sum()
            three_pct = round((total_3pm / total_3pa * 100), 1) if total_3pa > 0 else 0
            avg_3p = f"{avg_3pm:.1f}/{avg_3pa:.1f}"
        else:
            avg_3p = "N/A"
            three_pct = "N/A"
        
        # Free throws
        if 'FTM' in last_5_df.columns and 'FTA' in last_5_df.columns:
            avg_ftm = round(last_5_df['FTM'].mean(), 1)
            avg_fta = round(last_5_df['FTA'].mean(), 1)
            total_ftm = last_5_df['FTM'].sum()
            total_fta = last_5_df['FTA'].sum()
            ft_pct = round((total_ftm / total_fta * 100), 1) if total_fta > 0 else 0
            avg_ft = f"{avg_ftm:.1f}/{avg_fta:.1f}"
        else:
            avg_ft = "N/A"
            ft_pct = "N/A"
        
        # Calculate True Shooting Percentage for last 5 games
        if 'FGA' in last_5_df.columns and 'FTA' in last_5_df.columns:
            total_points = last_5_df['Points'].sum()
            total_fga_l5 = last_5_df['FGA'].sum()
            total_fta_l5 = last_5_df['FTA'].sum()
            ts_pct = round((total_points / (2 * (total_fga_l5 + 0.44 * total_fta_l5)) * 100), 1) if (total_fga_l5 + 0.44 * total_fta_l5) > 0 else 0
        else:
            ts_pct = "N/A"
        
        # Create averages row - leave W/L and Score blank
        averages_row = {
            'GAME_DATE': 'AVG (Last 5)',
            'MATCHUP': '',
            'W/L': '',  # Leave blank for averages
            'Score': '',  # Leave blank for averages
            'MIN': f"{avg_minutes:.1f}" if avg_minutes != "N/A" else "N/A",
            'Points': f"{avg_points:.1f}",
            'Rebounds': f"{avg_rebounds:.1f}",
            'Assists': f"{avg_assists:.1f}",
            'Steals': f"{avg_steals:.1f}",
            'Blocks': f"{avg_blocks:.1f}",
            'Turnovers': f"{avg_turnovers:.1f}",
            'PF': f"{round(last_5_df['PF'].mean(), 1):.1f}" if 'PF' in last_5_df.columns else "N/A",
            'FG': f"{fg_pct}%",
            '3P': f"{three_pct}%",
            'FT': f"{ft_pct}%",
            'TS%': f"{ts_pct:.1f}%" if isinstance(ts_pct, (int, float)) else ts_pct
        }
        
        # Add the averages row to the dataframe
        averages_df_row = pd.DataFrame([averages_row])
        
        # Combine with recent games
        display_df = pd.concat([recent_games, averages_df_row], ignore_index=True)
        
        # Highlight the averages row and color W/L
        def style_row(row):
            styles = [''] * len(row)
            if row['GAME_DATE'] == 'AVG (Last 5)':
                styles = ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
            return styles
        
        def style_wl(val):
            if val == 'W':
                return 'color: #10B981; font-weight: bold'
            elif val == 'L':
                return 'color: #EF4444; font-weight: bold'
            return ''
        
        # Display the table with styling
        styled_df = display_df.style.apply(style_row, axis=1)
        if 'W/L' in display_df.columns:
            styled_df = styled_df.applymap(style_wl, subset=['W/L'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "GAME_DATE": st.column_config.TextColumn("Date", width="medium"),
                "MATCHUP": st.column_config.TextColumn("Matchup", width="medium"),
                "W/L": st.column_config.TextColumn("W/L", width="small"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "MIN": st.column_config.TextColumn("MIN", width="small"),
                "Points": st.column_config.TextColumn("PTS", width="small"),
                "Rebounds": st.column_config.TextColumn("REB", width="small"),
                "Assists": st.column_config.TextColumn("AST", width="small"),
                "Steals": st.column_config.TextColumn("STL", width="small"),
                "Blocks": st.column_config.TextColumn("BLK", width="small"),
                "Turnovers": st.column_config.TextColumn("TO", width="small"),
                "PF": st.column_config.TextColumn("PF", width="small"),
                "TS%": st.column_config.TextColumn("TS%", width="small"),
            }
        )

        # Upcoming Games Section
        st.markdown("### Upcoming Games")
        upcoming_games = get_team_upcoming_games(player_team, nba_schedule, standings_df, num_games=5)
        
        if upcoming_games:
            st.caption("Click an upcoming opponent to quickly select them for prediction!")
            
            # Create clickable buttons for each upcoming game
            cols = st.columns(min(len(upcoming_games), 5))
            for i, game in enumerate(upcoming_games):
                with cols[i]:
                    home_away = "vs" if game['is_home'] else "@"
                    opp_rank = game.get('opponent_rank', 0)
                    opp_conf = game.get('opponent_conference', '')
                    if opp_rank:
                        label = f"{game['date']} {home_away} {game['opponent']} (#{opp_rank} {opp_conf})"
                    else:
                        label = f"{game['date']} {home_away} {game['opponent']}"
                    
                    if st.button(label, key=f"upcoming_{i}_{game['opponent']}", use_container_width=True):
                        st.session_state['selected_upcoming_opponent'] = game['opponent']
                        st.rerun()
        else:
            st.info("No upcoming games found.")
        
        st.markdown("---")

        # Opponent selection - filter out player's own team
        st.markdown("### Select Opponent")
        available_teams = sorted(team_def_ratings.keys())

        # Remove player's own team from available opponents
        if player_team in available_teams:
            available_teams.remove(player_team)

        # Also check for team abbreviations variations
        team_mapping = {
            'GSW': 'GS', 'PHX': 'PHO', 'NOP': 'NO', 'NYK': 'NY',
            'SAS': 'SA', 'UTA': 'UTAH'
        }

        # Check all possible variations
        variations_to_remove = []
        for team in available_teams:
            if team == player_team:
                variations_to_remove.append(team)
            elif team in team_mapping and team_mapping[team] == player_team:
                variations_to_remove.append(team)
            elif player_team in team_mapping and team_mapping[player_team] == team:
                variations_to_remove.append(team)

        for team in variations_to_remove:
            if team in available_teams:
                available_teams.remove(team)

        if not available_teams:
            st.error("No valid opponents available. This might be a data issue.")

        else:
            # Check if opponent was selected from upcoming games - set it directly on the selectbox key
            if 'selected_upcoming_opponent' in st.session_state:
                upcoming_opp = st.session_state['selected_upcoming_opponent']
                if upcoming_opp in available_teams:
                    # Set the selectbox's session state key directly
                    st.session_state['opponent_selectbox'] = upcoming_opp
                del st.session_state['selected_upcoming_opponent']
            
            # Initialize selectbox key if not set or invalid
            if 'opponent_selectbox' not in st.session_state or st.session_state.get('opponent_selectbox') not in available_teams:
                st.session_state['opponent_selectbox'] = available_teams[0]
            
            # Opponent selection with logo
            opp_col1, opp_col2 = st.columns([0.9, 0.1])
            
            with opp_col1:
                selected_opponent = st.selectbox(
                    "Opponent Team:",
                    available_teams,
                    key="opponent_selectbox",
                    format_func=lambda x: TEAM_NAME_MAP.get(x, x),
                    help=f"{selected_player} currently plays for {TEAM_NAME_MAP.get(player_team, player_team)} (excluded from list)"
                )
            
            with opp_col2:
                opp_logo = get_team_logo_url(selected_opponent)
                if opp_logo:
                    st.image(opp_logo, width=50)
            
            opp_rating = team_def_ratings.get(selected_opponent, 0)
            st.caption(f"Defensive Rating: **{opp_rating}** (Lower is better defense)")
            
            # Calculate player's averages against this opponent this season
            st.markdown("### Games vs " + selected_opponent + " This Season")
            
            # Filter games against this opponent
            games_vs_opponent = player_df[player_df['Opponent'] == selected_opponent]
            
            if len(games_vs_opponent) > 0:
                # Calculate averages against this opponent
                num_games = len(games_vs_opponent)
                
                # Calculate W-L record against this opponent
                if 'W/L' in games_vs_opponent.columns:
                    wins_vs = len(games_vs_opponent[games_vs_opponent['W/L'] == 'W'])
                    losses_vs = len(games_vs_opponent[games_vs_opponent['W/L'] == 'L'])
                    WL_record = f"{wins_vs}-{losses_vs}"
                else:
                    WL_record = "N/A"
                
                # Calculate basic stats
                avg_points_vs = round(games_vs_opponent['Points'].mean(), 1)
                avg_rebounds_vs = round(games_vs_opponent['Rebounds'].mean(), 1)
                avg_assists_vs = round(games_vs_opponent['Assists'].mean(), 1)
                avg_steals_vs = round(games_vs_opponent['Steals'].mean(), 1)
                avg_blocks_vs = round(games_vs_opponent['Blocks'].mean(), 1)
                avg_turnovers_vs = round(games_vs_opponent['Turnovers'].mean(), 1)
                
                # Calculate shooting stats
                if 'FGM' in games_vs_opponent.columns and 'FGA' in games_vs_opponent.columns:
                    total_fgm_vs = games_vs_opponent['FGM'].sum()
                    total_fga_vs = games_vs_opponent['FGA'].sum()
                    fg_pct_vs = round((total_fgm_vs / total_fga_vs * 100), 1) if total_fga_vs > 0 else 0
                    avg_fgm_vs = round(games_vs_opponent['FGM'].mean(), 1)
                    avg_fga_vs = round(games_vs_opponent['FGA'].mean(), 1)
                    avg_fg_vs = f"{avg_fgm_vs:.1f}/{avg_fga_vs:.1f}"
                else:
                    avg_fg_vs = "N/A"
                    fg_pct_vs = "N/A"
                
                if '3PM' in games_vs_opponent.columns and '3PA' in games_vs_opponent.columns:
                    total_3pm_vs = games_vs_opponent['3PM'].sum()
                    total_3pa_vs = games_vs_opponent['3PA'].sum()
                    three_pct_vs = round((total_3pm_vs / total_3pa_vs * 100), 1) if total_3pa_vs > 0 else 0
                    avg_3pm_vs = round(games_vs_opponent['3PM'].mean(), 1)
                    avg_3pa_vs = round(games_vs_opponent['3PA'].mean(), 1)
                    avg_3p_vs = f"{avg_3pm_vs:.1f}/{avg_3pa_vs:.1f}"
                else:
                    avg_3p_vs = "N/A"
                    three_pct_vs = "N/A"
                
                if 'FTM' in games_vs_opponent.columns and 'FTA' in games_vs_opponent.columns:
                    total_ftm_vs = games_vs_opponent['FTM'].sum()
                    total_fta_vs = games_vs_opponent['FTA'].sum()
                    ft_pct_vs = round((total_ftm_vs / total_fta_vs * 100), 1) if total_fta_vs > 0 else 0
                    avg_ftm_vs = round(games_vs_opponent['FTM'].mean(), 1)
                    avg_fta_vs = round(games_vs_opponent['FTA'].mean(), 1)
                    avg_ft_vs = f"{avg_ftm_vs:.1f}/{avg_fta_vs:.1f}"
                else:
                    avg_ft_vs = "N/A"
                    ft_pct_vs = "N/A"
                
                # Calculate minutes
                if 'MIN' in games_vs_opponent.columns:
                    def parse_minutes_simple(min_str):
                        if pd.isna(min_str):
                            return 0
                        try:
                            if ':' in str(min_str):
                                parts = str(min_str).split(':')
                                return int(parts[0])  # Just take minutes
                            else:
                                return float(min_str)
                        except:
                            return 0
                    
                    games_vs_opponent_copy = games_vs_opponent.copy()
                    games_vs_opponent_copy['MIN_NUM'] = games_vs_opponent_copy['MIN'].apply(parse_minutes_simple)
                    avg_minutes_vs = round(games_vs_opponent_copy['MIN_NUM'].mean(), 1)
                else:
                    avg_minutes_vs = "N/A"
                
                # Calculate TS%
                if 'FGA' in games_vs_opponent.columns and 'FTA' in games_vs_opponent.columns:
                    total_points_vs = games_vs_opponent['Points'].sum()
                    total_fga_vs_ts = games_vs_opponent['FGA'].sum()
                    total_fta_vs_ts = games_vs_opponent['FTA'].sum()
                    ts_pct_vs = round((total_points_vs / (2 * (total_fga_vs_ts + 0.44 * total_fta_vs_ts)) * 100), 1) if (total_fga_vs_ts + 0.44 * total_fta_vs_ts) > 0 else 0
                else:
                    ts_pct_vs = "N/A"
                
                # Show individual game results against this opponent with averages row
                st.markdown(f"**Games Played: {num_games}** | **Record: {WL_record}**")
                
                # Get individual games - include W/L and Score
                vs_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 
                          'Steals', 'Blocks', 'Turnovers', 'FG', '3P', 'FT', 'TS%']
                vs_opponent_display = games_vs_opponent[[c for c in vs_cols if c in games_vs_opponent.columns]].iloc[::-1].copy()
                
                # Format TS% for individual games
                if 'TS%' in vs_opponent_display.columns:
                    vs_opponent_display['TS%'] = vs_opponent_display['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                
                # Create averages row
                averages_row_vs = {
                    'GAME_DATE': 'AVG vs ' + selected_opponent,
                    'MATCHUP': f'({num_games} games)',
                    'W/L': '',
                    'Score': '',
                    'MIN': f"{avg_minutes_vs:.1f}" if avg_minutes_vs != "N/A" else "N/A",
                    'Points': f"{avg_points_vs:.1f}",
                    'Rebounds': f"{avg_rebounds_vs:.1f}",
                    'Assists': f"{avg_assists_vs:.1f}",
                    'Steals': f"{avg_steals_vs:.1f}",
                    'Blocks': f"{avg_blocks_vs:.1f}",
                    'Turnovers': f"{avg_turnovers_vs:.1f}",
                    'FG': f"{fg_pct_vs:.1f}%" if fg_pct_vs != "N/A" else "N/A",
                    '3P': f"{three_pct_vs:.1f}%" if three_pct_vs != "N/A" else "N/A",
                    'FT': f"{ft_pct_vs:.1f}%" if ft_pct_vs != "N/A" else "N/A",
                    'TS%': f"{ts_pct_vs:.1f}%" if isinstance(ts_pct_vs, (int, float)) else ts_pct_vs
                }
                
                # Add the averages row to the dataframe
                averages_df_row_vs = pd.DataFrame([averages_row_vs])
                
                # Combine with individual games
                combined_display = pd.concat([vs_opponent_display, averages_df_row_vs], ignore_index=True)
                
                # Highlight the averages row and W/L column
                def highlight_vs_average_row(row):
                    if 'AVG vs ' in str(row['GAME_DATE']):
                        return ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
                    else:
                        return [''] * len(row)
                
                def style_wl(val):
                    if val == 'W':
                        return 'color: #10B981; font-weight: bold'
                    elif val == 'L':
                        return 'color: #EF4444; font-weight: bold'
                    return ''
                
                # Apply row styling
                styled_df = combined_display.style.apply(highlight_vs_average_row, axis=1)
                
                # Apply W/L column styling if present
                if 'W/L' in combined_display.columns:
                    styled_df = styled_df.applymap(style_wl, subset=['W/L'])
                
                # Display the table with styling
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                opp_full = TEAM_NAME_MAP.get(selected_opponent, selected_opponent)
                st.info(f"**{selected_player}** has not played against the **{opp_full}** yet this season.")
        
            st.markdown("---")
            
            # Generate Prediction Button
            if st.button("Generate Prediction", type="primary", use_container_width=True):
                with st.spinner("Training prediction model..."):
                    model, stat_cols, scaler, filtered_df = train_hmm_with_drtg(
                        player_df, 
                        team_def_ratings, 
                        n_states=4,
                        use_temporal_weighting=True,
                        weight_strength='medium'
                    )
                
                if model is None:
                    st.error("Insufficient data to train model. Need at least 5 games.")
                else:
                    consistency = calculate_player_consistency(filtered_df, ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers'])
                    consistency_interpretation = "High Variance" if consistency > 0.5 else "Moderate" if consistency > 0.3 else "Very Consistent"
                    #st.info(f"Player Consistency: **{consistency_interpretation}** (CV: {consistency:.2f})")
                    
                    with st.spinner("Generating prediction..."):
                        prediction = predict_with_drtg(
                            model, stat_cols, scaler, filtered_df,
                            team_def_ratings, selected_opponent, 
                            full_player_df=player_df  # Pass full df for H2H data
                        )
                    
                    if prediction:
                        st.success("Prediction Complete!")
                        opp_full = TEAM_NAME_MAP.get(selected_opponent, selected_opponent)
                        st.markdown(f"### Predicted Stats: {selected_player} vs {opp_full}")
                        
                        # Display metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Points", int(round(prediction['Points'])))
                            st.metric("Steals", int(round(prediction['Steals'])))
                        
                        with metric_col2:
                            st.metric("Rebounds", int(round(prediction['Rebounds'])))
                            st.metric("Blocks", int(round(prediction['Blocks'])))
                        
                        with metric_col3:
                            st.metric("Assists", int(round(prediction['Assists'])))
                            st.metric("Turnovers", int(round(prediction['Turnovers'])))
                        
                        st.markdown("---")
                        
                        # Visualization
                        st.markdown("### Prediction vs Season Average")
                        
                        stats_to_plot = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks']
                        pred_values = [prediction[stat] for stat in stats_to_plot]
                        avg_values = [filtered_df[stat].mean() for stat in stats_to_plot]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig.patch.set_facecolor('#161B22')
                        ax.set_facecolor('#161B22')
                        
                        x = np.arange(len(stats_to_plot))
                        width = 0.35
                        
                        bars1 = ax.bar(x - width/2, pred_values, width, label='Predicted', color='#FF6B35')
                        bars2 = ax.bar(x + width/2, avg_values, width, label='Season Avg', color='#4ECDC4')
                        
                        ax.set_xlabel('Stats', color='#FAFAFA')
                        ax.set_ylabel('Value', color='#FAFAFA')
                        ax.set_title(f'{selected_player} - Prediction vs Season Average', color='#FAFAFA')
                        ax.set_xticks(x)
                        ax.set_xticklabels(stats_to_plot)
                        ax.tick_params(colors='#FAFAFA')
                        ax.legend(facecolor='#1F2937', edgecolor='#FAFAFA', labelcolor='#FAFAFA')
                        ax.grid(axis='y', alpha=0.3, color='#FAFAFA')
                        
                        st.pyplot(fig)
                    else:
                        st.error("Could not generate prediction. Try with a different opponent.")
        
        # Clear data button
        if st.button("Clear Current Player", type="secondary"):
            clear_player_data()
            st.rerun()
    
    else:
        st.info("Search for a player and click 'Load Player Data' to get started")


# ==================== PLAYER STATS PAGE ====================
elif page == "Player Stats":
    st.title("Player Season Statistics")
    st.markdown("View detailed season statistics for any NBA player")
    st.caption("Only showing players active in current season")
    
    # Check for redirect from Favorites roster
    initial_player_search = st.session_state.pop('player_stats_search', '')
    
    player_search = st.text_input("Search player:", placeholder="e.g., Anthony Edwards", key="stats_search", value=initial_player_search)
    
    # Track search changes for Stats page
    current_stats_search = player_search.strip()
    if current_stats_search != st.session_state.get('stats_last_search', ''):
        if (st.session_state.get('stats_last_search', '') != "" and current_stats_search != st.session_state.get('stats_last_search', '')) or \
           (st.session_state.stats_player_data is not None and current_stats_search == ""):
            st.session_state.stats_player_data = None
            st.session_state.stats_player_team = None
            st.session_state.stats_selected_player = None
        st.session_state.stats_last_search = current_stats_search

    if player_search:
        matching_players = search_players(player_search, season)
        if matching_players:
            selected_player = st.selectbox("Select player:", matching_players, key="stats_select")
            
            if st.button("Load Stats", type="primary"):
                with st.spinner(f"Loading {selected_player}'s stats..."):
                    p_df, p_team = get_player_game_log(selected_player, season)
                
                if p_df is None or len(p_df) == 0:
                    st.error(f"No data found for {selected_player} in {season}")
                else:
                    # Get player bio first for better branding
                    p_bio = fetch_player_bio(selected_player)
                    if p_bio and p_bio.get('team_abbrev'):
                        p_team = p_bio['team_abbrev']
                    
                    st.session_state.stats_player_data = p_df
                    st.session_state.stats_player_team = p_team
                    st.session_state.stats_selected_player = selected_player
                    st.session_state.stats_player_bio = p_bio
                    st.rerun()
            
            # Show player data if loaded and matches selection
            if st.session_state.stats_player_data is not None and st.session_state.stats_selected_player == selected_player:
                player_df = st.session_state.stats_player_data
                player_team = st.session_state.stats_player_team
                bio = st.session_state.get('stats_player_bio')


                # Show player photo and team logo centered (Standardized layout) - Slightly shifted right for better balance
                spacer1, photo_col, logo_col, spacer2 = st.columns([1.35, 0.8, 0.8, 1.05])
                with photo_col:
                    player_photo = get_player_photo_url(selected_player)
                    if player_photo:
                        st.image(player_photo, width=150)
                with logo_col:
                    team_logo = get_team_logo_url(player_team)
                    if team_logo:
                        st.image(team_logo, width=120)
                
                # Get team record, rank, and full name
                standings_df = get_league_standings(season)
                team_record = "N/A"
                team_rank = "N/A"
                team_conf = "N/A"
                team_full_name = player_team
                if not standings_df.empty:
                    for _, row in standings_df.iterrows():
                        if get_team_abbrev(row['TeamCity']) == player_team:
                            team_record = row['Record']
                            team_rank = row['PlayoffRank']
                            team_conf = row['Conference']
                            team_full_name = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                            break
                # Calculate stats
                ppg = player_df['Points'].mean() if 'Points' in player_df.columns else 0
                rpg = player_df['Rebounds'].mean() if 'Rebounds' in player_df.columns else 0
                apg = player_df['Assists'].mean() if 'Assists' in player_df.columns else 0
                player_stats_line = f"{ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG"
                
                # Calculate shooting splits (for display in shooting_line)
                total_fgm = player_df['FGM'].sum() if 'FGM' in player_df.columns else 0
                total_fga = player_df['FGA'].sum() if 'FGA' in player_df.columns else 0
                fg_pct_display = (total_fgm / total_fga * 100) if total_fga > 0 else 0
                
                total_3pm = player_df['3PM'].sum() if '3PM' in player_df.columns else 0
                total_3pa = player_df['3PA'].sum() if '3PA' in player_df.columns else 0
                three_pct_display = (total_3pm / total_3pa * 100) if total_3pa > 0 else 0
                
                total_ftm = player_df['FTM'].sum() if 'FTM' in player_df.columns else 0
                total_fta = player_df['FTA'].sum() if 'FTA' in player_df.columns else 0
                ft_pct_display = (total_ftm / total_fta * 100) if total_fta > 0 else 0
                
                shooting_line = f"{format_pct(fg_pct_display)} FG% • {format_pct(three_pct_display)} 3P% • {format_pct(ft_pct_display)} FT%"
                
                # Calculate individual record using correct column name
                wins = (player_df['W/L'] == 'W').sum() if 'W/L' in player_df.columns else 0
                losses = (player_df['W/L'] == 'L').sum() if 'W/L' in player_df.columns else 0
                ind_record = f"{wins}-{losses}"
                
                # Player name and position centered
                pos_label = ""
                if bio and bio.get('position'):
                    abbrev_pos = abbreviate_position(bio['position'], selected_player)
                    pos_label = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 1.5rem;'>({abbrev_pos})</span>"

                # Format Bio Info with muted labels and bold values (Awards card style)
                if bio:
                    height = bio.get('height', '-')
                    weight = bio.get('weight', '-')
                    age = bio.get('age', '-')
                    draft_year = bio.get('draft_year', '-')
                    draft_round = bio.get('draft_round', '')
                    draft_num = bio.get('draft_number', '')
                    draft_display = draft_year
                    if draft_round and draft_num and draft_year != 'Undrafted':
                        draft_display = f"{draft_year} R{draft_round}, #{draft_num}"

                    bio_html = f"""<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem; margin-bottom: 12px;">
<span style="color: #9CA3AF;">HT:</span> <span style="color: #FAFAFA; font-weight: bold;">{height}</span> • 
<span style="color: #9CA3AF;">WT:</span> <span style="color: #FAFAFA; font-weight: bold;">{weight} lbs</span> • 
<span style="color: #9CA3AF;">Age:</span> <span style="color: #FAFAFA; font-weight: bold;">{age}</span> • 
<span style="color: #9CA3AF;">Draft:</span> <span style="color: #FAFAFA; font-weight: bold;">{draft_display}</span>
</div>"""
                else:
                    bio_html = f"<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem; margin-bottom: 12px;'>Team: {player_team}</div>"

                st.markdown(f"""<div style="text-align: center; margin-top: 15px;">
<div style="color: #FAFAFA; font-weight: bold; font-size: 1.5rem; margin-bottom: 0px;">{selected_player}{pos_label}</div>
<div style="color: #9CA3AF; font-size: 1.1rem; margin-bottom: 10px;">{team_full_name} <span style="color: #9CA3AF; font-size: 1.1rem;">({team_record} | #{team_rank} in {team_conf})</span></div>
{bio_html}
<div style="display: flex; justify-content: center; gap: 12px; font-size: 0.9rem;">
<div style="background: #374151; padding: 4px 12px; border-radius: 5px;">
<span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{len(player_df)}</span>
</div>
<div style="background: #374151; padding: 4px 12px; border-radius: 5px;">
<span style="color: #9CA3AF;">IND REC:</span> <span style="color: #FAFAFA; font-weight: bold;">{ind_record}</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
                
                # Calculate shooting stats
                if 'FGM' in player_df.columns and 'FGA' in player_df.columns:
                    total_fgm = player_df['FGM'].sum()
                    total_fga = player_df['FGA'].sum()
                    fg_pct = (total_fgm / total_fga * 100) if total_fga > 0 else 0
                else:
                    fg_pct = 0
                
                if '3PM' in player_df.columns and '3PA' in player_df.columns:
                    total_3pm = player_df['3PM'].sum()
                    total_3pa = player_df['3PA'].sum()
                    three_pct = (total_3pm / total_3pa * 100) if total_3pa > 0 else 0
                else:
                    three_pct = 0
                
                if 'FTM' in player_df.columns and 'FTA' in player_df.columns:
                    total_ftm = player_df['FTM'].sum()
                    total_fta = player_df['FTA'].sum()
                    ft_pct = (total_ftm / total_fta * 100) if total_fta > 0 else 0
                else:
                    ft_pct = 0
                
                # Summary stats
                st.markdown("### Season Averages")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("PPG", f"{player_df['Points'].mean():.1f}")
                    st.metric("SPG", f"{player_df['Steals'].mean():.1f}")
                
                with col2:
                    st.metric("RPG", f"{player_df['Rebounds'].mean():.1f}")
                    st.metric("BPG", f"{player_df['Blocks'].mean():.1f}")
                
                with col3:
                    st.metric("APG", f"{player_df['Assists'].mean():.1f}")
                    st.metric("TO", f"{player_df['Turnovers'].mean():.1f}")
                
                with col4:
                    st.metric("FG%", format_pct(fg_pct))
                    st.metric("3P%", format_pct(three_pct))
                
                with col5:
                    st.metric("FT%", format_pct(ft_pct))
                    st.metric("Games", len(player_df))
                
                with col6:
                    # Calculate TS%
                    total_pts = player_df['Points'].sum()
                    ts_pct = (total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                    st.metric("TS%", format_pct(ts_pct))
                    st.metric("MPG", f"{player_df['MIN'].mean():.1f}")
                
                st.markdown("---")
                
                # Game log
                st.markdown("### Game Log")
                
                # Calculate Score column if not present
                if 'Score' not in player_df.columns:
                    # Find all unique teams the player has played for this season explicitly from their matchups
                    unique_teams = set()
                    if 'MATCHUP' in player_df.columns:
                        for match in player_df['MATCHUP']:
                            if len(match) >= 3:
                                unique_teams.add(match[:3])
                    
                    # Fallback to current team
                    if not unique_teams:
                        unique_teams.add(player_team)
                    
                    score_lookup = {}
                    
                    # Fetch game data for ALL teams
                    for team_abbrev in unique_teams:
                        team_game_data = get_team_game_log(team_abbrev, season, num_games=82)
                        
                        if team_game_data is not None and len(team_game_data) > 0:
                            for _, trow in team_game_data.iterrows():
                                game_id = str(trow.get('GAME_ID', ''))
                                if 'PTS' in trow and 'PLUS_MINUS' in trow:
                                    team_pts = int(trow['PTS'])
                                    opp_pts = int(trow['PTS'] - trow['PLUS_MINUS'])
                                    score_lookup[game_id] = f"{team_pts} - {opp_pts}"
                    
                    if score_lookup:
                        player_df['Score'] = player_df.apply(
                            lambda row: score_lookup.get(str(row.get('Game_ID', row.get('GAME_ID', ''))), 'N/A'),
                            axis=1
                        )
                    else:
                        player_df['Score'] = 'N/A'
                
                display_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 
                               'FG', 'FG%', '3P', '3P%', 'FT', 'FT%', 'TS%',
                               'Steals', 'Blocks', 'Turnovers', 'PF']
                available_cols = [col for col in display_cols if col in player_df.columns]
                
                display_df = player_df[available_cols].iloc[::-1].copy()
                
                # Format percentage columns (FG%, 3P%, FT% come as decimals from NBA API, TS% is already calculated as percentage)
                for pct_col in ['FG%', '3P%', 'FT%']:
                    if pct_col in display_df.columns:
                        display_df[pct_col] = display_df[pct_col].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x != 0 else "-")
                if 'TS%' in display_df.columns:
                    display_df['TS%'] = display_df['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) and x != 0 else "-")
                
                # Calculate averages for the whole log
                if not player_df.empty:
                    last_all_df = player_df.copy()
                    
                    # Calculate numeric averages
                    avg_pts = last_all_df['Points'].mean() if 'Points' in last_all_df.columns else 0
                    avg_reb = last_all_df['Rebounds'].mean() if 'Rebounds' in last_all_df.columns else 0
                    avg_ast = last_all_df['Assists'].mean() if 'Assists' in last_all_df.columns else 0
                    avg_stl = last_all_df['Steals'].mean() if 'Steals' in last_all_df.columns else 0
                    avg_blk = last_all_df['Blocks'].mean() if 'Blocks' in last_all_df.columns else 0
                    avg_to = last_all_df['Turnovers'].mean() if 'Turnovers' in last_all_df.columns else 0
                    avg_pf = last_all_df['PF'].mean() if 'PF' in last_all_df.columns else 0
                    
                    # Handle MIN
                    def parse_minutes(m):
                        try:
                            if isinstance(m, str) and ':' in m:
                                return float(m.split(':')[0]) + float(m.split(':')[1])/60
                            return float(m)
                        except: return 0
                    avg_min = last_all_df['MIN'].apply(parse_minutes).mean() if 'MIN' in last_all_df.columns else 0
                    
                    # Calculate shooting % (weighted)
                    def calc_pct(num_col, den_col):
                        num = last_all_df[num_col].sum() if num_col in last_all_df.columns else 0
                        den = last_all_df[den_col].sum() if den_col in last_all_df.columns else 0
                        return (num / den * 100) if den > 0 else 0
                    
                    avg_fg_pct = calc_pct('FGM', 'FGA')
                    avg_3p_pct = calc_pct('3PM', '3PA')
                    avg_ft_pct = calc_pct('FTM', 'FTA')
                    
                    # Calculate TS%
                    total_pts = last_all_df['Points'].sum()
                    total_fga = last_all_df['FGA'].sum() if 'FGA' in last_all_df.columns else 0
                    total_fta = last_all_df['FTA'].sum() if 'FTA' in last_all_df.columns else 0
                    avg_ts_pct = (total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                    
                    # Create average row
                    avg_row = {
                        'GAME_DATE': 'AVERAGE',
                        'MATCHUP': '',
                        'W/L': '',
                        'Score': '',
                        'MIN': f"{avg_min:.1f}",
                        'Points': f"{avg_pts:.1f}",
                        'Rebounds': f"{avg_reb:.1f}",
                        'Assists': f"{avg_ast:.1f}",
                        'Steals': f"{avg_stl:.1f}",
                        'Blocks': f"{avg_blk:.1f}",
                        'Turnovers': f"{avg_to:.1f}",
                        'PF': f"{avg_pf:.1f}",
                        'FG': f"{last_all_df['FGM'].mean():.1f}/{last_all_df['FGA'].mean():.1f}" if 'FGM' in last_all_df.columns else '',
                        'FG%': f"{avg_fg_pct:.1f}%",
                        '3P': f"{last_all_df['3PM'].mean():.1f}/{last_all_df['3PA'].mean():.1f}" if '3PM' in last_all_df.columns else '',
                        '3P%': f"{avg_3p_pct:.1f}%",
                        'FT': f"{last_all_df['FTM'].mean():.1f}/{last_all_df['FTA'].mean():.1f}" if 'FTM' in last_all_df.columns else '',
                        'FT%': f"{avg_ft_pct:.1f}%",
                        'TS%': f"{avg_ts_pct:.1f}%"
                    }
                    
                    # Filter avg_row to only available columns
                    avg_row_filtered = {k: v for k, v in avg_row.items() if k in display_df.columns}
                    avg_df = pd.DataFrame([avg_row_filtered])
                    display_df = pd.concat([display_df, avg_df], ignore_index=True)

                def highlight_avg(row):
                    if row.iloc[0] == 'AVERAGE':
                        return ['background-color: #374151; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                def style_wl(val):
                    if val == 'W':
                        return 'color: #10B981; font-weight: bold'
                    elif val == 'L':
                        return 'color: #EF4444; font-weight: bold'
                    return ''
                
                styled_display_df = display_df.style.apply(highlight_avg, axis=1)
                if 'W/L' in display_df.columns:
                    styled_display_df = styled_display_df.applymap(style_wl, subset=['W/L'])
                
                st.dataframe(
                    styled_display_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.TextColumn("Score", width="small"),
                        "FG": st.column_config.TextColumn("FG", width="small"),
                        "FG%": st.column_config.TextColumn("FG%", width="small"),
                        "3P": st.column_config.TextColumn("3P", width="small"),
                        "3P%": st.column_config.TextColumn("3P%", width="small"),
                        "FT": st.column_config.TextColumn("FT", width="small"),
                        "FT%": st.column_config.TextColumn("FT%", width="small"),
                        "TS%": st.column_config.TextColumn("TS%", width="small"),
                    }
                )
                
                # Last 5 Games Stats
                st.markdown("---")
                st.markdown("### Last 5 Games Averages")
                
                # Get last 5 games (most recent)
                last_5 = player_df.tail(5)
                
                if len(last_5) > 0:
                    # Calculate averages for last 5 games
                    l5_ppg = f"{last_5['Points'].mean():.1f}"
                    l5_rpg = f"{last_5['Rebounds'].mean():.1f}"
                    l5_apg = f"{last_5['Assists'].mean():.1f}"
                    l5_spg = f"{last_5['Steals'].mean():.1f}"
                    l5_bpg = f"{last_5['Blocks'].mean():.1f}"
                    l5_tpg = f"{last_5['Turnovers'].mean():.1f}"
                    
                    # Shooting percentages for last 5
                    l5_fgp = "N/A"
                    l5_3pp = "N/A"
                    l5_ftp = "N/A"
                    
                    if 'FGM' in last_5.columns and 'FGA' in last_5.columns:
                        total_fgm = last_5['FGM'].sum()
                        total_fga = last_5['FGA'].sum()
                        l5_fgp = format_pct(total_fgm / total_fga * 100) if total_fga > 0 else "N/A"
                    
                    if '3PM' in last_5.columns and '3PA' in last_5.columns:
                        total_3pm = last_5['3PM'].sum()
                        total_3pa = last_5['3PA'].sum()
                        l5_3pp = format_pct(total_3pm / total_3pa * 100) if total_3pa > 0 else "N/A"
                    
                    if 'FTM' in last_5.columns and 'FTA' in last_5.columns:
                        total_ftm = last_5['FTM'].sum()
                        total_fta = last_5['FTA'].sum()
                        l5_ftp = format_pct(total_ftm / total_fta * 100) if total_fta > 0 else "N/A"
                    
                    # Display as a compact row
                    l5_cols = st.columns(9)
                    stat_names = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'FT%']
                    stat_values = [l5_ppg, l5_rpg, l5_apg, l5_spg, l5_bpg, l5_tpg, l5_fgp, l5_3pp, l5_ftp]
                    
                    for i, (name, val) in enumerate(zip(stat_names, stat_values)):
                        with l5_cols[i]:
                            st.metric(name, val)
                else:
                    st.info("Not enough games to show last 5 averages.")
                
                # Home vs Away Splits
                st.markdown("---")
                st.markdown("### Home vs Away Splits")
                
                # Determine home/away from MATCHUP column (@ = away, vs. = home)
                home_games = player_df[player_df['MATCHUP'].str.contains(' vs. ', na=False)]
                away_games = player_df[player_df['MATCHUP'].str.contains(' @ ', na=False)]
                
                def calc_split_stats(games_df, split_name):
                    """Calculate stats for a split."""
                    if len(games_df) == 0:
                        return None
                    
                    # Calculate MPG
                    if 'MIN' in games_df.columns:
                        def parse_min_val(min_str):
                            if pd.isna(min_str): return 0
                            try:
                                if ':' in str(min_str):
                                    parts = str(min_str).split(':')
                                    return int(parts[0]) + (int(parts[1])/60)
                                else:
                                    return float(min_str)
                            except: return 0
                        
                        avg_min = games_df['MIN'].apply(parse_min_val).mean()
                        mpg_str = f"{avg_min:.1f}"
                    else:
                        mpg_str = "N/A"
                    
                    # Calculate Record (W-L)
                    if 'W/L' in games_df.columns:
                        wins = len(games_df[games_df['W/L'] == 'W'])
                        losses = len(games_df[games_df['W/L'] == 'L'])
                        record_str = f"{wins}-{losses}"
                    else:
                        record_str = "N/A"
                    
                    stats = {
                        'Split': split_name,
                        'Record': record_str,
                        'GP': str(len(games_df)),
                        'MPG': mpg_str,
                        'PPG': f"{games_df['Points'].mean():.1f}",
                        'RPG': f"{games_df['Rebounds'].mean():.1f}",
                        'APG': f"{games_df['Assists'].mean():.1f}",
                        'SPG': f"{games_df['Steals'].mean():.1f}",
                        'BPG': f"{games_df['Blocks'].mean():.1f}",
                        'TPG': f"{games_df['Turnovers'].mean():.1f}",
                    }
                    
                    # FG%
                    if 'FGM' in games_df.columns and 'FGA' in games_df.columns:
                        total_fgm = games_df['FGM'].sum()
                        total_fga = games_df['FGA'].sum()
                        stats['FG%'] = f"{round((total_fgm / total_fga * 100), 1)}%" if total_fga > 0 else "N/A"
                    else:
                        stats['FG%'] = "N/A"
                    
                    # 3P%
                    if '3PM' in games_df.columns and '3PA' in games_df.columns:
                        total_3pm = games_df['3PM'].sum()
                        total_3pa = games_df['3PA'].sum()
                        stats['3P%'] = f"{round((total_3pm / total_3pa * 100), 1)}%" if total_3pa > 0 else "N/A"
                    else:
                        stats['3P%'] = "N/A"
                    
                    # FT%
                    if 'FTM' in games_df.columns and 'FTA' in games_df.columns:
                        total_ftm = games_df['FTM'].sum()
                        total_fta = games_df['FTA'].sum()
                        stats['FT%'] = f"{round((total_ftm / total_fta * 100), 1)}%" if total_fta > 0 else "N/A"
                    else:
                        stats['FT%'] = "N/A"
                    
                    # TS%
                    if 'FGA' in games_df.columns and 'FTA' in games_df.columns:
                        total_pts = games_df['Points'].sum()
                        total_fga = games_df['FGA'].sum()
                        total_fta = games_df['FTA'].sum()
                        ts = (total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                        stats['TS%'] = f"{round(ts, 1)}%"
                    else:
                        stats['TS%'] = "N/A"
                    
                    return stats
                
                home_stats = calc_split_stats(home_games, "Home")
                away_stats = calc_split_stats(away_games, "Away")
                
                splits_data = []
                if home_stats:
                    splits_data.append(home_stats)
                if away_stats:
                    splits_data.append(away_stats)
                
                if splits_data:
                    splits_df = pd.DataFrame(splits_data)
                    
                    # Style the splits table
                    def highlight_splits(row):
                        if "Home" in str(row['Split']):
                            return ['background-color: #1E3A5F; text-align: left'] * len(row)
                        else:
                            return ['background-color: #3D1E3F; text-align: left'] * len(row)
                    
                    styled_splits = splits_df.style.apply(highlight_splits, axis=1).set_properties(**{'text-align': 'left'})
                    st.dataframe(styled_splits, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data to calculate home/away splits.")
                
                # Win vs Loss Splits
                st.markdown("---")
                st.markdown("### Win vs Loss Splits")
                
                # Filter by W/L column
                if 'W/L' in player_df.columns:
                    win_games = player_df[player_df['W/L'] == 'W']
                    loss_games = player_df[player_df['W/L'] == 'L']
                    
                    win_stats = calc_split_stats(win_games, "Wins")
                    loss_stats = calc_split_stats(loss_games, "Losses")
                    
                    wl_splits_data = []
                    if win_stats:
                        wl_splits_data.append(win_stats)
                    if loss_stats:
                        wl_splits_data.append(loss_stats)
                    
                    if wl_splits_data:
                        wl_splits_df = pd.DataFrame(wl_splits_data)
                        
                        # Style the splits table with Win/Loss colors
                        def highlight_wl_splits(row):
                            if "Win" in str(row['Split']):
                                return ['background-color: #134e37; text-align: left'] * len(row)  # Green tint
                            else:
                                return ['background-color: #4e1313; text-align: left'] * len(row)  # Red tint
                        
                        styled_wl_splits = wl_splits_df.style.apply(highlight_wl_splits, axis=1).set_properties(**{'text-align': 'left'})
                        st.dataframe(styled_wl_splits, use_container_width=True, hide_index=True)
                    else:
                        st.info("Not enough data to calculate win/loss splits.")
                else:
                    st.info("Win/Loss data not available.")
                
                # Conference Splits (East vs West)
                st.markdown("---")
                st.markdown("### Conference Splits")
                
                # Define conference teams
                EAST_TEAMS = {'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS'}
                WEST_TEAMS = {'DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN', 'NOP', 'OKC', 'PHX', 'POR', 'SAC', 'SAS', 'UTA'}
                
                if 'MATCHUP' in player_df.columns:
                    # Extract opponent from matchup (e.g., "BOS vs. LAL" or "BOS @ LAL")
                    def get_opponent(matchup):
                        if '@' in matchup:
                            return matchup.split('@')[-1].strip()
                        elif 'vs.' in matchup:
                            return matchup.split('vs.')[-1].strip()
                        return None
                    
                    player_df['Opponent'] = player_df['MATCHUP'].apply(get_opponent)
                    
                    # Classify games by opponent conference
                    east_games = player_df[player_df['Opponent'].isin(EAST_TEAMS)]
                    west_games = player_df[player_df['Opponent'].isin(WEST_TEAMS)]
                    
                    east_stats = calc_split_stats(east_games, f"vs East ({len(east_games)} G)")
                    west_stats = calc_split_stats(west_games, f"vs West ({len(west_games)} G)")
                    
                    conf_splits_data = []
                    if east_stats:
                        conf_splits_data.append(east_stats)
                    if west_stats:
                        conf_splits_data.append(west_stats)
                    
                    if conf_splits_data:
                        conf_splits_df = pd.DataFrame(conf_splits_data)
                        
                        # Style the splits table with conference colors
                        def highlight_conf_splits(row):
                            if "East" in str(row['Split']):
                                return ['background-color: #1E3A5F; text-align: left'] * len(row)  # Blue tint
                            else:
                                return ['background-color: #5F1E1E; text-align: left'] * len(row)  # Red tint
                        
                        styled_conf_splits = conf_splits_df.style.apply(highlight_conf_splits, axis=1).set_properties(**{'text-align': 'left'})
                        st.dataframe(styled_conf_splits, use_container_width=True, hide_index=True)
                    else:
                        st.info("Not enough data to calculate conference splits.")
                else:
                    st.info("Matchup data not available for conference splits.")
                
                # Splits vs Winning Teams (≥ .500 win percentage)
                st.markdown("---")
                st.markdown("### Splits vs Winning Teams")
                
                # Fetch standings to get team win percentages
                standings_data = get_league_standings(season)
                
                if not standings_data.empty and 'MATCHUP' in player_df.columns:
                    # Build win percentage lookup by team abbreviation
                    team_win_pcts = {}
                    for _, row in standings_data.iterrows():
                        city = row.get('TeamCity', '')
                        team_abbrev = get_team_abbrev(city)
                        if team_abbrev:
                            win_pct = row.get('WinPct', 0)
                            # Handle string format if needed
                            if isinstance(win_pct, str):
                                try:
                                    win_pct = float(win_pct)
                                except:
                                    win_pct = 0
                            team_win_pcts[team_abbrev] = win_pct
                    
                    if team_win_pcts:
                        # Extract opponent from matchup column
                        def get_opponent_abbrev(matchup):
                            if '@' in matchup:
                                return matchup.split('@')[-1].strip()
                            elif 'vs.' in matchup:
                                return matchup.split('vs.')[-1].strip()
                            return None
                        
                        player_df['OppAbbrev'] = player_df['MATCHUP'].apply(get_opponent_abbrev)
                        player_df['OppWinPct'] = player_df['OppAbbrev'].map(team_win_pcts)
                        
                        # Split games by opponent win percentage
                        # Winning teams: >= 0.500
                        winning_opp_games = player_df[player_df['OppWinPct'] >= 0.500]
                        losing_opp_games = player_df[player_df['OppWinPct'] < 0.500]
                        
                        winning_opp_stats = calc_split_stats(winning_opp_games, f"vs ≥.500 ({len(winning_opp_games)} G)")
                        losing_opp_stats = calc_split_stats(losing_opp_games, f"vs <.500 ({len(losing_opp_games)} G)")
                        
                        record_splits_data = []
                        if winning_opp_stats:
                            record_splits_data.append(winning_opp_stats)
                        if losing_opp_stats:
                            record_splits_data.append(losing_opp_stats)
                        
                        if record_splits_data:
                            record_splits_df = pd.DataFrame(record_splits_data)
                            
                            # Style the splits table - winning teams in orange/gold, losing in lighter shade
                            def highlight_record_splits(row):
                                if "≥.500" in str(row['Split']):
                                    return ['background-color: #5F4B1E; text-align: left'] * len(row)  # Gold/bronze tint
                                else:
                                    return ['background-color: #2E3D4E; text-align: left'] * len(row)  # Slate/gray tint
                            
                            styled_record_splits = record_splits_df.style.apply(highlight_record_splits, axis=1).set_properties(**{'text-align': 'left'})
                            st.dataframe(styled_record_splits, use_container_width=True, hide_index=True)
                        else:
                            st.info("Not enough data to calculate winning team splits.")
                    else:
                        st.info("Could not determine team win percentages.")
                else:
                    st.info("Standings data not available for winning team splits.")
                
                # Stats vs Specific Team
                st.markdown("---")
                st.markdown("### Stats vs Specific Teams")
                
                # Fetch def ratings for consistency with Predictions page
                with st.spinner("Fetching team ratings..."):
                    team_def_ratings = get_current_defensive_ratings(season)
                
                all_teams_list = sorted(list(team_def_ratings.keys())) if team_def_ratings else ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
                
                available_opponents = [t for t in all_teams_list if t != player_team]
                
                opp_col1, opp_col2 = st.columns([0.9, 0.1])
                with opp_col1:
                    selected_opp = st.selectbox(
                        "Select Team:", 
                        available_opponents, 
                        key="stats_vs_team_select",
                        format_func=lambda x: TEAM_NAME_MAP.get(x, x)
                    )
                with opp_col2:
                    opp_logo = get_team_logo_url(selected_opp)
                    if opp_logo:
                        st.image(opp_logo, width=50)
                
                if selected_opp:
                    opp_rating = team_def_ratings.get(selected_opp, "N/A")
                    st.caption(f"Defensive Rating: **{opp_rating}** (Lower is better defense)")
                    
                    # Ensure opponent column exists (it should from conference splits logic above)
                    if 'Opponent' not in player_df.columns and 'MATCHUP' in player_df.columns:
                        def get_opp_abbrev(matchup):
                            if '@' in matchup: return matchup.split('@')[-1].strip()
                            elif 'vs.' in matchup: return matchup.split('vs.')[-1].strip()
                            return None
                        player_df['Opponent'] = player_df['MATCHUP'].apply(get_opp_abbrev)
                    
                    opp_games = player_df[player_df['Opponent'] == selected_opp]
                    
                    # Get opponent record and rank
                    opp_record_str = "N/A"
                    opp_rank_str = "N/A"
                    opp_conf_str = "N/A"
                    if not standings_df.empty:
                        for _, row in standings_df.iterrows():
                            if get_team_abbrev(row['TeamCity']) == selected_opp:
                                opp_record_str = row['Record']
                                opp_rank_str = row['PlayoffRank']
                                opp_conf_str = row['Conference']
                                break
                    
                    if not opp_games.empty:
                        opp_full = TEAM_NAME_MAP.get(selected_opp, selected_opp)
                        st.markdown(f"### Games vs {opp_full} <span style='color: #6B7280; font-size: 1.5rem;'> (#{opp_rank_str} {opp_conf_str}, {opp_record_str})</span>", unsafe_allow_html=True)

                    
                    if len(opp_games) > 0:
                        # Calculate averages against this opponent
                        num_games = len(opp_games)
                        
                        # Calculate W-L record against this opponent
                        if 'W/L' in opp_games.columns:
                            wins_vs = len(opp_games[opp_games['W/L'] == 'W'])
                            losses_vs = len(opp_games[opp_games['W/L'] == 'L'])
                            WL_record = f"{wins_vs}-{losses_vs}"
                        else:
                            WL_record = "N/A"
                        
                        # Calculate basic stats
                        avg_points_vs = round(opp_games['Points'].mean(), 1)
                        avg_rebounds_vs = round(opp_games['Rebounds'].mean(), 1)
                        avg_assists_vs = round(opp_games['Assists'].mean(), 1)
                        avg_steals_vs = round(opp_games['Steals'].mean(), 1)
                        avg_blocks_vs = round(opp_games['Blocks'].mean(), 1)
                        avg_turnovers_vs = round(opp_games['Turnovers'].mean(), 1)
                        
                        # Calculate shooting stats
                        if 'FGM' in opp_games.columns and 'FGA' in opp_games.columns:
                            total_fgm_vs = opp_games['FGM'].sum()
                            total_fga_vs = opp_games['FGA'].sum()
                            fg_pct_vs = round((total_fgm_vs / total_fga_vs * 100), 1) if total_fga_vs > 0 else 0
                            avg_fgm_vs = round(opp_games['FGM'].mean(), 1)
                            avg_fga_vs = round(opp_games['FGA'].mean(), 1)
                            avg_fg_vs = f"{avg_fgm_vs:.1f}/{avg_fga_vs:.1f}"
                        else:
                            avg_fg_vs = "N/A"
                            fg_pct_vs = "N/A"
                        
                        if '3PM' in opp_games.columns and '3PA' in opp_games.columns:
                            total_3pm_vs = opp_games['3PM'].sum()
                            total_3pa_vs = opp_games['3PA'].sum()
                            three_pct_vs = round((total_3pm_vs / total_3pa_vs * 100), 1) if total_3pa_vs > 0 else 0
                            avg_3pm_vs = round(opp_games['3PM'].mean(), 1)
                            avg_3pa_vs = round(opp_games['3PA'].mean(), 1)
                            avg_3p_vs = f"{avg_3pm_vs:.1f}/{avg_3pa_vs:.1f}"
                        else:
                            avg_3p_vs = "N/A"
                            three_pct_vs = "N/A"
                        
                        if 'FTM' in opp_games.columns and 'FTA' in opp_games.columns:
                            total_ftm_vs = opp_games['FTM'].sum()
                            total_fta_vs = opp_games['FTA'].sum()
                            ft_pct_vs = round((total_ftm_vs / total_fta_vs * 100), 1) if total_fta_vs > 0 else 0
                            avg_ftm_vs = round(opp_games['FTM'].mean(), 1)
                            avg_fta_vs = round(opp_games['FTA'].mean(), 1)
                            avg_ft_vs = f"{avg_ftm_vs:.1f}/{avg_fta_vs:.1f}"
                        else:
                            avg_ft_vs = "N/A"
                            ft_pct_vs = "N/A"
                        
                        # Calculate minutes
                        if 'MIN' in opp_games.columns:
                            def parse_minutes_simple(min_str):
                                if pd.isna(min_str):
                                    return 0
                                try:
                                    if ':' in str(min_str):
                                        parts = str(min_str).split(':')
                                        return int(parts[0])  # Just take minutes
                                    else:
                                        return float(min_str)
                                except:
                                    return 0
                            
                            opp_games_copy = opp_games.copy()
                            opp_games_copy['MIN_NUM'] = opp_games_copy['MIN'].apply(parse_minutes_simple)
                            avg_minutes_vs = round(opp_games_copy['MIN_NUM'].mean(), 1)
                        else:
                            avg_minutes_vs = "N/A"
                        
                        # Calculate TS%
                        if 'FGA' in opp_games.columns and 'FTA' in opp_games.columns:
                            total_points_vs = opp_games['Points'].sum()
                            total_fga_vs_ts = opp_games['FGA'].sum()
                            total_fta_vs_ts = opp_games['FTA'].sum()
                            ts_pct_vs = round((total_points_vs / (2 * (total_fga_vs_ts + 0.44 * total_fta_vs_ts)) * 100), 1) if (total_fga_vs_ts + 0.44 * total_fta_vs_ts) > 0 else 0
                        else:
                            ts_pct_vs = "N/A"
                        
                        # Show individual game results against this opponent with averages row
                        st.markdown(f"**Games Played: {num_games}** | **Record: {WL_record}**")
                        
                        # Get individual games - include W/L and Score
                        vs_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 
                                  'Steals', 'Blocks', 'Turnovers', 'FG', '3P', 'FT', 'TS%']
                        vs_opponent_display = opp_games[[c for c in vs_cols if c in opp_games.columns]].iloc[::-1].copy()
                        
                        # Format TS% for individual games
                        if 'TS%' in vs_opponent_display.columns:
                            vs_opponent_display['TS%'] = vs_opponent_display['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                        
                        # Create averages row
                        averages_row_vs = {
                            'GAME_DATE': 'AVG vs ' + selected_opp,
                            'MATCHUP': f'({num_games} games)',
                            'W/L': '',
                            'Score': '',
                            'MIN': f"{avg_minutes_vs:.1f}" if avg_minutes_vs != "N/A" else "N/A",
                            'Points': f"{avg_points_vs:.1f}",
                            'Rebounds': f"{avg_rebounds_vs:.1f}",
                            'Assists': f"{avg_assists_vs:.1f}",
                            'Steals': f"{avg_steals_vs:.1f}",
                            'Blocks': f"{avg_blocks_vs:.1f}",
                            'Turnovers': f"{avg_turnovers_vs:.1f}",
                            'FG': f"{fg_pct_vs:.1f}%" if fg_pct_vs != "N/A" else "N/A",
                            '3P': f"{three_pct_vs:.1f}%" if three_pct_vs != "N/A" else "N/A",
                            'FT': f"{ft_pct_vs:.1f}%" if ft_pct_vs != "N/A" else "N/A",
                            'TS%': f"{ts_pct_vs:.1f}%" if isinstance(ts_pct_vs, (int, float)) else ts_pct_vs
                        }
                        
                        # Add the averages row to the dataframe
                        averages_df_row_vs = pd.DataFrame([averages_row_vs])
                        
                        # Combine with individual games
                        combined_display = pd.concat([vs_opponent_display, averages_df_row_vs], ignore_index=True)
                        
                        # Highlight the averages row and W/L column
                        def highlight_vs_average_row(row):
                            if 'AVG vs ' in str(row['GAME_DATE']):
                                return ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
                            else:
                                return [''] * len(row)
                        
                        def style_opponent_wl(val):
                            if val == 'W':
                                return 'color: #10B981; font-weight: bold'
                            elif val == 'L':
                                return 'color: #EF4444; font-weight: bold'
                            return ''
                        
                        # Apply row styling
                        styled_df = combined_display.style.apply(highlight_vs_average_row, axis=1)
                        
                        # Apply W/L column styling if present
                        if 'W/L' in combined_display.columns:
                            styled_df = styled_df.applymap(style_opponent_wl, subset=['W/L'])
                        
                        # Display the table with styling
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    else:
                        opp_full = TEAM_NAME_MAP.get(selected_opp, selected_opp)
                        st.info(f"**{selected_player}** has not played against the **{opp_full}** yet this season.")


# ==================== FAVORITES PAGE ====================
elif page == "Favorites":
    st.title("Your Favorites")
    
    if not is_authenticated:
        st.warning("Please login to view your favorites!")
        auth.show_login_button()
    else:
        # Get current user info
        current_user = auth.get_current_user()
        if current_user:
            st.info(f"Logged in as: **{current_user.email}**")
        
        # Fetch schedule and standings data for upcoming games
        nba_schedule = get_nba_schedule()
        standings_df = get_league_standings(season)
        
        # Tab selection logic
        requested_tab = st.session_state.get('favorites_requested_tab', "Favorite Players")
        if requested_tab == "Favorite Teams":
            tab_teams, tab_players = st.tabs(["Favorite Teams", "Favorite Players"])
            # Clear the override after the first render so it doesn't stick
            # But wait, if we clear it here, it might reset on next interactive element?
            # Actually, let's clear it ONLY when navigating AWAY or if we want it to persist during this visit.
            # User said "redirect me", implying a one-time preference.
        else:
            tab_players, tab_teams = st.tabs(["Favorite Players", "Favorite Teams"])
        
        with tab_players:
            # Header with add button
            header_col, add_col = st.columns([4, 1])
            with header_col:
                st.markdown("### Your Favorite Players")
            with add_col:
                add_player_expanded = st.button("➕ Add", key="add_player_btn", use_container_width=True)
            
            # Add player form
            if add_player_expanded or st.session_state.get('show_add_player', False):
                st.session_state['show_add_player'] = True
                with st.container():
                    st.markdown("---")
                    player_search = st.text_input("Search player:", placeholder="e.g., LeBron James", key="new_player_search")
                    
                    # Show matching players as user types
                    if player_search:
                        matching_players = search_players(player_search, season)
                        if matching_players:
                            selected_player = st.selectbox(
                                "Select player:", 
                                matching_players, 
                                key="new_player_select"
                            )
                            add_col1, add_col2 = st.columns(2)
                            with add_col1:
                                if st.button("Add Player", use_container_width=True, type="primary"):
                                    if auth.add_favorite_player(selected_player):
                                        st.toast(f"Added {selected_player} to favorites!")
                                        st.session_state['show_add_player'] = False
                                        st.rerun()
                                    else:
                                        st.warning(f"{selected_player} is already in favorites!")
                            with add_col2:
                                if st.button("Cancel", use_container_width=True):
                                    st.session_state['show_add_player'] = False
                                    st.rerun()
                        else:
                            st.info("No players found matching your search.")
                            if st.button("Cancel", use_container_width=True):
                                st.session_state['show_add_player'] = False
                                st.rerun()
                    else:
                        st.caption("Start typing to see matching players...")
                        if st.button("Cancel", use_container_width=True):
                            st.session_state['show_add_player'] = False
                            st.rerun()
                    st.markdown("---")
            
            favorite_players = auth.get_favorite_players()
            
            if favorite_players:
                st.write(f"You have **{len(favorite_players)}** favorite player(s):")
                st.markdown("---")
                
                # Need to run queue for recent/upcoming games to optimize? No, let's keep it simple first
                
                for player in favorite_players:
                    # Fetch bio and stats
                    bio = fetch_player_bio(player)
                    
                    # Create a nice card container
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
                            border-radius: 12px;
                            padding: 24px;
                            border: 1px solid #374151;
                            margin-bottom: 20px;
                        ">
                        """, unsafe_allow_html=True)
                        
                        # Top Row: Info + Actions
                        col_img, col_logo, col_info, col_actions = st.columns([1.0, 1.0, 2.5, 1])
                        
                        with col_img:
                            if bio and bio.get('player_id'):
                                headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{bio['player_id']}.png"
                                st.image(headshot_url, width=110)
                            else:
                                st.write("👤")

                        with col_logo:
                            team_abbrev = bio.get('team_abbrev') if bio else None
                            if team_abbrev:
                                logo = get_team_logo_url(team_abbrev)
                                if logo:
                                    st.image(logo, width=110)
                                else:
                                    st.write(team_abbrev)
                        
                        with col_info:
                            st.markdown(f"### {player}")
                            
                            # Bio Stats
                            if bio:
                                st.markdown(f"""
                                <div style="display: flex; gap: 15px; flex-wrap: wrap; color: #9CA3AF; font-size: 0.9rem; margin-top: 8px;">
                                    <span>Height: <strong style="color: #FAFAFA;">{bio.get('height', '-')}</strong></span>
                                    <span>Weight: <strong style="color: #FAFAFA;">{bio.get('weight', '-')}</strong></span>
                                    <span>Age: <strong style="color: #FAFAFA;">{bio.get('age', '-')}</strong></span>
                                    <span>Draft: <strong style="color: #FAFAFA;">{bio.get('draft_year', '-')} (R{bio.get('draft_round', '-')}, #{bio.get('draft_number', '-')})</strong></span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_actions:
                            # Reordering buttons
                            ord_c1, ord_c2 = st.columns(2)
                            with ord_c1:
                                if st.button("▲", key=f"fav_up_{player}", use_container_width=True, help="Move Up"):
                                    auth.reorder_favorite_player(player, "up")
                                    st.rerun()
                            with ord_c2:
                                if st.button("▼", key=f"fav_down_{player}", use_container_width=True, help="Move Down"):
                                    auth.reorder_favorite_player(player, "down")
                                    st.rerun()
                            
                            st.write("") # Spacer
                            
                            if st.button("📊 Analyze", key=f"fav_analyze_{player}", use_container_width=True, type="primary"):
                                st.session_state['auto_load_player'] = player
                                st.session_state['pending_nav_target'] = "Predictions"
                                st.rerun()
                            
                            if st.button("❌ Remove", key=f"fav_remove_{player}", use_container_width=True):
                                auth.remove_favorite_player(player)
                                st.toast(f"Removed {player} from favorites")
                                st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Expanders for Games
                        # Recent Games - matching Live Predictions display
                        with st.expander(f"Recent Games - {player}"):
                            player_df, player_team = get_player_game_log(player, season)
                            if player_df is not None and not player_df.empty:
                                # Calculate Score column if missing (same as Live Predictions)
                                if 'Score' not in player_df.columns:
                                    # Get unique teams player has played for
                                    unique_teams = set()
                                    if 'MATCHUP' in player_df.columns:
                                        for match in player_df['MATCHUP']:
                                            if len(match) >= 3:
                                                unique_teams.add(match[:3])
                                    if not unique_teams and player_team:
                                        unique_teams.add(player_team)
                                    
                                    score_lookup = {}
                                    for team_abbrev in unique_teams:
                                        team_game_data = get_team_game_log(team_abbrev, season, num_games=82)
                                        if team_game_data is not None and len(team_game_data) > 0:
                                            for _, trow in team_game_data.iterrows():
                                                game_id = str(trow.get('GAME_ID', ''))
                                                if 'PTS' in trow and 'PLUS_MINUS' in trow:
                                                    team_pts = int(trow['PTS'])
                                                    opp_pts = int(trow['PTS'] - trow['PLUS_MINUS'])
                                                    score_lookup[game_id] = f"{team_pts} - {opp_pts}"
                                    
                                    if score_lookup:
                                        player_df['Score'] = player_df.apply(
                                            lambda row: score_lookup.get(str(row.get('Game_ID', row.get('GAME_ID', ''))), 'N/A'),
                                            axis=1
                                        )
                                    else:
                                        player_df['Score'] = 'N/A'
                                
                                # Get MOST RECENT 5 games (tail, then reverse for newest first)
                                recent = player_df.tail(5).iloc[::-1].copy()
                                
                                # Display columns matching Live Predictions + shooting metrics
                                display_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 
                                               'FG', 'FG%', '3P', '3P%', 'FT', 'FT%', 'TS%',
                                               'Steals', 'Blocks', 'Turnovers', 'PF']
                                available = [c for c in display_cols if c in recent.columns]
                                recent_display = recent[available].copy()
                                
                                # Format percentage columns (FG%, 3P%, FT% come as decimals from NBA API, TS% is already calculated as percentage)
                                for pct_col in ['FG%', '3P%', 'FT%']:
                                    if pct_col in recent_display.columns:
                                        recent_display[pct_col] = recent_display[pct_col].apply(
                                            lambda x: f"{x*100:.1f}%" if pd.notna(x) and x != 0 else "0"
                                        )
                                if 'TS%' in recent_display.columns:
                                    recent_display['TS%'] = recent_display['TS%'].apply(
                                        lambda x: f"{x:.1f}%" if pd.notna(x) and x != 0 else "0"
                                    )
                                
                                # Rename columns
                                rename_map = {
                                    'GAME_DATE': 'Date', 'MATCHUP': 'Matchup', 
                                    'Points': 'PTS', 'Assists': 'AST', 'Rebounds': 'REB',
                                    'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TO'
                                }
                                recent_display = recent_display.rename(columns=rename_map)
                                
                                # Calculate averages for the last 5 games
                                if not recent.empty:
                                    last_5_df = recent.copy()
                                    
                                    # Calculate numeric averages
                                    avg_pts = last_5_df['Points'].mean() if 'Points' in last_5_df.columns else 0
                                    avg_reb = last_5_df['Rebounds'].mean() if 'Rebounds' in last_5_df.columns else 0
                                    avg_ast = last_5_df['Assists'].mean() if 'Assists' in last_5_df.columns else 0
                                    avg_stl = last_5_df['Steals'].mean() if 'Steals' in last_5_df.columns else 0
                                    avg_blk = last_5_df['Blocks'].mean() if 'Blocks' in last_5_df.columns else 0
                                    avg_to = last_5_df['Turnovers'].mean() if 'Turnovers' in last_5_df.columns else 0
                                    avg_pf = last_5_df['PF'].mean() if 'PF' in last_5_df.columns else 0
                                    
                                    # Handle MIN
                                    def parse_minutes(m):
                                        try:
                                            if isinstance(m, str) and ':' in m:
                                                return float(m.split(':')[0]) + float(m.split(':')[1])/60
                                            return float(m)
                                        except: return 0
                                    avg_min = last_5_df['MIN'].apply(parse_minutes).mean() if 'MIN' in last_5_df.columns else 0
                                    
                                    # Calculate shooting % (weighted)
                                    def calc_pct(num_col, den_col):
                                        num = last_5_df[num_col].sum() if num_col in last_5_df.columns else 0
                                        den = last_5_df[den_col].sum() if den_col in last_5_df.columns else 0
                                        return (num / den * 100) if den > 0 else 0
                                    
                                    avg_fg_pct = calc_pct('FGM', 'FGA')
                                    avg_3p_pct = calc_pct('3PM', '3PA')
                                    avg_ft_pct = calc_pct('FTM', 'FTA')
                                    
                                    # Calculate TS%
                                    total_pts = last_5_df['Points'].sum()
                                    total_fga = last_5_df['FGA'].sum() if 'FGA' in last_5_df.columns else 0
                                    total_fta = last_5_df['FTA'].sum() if 'FTA' in last_5_df.columns else 0
                                    avg_ts_pct = (total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                                    
                                    # Create average row
                                    avg_row = {
                                        'Date': 'AVERAGE',
                                        'Matchup': '',
                                        'W/L': '',
                                        'Score': '',
                                        'MIN': f"{avg_min:.1f}",
                                        'PTS': f"{avg_pts:.1f}",
                                        'REB': f"{avg_reb:.1f}",
                                        'AST': f"{avg_ast:.1f}",
                                        'STL': f"{avg_stl:.1f}",
                                        'BLK': f"{avg_blk:.1f}",
                                        'TO': f"{avg_to:.1f}",
                                        'PF': f"{avg_pf:.1f}",
                                        'FG': f"{avg_fg_pct:.1f}%",
                                        '3P': f"{avg_3p_pct:.1f}%",
                                        'FT': f"{avg_ft_pct:.1f}%",
                                        'TS%': f"{avg_ts_pct:.1f}%"
                                    }
                                    
                                    # Append average row
                                    avg_df = pd.DataFrame([avg_row])
                                    recent_display = pd.concat([recent_display, avg_df], ignore_index=True)
                                
                                # Style W/L
                                def highlight_avg(row):
                                    if row.iloc[0] == 'AVERAGE':
                                        return ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
                                    return [''] * len(row)

                                def color_wl(val):
                                    if val == 'W': return 'color: #10B981; font-weight: bold'
                                    elif val == 'L': return 'color: #EF4444; font-weight: bold'
                                    return ''
                                
                                # Drop % columns for display since we formatted the main columns
                                recent_display = recent_display.drop(columns=['FG%', '3P%', 'FT%'], errors='ignore')
                                
                                if 'W/L' in recent_display.columns:
                                    # Combine styling
                                    styled = recent_display.style.apply(highlight_avg, axis=1)
                                    styled = styled.applymap(color_wl, subset=['W/L'])
                                else:
                                    styled = recent_display.style.apply(highlight_avg, axis=1)
                                    
                                st.dataframe(
                                    styled, 
                                    use_container_width=True, 
                                    hide_index=True,
                                    column_config={
                                        "Date": st.column_config.TextColumn("Date", width="medium"),
                                        "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                                        "W/L": st.column_config.TextColumn("W/L", width="small"),
                                        "Score": st.column_config.TextColumn("Score", width="small"),
                                        "FG": st.column_config.TextColumn("FG", width="small"),
                                        "3P": st.column_config.TextColumn("3P", width="small"),
                                        "FT": st.column_config.TextColumn("FT", width="small"),
                                        "TS%": st.column_config.TextColumn("TS%", width="small"),
                                    }
                                )
                            else:
                                st.info("No recent games found.")

                        # Upcoming Games
                        with st.expander(f"Upcoming Games - {player}"):
                            if bio and bio.get('team_abbrev'):
                                upcoming = get_team_upcoming_games(bio['team_abbrev'], nba_schedule, standings_df, num_games=5)
                                if upcoming:
                                    st.caption("Click an upcoming game to predict performance")
                                    cols = st.columns(min(len(upcoming), 5))
                                    for i, game in enumerate(upcoming):
                                        with cols[i]:
                                            home_away = "vs" if game['is_home'] else "@"
                                            opp_rank = game.get('opponent_rank', 0)
                                            opp_conf = game.get('opponent_conference', '')
                                            if opp_rank:
                                                label = f"{game['date']} {home_away} {game['opponent']} (#{opp_rank} {opp_conf})"
                                            else:
                                                label = f"{game['date']} {home_away} {game['opponent']}"
                                            
                                            if st.button(label, key=f"fav_upcoming_{player}_{i}_{game['opponent']}", use_container_width=True):
                                                st.session_state["auto_load_player"] = player
                                                st.session_state["redirect_to_predictions"] = player
                                                st.session_state["selected_upcoming_opponent"] = game['opponent']
                                                st.session_state['pending_nav_target'] = "Predictions"
                                                st.rerun()
                                else:
                                    st.info("No upcoming games found.")
                            else:
                                st.info("Could not determine team for upcoming games.")
                
                st.write("") # Spacer
            else:
                render_empty_state("No favorite players yet! Click ➕ Add above to add some.", "")
        
        with tab_teams:
            # Header with add button
            header_col, add_col = st.columns([4, 1])
            with header_col:
                st.markdown("### Your Favorite Teams")
            with add_col:
                add_team_expanded = st.button("➕ Add", key="add_team_btn", use_container_width=True)
            
            favorite_teams = auth.get_favorite_teams()
            team_ratings = get_current_defensive_ratings(season)
            team_ratings_full = get_team_ratings_with_ranks(season)  # Get full ratings with ranks
            
            # All team abbreviations for dropdown
            all_team_abbrevs = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
                               "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
                               "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
            
            # Add team form
            if add_team_expanded or st.session_state.get('show_add_team', False):
                st.session_state['show_add_team'] = True
                with st.container():
                    st.markdown("---")
                    # Filter out teams already in favorites
                    available_teams = [t for t in all_team_abbrevs if t not in favorite_teams]
                    if available_teams:
                        new_team = st.selectbox(
                            "Select team to add:", 
                            available_teams, 
                            key="new_team_select",
                            format_func=lambda x: TEAM_NAME_MAP.get(x, x)
                        )
                        add_col1, add_col2 = st.columns(2)
                        with add_col1:
                            if st.button("Add Team", use_container_width=True, type="primary", key="confirm_add_team"):
                                if auth.add_favorite_team(new_team):
                                    st.toast(f"Added {new_team} to favorite teams!")
                                    st.session_state['show_add_team'] = False
                                    st.rerun()
                        with add_col2:
                            if st.button("Cancel", use_container_width=True, key="cancel_add_team"):
                                st.session_state['show_add_team'] = False
                                st.rerun()
                    else:
                        st.info("You're already watching all 30 teams!")
                        if st.button("Close", use_container_width=True):
                            st.session_state['show_add_team'] = False
                            st.rerun()
                    st.markdown("---")
            
            # Fetch standings data once for all teams
            standings_df = get_league_standings(season)
            
            if favorite_teams:
                st.write(f"You have **{len(favorite_teams)}** favorite team(s):")
                st.markdown("---")
                
                # Team names and city mapping
                team_info = {
                    "LAL": {"name": "Lakers", "city": "Los Angeles"},
                    "GSW": {"name": "Warriors", "city": "Golden State"},
                    "MIL": {"name": "Bucks", "city": "Milwaukee"},
                    "BOS": {"name": "Celtics", "city": "Boston"},
                    "PHX": {"name": "Suns", "city": "Phoenix"},
                    "MIA": {"name": "Heat", "city": "Miami"},
                    "DEN": {"name": "Nuggets", "city": "Denver"},
                    "PHI": {"name": "76ers", "city": "Philadelphia"},
                    "LAC": {"name": "Clippers", "city": "Los Angeles"},
                    "DAL": {"name": "Mavericks", "city": "Dallas"},
                    "MEM": {"name": "Grizzlies", "city": "Memphis"},
                    "CLE": {"name": "Cavaliers", "city": "Cleveland"},
                    "NYK": {"name": "Knicks", "city": "New York"},
                    "BKN": {"name": "Nets", "city": "Brooklyn"},
                    "ATL": {"name": "Hawks", "city": "Atlanta"},
                    "CHI": {"name": "Bulls", "city": "Chicago"},
                    "TOR": {"name": "Raptors", "city": "Toronto"},
                    "SAC": {"name": "Kings", "city": "Sacramento"},
                    "MIN": {"name": "Timberwolves", "city": "Minnesota"},
                    "NOP": {"name": "Pelicans", "city": "New Orleans"},
                    "OKC": {"name": "Thunder", "city": "Oklahoma City"},
                    "POR": {"name": "Trail Blazers", "city": "Portland"},
                    "UTA": {"name": "Jazz", "city": "Utah"},
                    "IND": {"name": "Pacers", "city": "Indiana"},
                    "WAS": {"name": "Wizards", "city": "Washington"},
                    "ORL": {"name": "Magic", "city": "Orlando"},
                    "CHA": {"name": "Hornets", "city": "Charlotte"},
                    "DET": {"name": "Pistons", "city": "Detroit"},
                    "HOU": {"name": "Rockets", "city": "Houston"},
                    "SAS": {"name": "Spurs", "city": "San Antonio"}
                }
                
                # Load Coach-to-Team mapping from CSV
                coach_team_map = {}
                try:
                    coaches_df = pd.read_csv('nbacoaches.csv')
                    for _, row in coaches_df.iterrows():
                        coach_name = row['NAME']
                        team_abbrev = row['ABBREV']
                        coach_team_map[team_abbrev] = {"name": coach_name}
                except Exception:
                    pass

                for team in favorite_teams:
                    rating = team_ratings.get(team, "N/A")
                    info = team_info.get(team, {"name": team, "city": ""})
                    team_name = info["name"]
                    team_city = info["city"]
                    
                    # Get offense/defense ratings with ranks
                    team_rtg = team_ratings_full.get(team, {})
                    off_rtg = team_rtg.get('off_rtg', 'N/A')
                    def_rtg = team_rtg.get('def_rtg', 'N/A')
                    off_rank = team_rtg.get('off_rank', 'N/A')
                    def_rank = team_rtg.get('def_rank', 'N/A')
                    
                    # Find team in standings
                    team_standing = None
                    if not standings_df.empty:
                        # Match by abbreviation directly
                        matching = standings_df[standings_df['TeamAbbrev'] == team]
                        if not matching.empty:
                            team_standing = matching.iloc[0]
                    
                    # Build team info display
                    record = team_standing['Record'] if team_standing is not None else "N/A"
                    conf = team_standing['Conference'] if team_standing is not None else "N/A"
                    rank = int(team_standing['PlayoffRank']) if team_standing is not None else "N/A"
                    l10 = team_standing['L10'] if team_standing is not None else "N/A"
                    streak = team_standing['strCurrentStreak'] if team_standing is not None else "N/A"
                    home = team_standing['HOME'] if team_standing is not None else "N/A"
                    road = team_standing['ROAD'] if team_standing is not None else "N/A"
                    division = team_standing['Division'] if team_standing is not None else "N/A"
                    div_rank = team_standing.get('DivisionRank', 'N/A') if team_standing is not None else "N/A"
                    
                    # Calculate record vs winning teams (≥ .500) and losing teams (< .500)
                    vs_winning = "N/A"
                    vs_losing = "N/A"
                    try:
                        # Build win percentage lookup from standings
                        team_win_pcts = {}
                        for _, s_row in standings_df.iterrows():
                            opp_city = s_row.get('TeamCity', '')
                            opp_abbrev = get_team_abbrev(opp_city)
                            if opp_abbrev:
                                win_pct = s_row.get('WinPct', 0)
                                if isinstance(win_pct, str):
                                    try:
                                        win_pct = float(win_pct)
                                    except:
                                        win_pct = 0
                                team_win_pcts[opp_abbrev] = win_pct
                        
                        # Fetch full season game log for this team
                        full_games = get_team_game_log(team, season, num_games=100)
                        if full_games is not None and len(full_games) > 0 and team_win_pcts:
                            # Extract opponent from matchup (e.g., "LAL @ BOS" or "LAL vs. BOS")
                            def get_opp_from_matchup(matchup):
                                if '@' in matchup:
                                    return matchup.split('@')[-1].strip()
                                elif 'vs.' in matchup:
                                    return matchup.split('vs.')[-1].strip()
                                return None
                            
                            full_games['OppAbbrev'] = full_games['MATCHUP'].apply(get_opp_from_matchup)
                            full_games['OppWinPct'] = full_games['OppAbbrev'].map(team_win_pcts)
                            
                            # Filter to games vs winning teams (≥ 0.500)
                            vs_winning_games = full_games[full_games['OppWinPct'] >= 0.500]
                            if len(vs_winning_games) > 0:
                                wins_vs_winning = len(vs_winning_games[vs_winning_games['WL'] == 'W'])
                                losses_vs_winning = len(vs_winning_games[vs_winning_games['WL'] == 'L'])
                                vs_winning = f"{wins_vs_winning}-{losses_vs_winning}"
                            
                            # Filter to games vs losing teams (< 0.500)
                            vs_losing_games = full_games[full_games['OppWinPct'] < 0.500]
                            if len(vs_losing_games) > 0:
                                wins_vs_losing = len(vs_losing_games[vs_losing_games['WL'] == 'W'])
                                losses_vs_losing = len(vs_losing_games[vs_losing_games['WL'] == 'L'])
                                vs_losing = f"{wins_vs_losing}-{losses_vs_losing}"
                    except Exception:
                        vs_winning = "N/A"
                        vs_losing = "N/A"
                    
                    logo_url = get_team_logo_url(team)
                    
                    # Show team logo and card side by side
                    logo_col, card_col = st.columns([0.25, 0.75])
                    
                    with logo_col:
                        if logo_url:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                                <img src="{logo_url}" style="width: 120px; height: 120px; filter: drop-shadow(0px 4px 8px rgba(0,0,0,0.5));">
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with card_col:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
                            border-radius: 12px;
                            padding: 16px;
                            border: 2px solid #FF6B35;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <div>
                                    <div style="font-weight: 700; font-size: 1.3rem; color: #FF6B35;">{team} - {team_city} {team_name}</div>
                                    <div style="color: #9CA3AF; font-size: 0.9rem;">Record: <strong style="color: #FAFAFA;">{record}</strong> | {conf}{'ern' if conf != 'N/A' else ''} Conference #{rank}</div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">ORTG</div>
                                    <div style="color: #10B981; font-weight: 600;">{off_rtg} <span style="color: #6B7280; font-size: 0.8rem;">(#{off_rank})</span></div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">DRTG</div>
                                    <div style="color: #3B82F6; font-weight: 600;">{def_rtg} <span style="color: #6B7280; font-size: 0.8rem;">(#{def_rank})</span></div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">DIVISION</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{division}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">DIV SEED</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">#{div_rank}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">HOME</div>
                                    <div style="color: {get_record_color(home)}; font-weight: 600;">{home}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">ROAD</div>
                                    <div style="color: {get_record_color(road)}; font-weight: 600;">{road}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">L10</div>
                                    <div style="color: {get_record_color(l10)}; font-weight: 600;">{l10}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">STREAK</div>
                                    <div style="color: {get_streak_color(streak)}; font-weight: 600;">{streak}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">vs ≥.500</div>
                                    <div style="color: {get_record_color(vs_winning)}; font-weight: 600;">{vs_winning}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">vs <.500</div>
                                    <div style="color: {get_record_color(vs_losing)}; font-weight: 600;">{vs_losing}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">COACH</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{coach_team_map.get(team, {}).get('name', 'N/A')}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Expander for recent games
                    with st.expander(f"Recent Games - {team}"):
                        team_games = get_team_game_log(team, season, num_games=5)
                        if team_games is not None and len(team_games) > 0:
                            # Create final score column (team PTS vs opponent PTS)
                            games_display = team_games.copy()
                            
                            # Calculate full score using PTS and PLUS_MINUS
                            if 'PTS' in games_display.columns:
                                if 'PLUS_MINUS' in games_display.columns:
                                    # Opponent points = our points - plus/minus
                                    games_display['OPP_PTS'] = games_display['PTS'] - games_display['PLUS_MINUS']
                                    games_display['SCORE'] = games_display.apply(
                                        lambda row: f"{int(row['PTS'])} - {int(row['OPP_PTS'])}", axis=1
                                    )
                                else:
                                    games_display['SCORE'] = games_display['PTS'].astype(int).astype(str)
                                
                            # Select only the columns we want
                            # Select columns - matching the API source which uses 'WL'
                            final_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'SCORE']
                            
                            # Filter to only columns that exist
                            available_cols = [c for c in final_cols if c in games_display.columns]
                            games_display = games_display[available_cols]
                            
                            # Rename columns for display to match user request
                            # We use a mapping dict to be safe regardless of what was found
                            rename_map = {
                                'GAME_DATE': 'Date',
                                'MATCHUP': 'Matchup',
                                'WL': 'W/L',
                                'SCORE': 'Score'
                            }
                            games_display = games_display.rename(columns=rename_map)
                            
                            # Style W/L column with colors
                            def color_wl(val):
                                if val == 'W':
                                    return 'color: #10B981; font-weight: bold'
                                elif val == 'L':
                                    return 'color: #EF4444; font-weight: bold'
                                return ''
                            
                            # Apply styling if W/L column exists
                            if 'W/L' in games_display.columns:
                                styled_df = games_display.style.applymap(color_wl, subset=['W/L'])
                            else:
                                styled_df = games_display
                            
                            # Display with fixed columns (no resizing)
                            st.dataframe(
                                styled_df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "Date": st.column_config.TextColumn("Date", width="medium"),
                                    "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                                    "W/L": st.column_config.TextColumn("W/L", width="small"),
                                    "Score": st.column_config.TextColumn("Score", width="small"),
                                }
                            )
                        else:
                            st.info("No recent games available.")
                    
                    # Expander for upcoming games
                    with st.expander(f"Upcoming Games - {team}"):
                        upcoming = get_team_upcoming_games(team, nba_schedule, standings_df, num_games=5)
                        if upcoming:
                            games_display = []
                            for game in upcoming:
                                home_away = "vs" if game['is_home'] else "@"
                                opp_rank = game['opponent_rank']
                                opp_record = game.get('opponent_record', '')
                                if opp_rank:
                                    opp_display = f"#{opp_rank} {game['opponent_name']}"
                                else:
                                    opp_display = game['opponent_name']
                                # Add opponent record if available
                                if opp_record:
                                    opp_display += f" ({opp_record})"
                                games_display.append({
                                    'Date': game['date'],
                                    'Game': f"{home_away} {opp_display}"
                                })
                            games_df = pd.DataFrame(games_display)
                            st.dataframe(games_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No upcoming games found.")
                    
                    # Team Roster expander
                    with st.expander(f"Team Roster - {team}"):
                        roster = get_team_roster(team)
                        if roster is not None and len(roster) > 0:
                            st.caption("Click a player name to view their stats")
                            # Display roster with clickable player names
                            for idx, player_row in roster.iterrows():
                                player_name = player_row['Player']
                                pos = player_row['Pos']
                                num = player_row['#']
                                
                                col_num, col_name, col_pos, col_stats, col_analyze = st.columns([0.5, 2, 0.5, 1, 1])
                                with col_num:
                                    st.write(f"#{num}")
                                with col_name:
                                    st.write(f"**{player_name}**")
                                with col_pos:
                                    st.write(pos)
                                with col_stats:
                                    if st.button("Stats", key=f"roster_stats_{team}_{idx}_{player_name[:10]}", use_container_width=True):
                                        # Navigate to Player Stats page with this player
                                        st.session_state['player_stats_search'] = player_name
                                        st.session_state['pending_nav_target'] = "Player Stats"
                                        st.rerun()
                                with col_analyze:
                                    if st.button("Analyze", key=f"roster_analyze_{team}_{idx}_{player_name[:10]}", use_container_width=True):
                                        # Navigate to Predictions page with this player
                                        st.session_state['auto_load_player'] = player_name
                                        st.session_state['pending_nav_target'] = "Predictions"
                                        st.rerun()
                        else:
                            st.info("Could not load roster.")
                    
                    # Reorder and Remove buttons
                    re_col1, re_col2, rem_col = st.columns([1, 1, 3])
                    with re_col1:
                        if st.button("▲", key=f"fav_team_up_{team}", use_container_width=True, help=f"Move {team} Up"):
                            auth.reorder_favorite_team(team, "up")
                            st.rerun()
                    with re_col2:
                        if st.button("▼", key=f"fav_team_down_{team}", use_container_width=True, help=f"Move {team} Down"):
                            auth.reorder_favorite_team(team, "down")
                            st.rerun()
                    with rem_col:
                        if st.button("❌ Remove", key=f"team_remove_{team}", use_container_width=True):
                            auth.remove_favorite_team(team)
                            st.toast(f"Removed {team} from favorite teams")
                            st.rerun()
                    
                    st.markdown("---")
            else:
                render_empty_state("No favorite teams yet! Add teams from the Live Predictions page.", "")

# ==================== COMPARE PLAYERS PAGE ====================
elif page == "Compare Players":
    st.title("Compare Players")
    st.markdown("Compare two players head-to-head across all stats")
    
    season = "2025-26"
    
    st.markdown("---")
    
    # Player selection with autocomplete
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Player 1")
        player1_search = st.text_input("Search Player 1", key="compare_player1_search", placeholder="e.g., LeBron James")
        selected_player1 = None
        if player1_search:
            matching1 = search_players(player1_search, season)
            if matching1:
                selected_player1 = st.selectbox("Select Player 1:", matching1, key="compare_select1")
    
    with col2:
        st.markdown("### Player 2")
        player2_search = st.text_input("Search Player 2", key="compare_player2_search", placeholder="e.g., Stephen Curry")
        selected_player2 = None
        if player2_search:
            matching2 = search_players(player2_search, season)
            if matching2:
                selected_player2 = st.selectbox("Select Player 2:", matching2, key="compare_select2")
    
    # Compare button
    trigger_compare = st.session_state.get('trigger_compare', False)
    if selected_player1 and selected_player2:
        if st.button("🔍 Compare Players", type="primary", use_container_width=True) or trigger_compare:
            if trigger_compare:
                if 'trigger_compare' in st.session_state:
                    del st.session_state['trigger_compare']
            with st.spinner("Loading player stats..."):
                # Get player data for both
                player1_df, player1_team = get_player_game_log(selected_player1, season)
                player2_df, player2_team = get_player_game_log(selected_player2, season)
                
                # Load schedule for next matchup display
                nba_schedule = get_nba_schedule()
                
                if player1_df is None or len(player1_df) == 0:
                    st.error(f"No data found for {selected_player1}")
                elif player2_df is None or len(player2_df) == 0:
                    st.error(f"No data found for {selected_player2}")
                else:
                    player1_name = selected_player1
                    player2_name = selected_player2
                    
                    # Get bios
                    bio1 = fetch_player_bio(player1_name)
                    bio2 = fetch_player_bio(player2_name)
                    
                    if bio1 and bio1.get('team_abbrev'):
                        player1_team = bio1['team_abbrev']
                    if bio2 and bio2.get('team_abbrev'):
                        player2_team = bio2['team_abbrev']
                    
                    st.markdown("---")
                    
                    # Player profile cards (MVP Ladder style)
                    col1, col2 = st.columns(2)
                    
                    # Get standings for team records
                    standings_df = get_league_standings(season)
                    
                    with col1:
                        # Player 1 card
                        p1_photo = get_player_photo_url(player1_name)
                        p1_logo = get_team_logo_url(player1_team)
                        
                        # Calculate stats
                        ppg1 = player1_df['Points'].mean() if 'Points' in player1_df.columns else 0
                        rpg1 = player1_df['Rebounds'].mean() if 'Rebounds' in player1_df.columns else 0
                        apg1 = player1_df['Assists'].mean() if 'Assists' in player1_df.columns else 0
                        stats1 = f"{ppg1:.1f} PPG, {rpg1:.1f} RPG, {apg1:.1f} APG"
                        
                        # Shooting splits
                        total_fgm1 = player1_df['FGM'].sum() if 'FGM' in player1_df.columns else 0
                        total_fga1 = player1_df['FGA'].sum() if 'FGA' in player1_df.columns else 0
                        fg_pct1 = (total_fgm1 / total_fga1 * 100) if total_fga1 > 0 else 0
                        total_3pm1 = player1_df['3PM'].sum() if '3PM' in player1_df.columns else 0
                        total_3pa1 = player1_df['3PA'].sum() if '3PA' in player1_df.columns else 0
                        three_pct1 = (total_3pm1 / total_3pa1 * 100) if total_3pa1 > 0 else 0
                        total_ftm1 = player1_df['FTM'].sum() if 'FTM' in player1_df.columns else 0
                        total_fta1 = player1_df['FTA'].sum() if 'FTA' in player1_df.columns else 0
                        ft_pct1 = (total_ftm1 / total_fta1 * 100) if total_fta1 > 0 else 0
                        shooting1 = f"{format_pct(fg_pct1)} FG% • {format_pct(three_pct1)} 3P% • {format_pct(ft_pct1)} FT%"
                        
                        # Team record, rank, and full name
                        team_record1 = "N/A"
                        team_rank1 = "N/A"
                        team_conf1 = "N/A"
                        team_full_name1 = player1_team
                        if not standings_df.empty:
                            for _, row in standings_df.iterrows():
                                if get_team_abbrev(row['TeamCity']) == player1_team:
                                    team_record1 = row['Record']
                                    team_rank1 = row['PlayoffRank']
                                    team_conf1 = row['Conference']
                                    team_full_name1 = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                                    break
                        
                        # Photo and logo (Centered standardized layout) - Slightly shifted right
                        spacer1, photo_col1, logo_col1, spacer2 = st.columns([1.35, 1, 1, 0.65])
                        with photo_col1:
                            if p1_photo:
                                st.image(p1_photo, width=120)
                        with logo_col1:
                            if p1_logo:
                                st.image(p1_logo, width=100)
                        
                        p1_pos = ""
                        if bio1 and bio1.get('position'):
                            abbrev_p1 = abbreviate_position(bio1['position'], player1_name)
                            p1_pos = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 1.1rem;'>({abbrev_p1})</span>"
                        
                        # Bio for player 1 (Muted labels, bold values)
                        p1_height = bio1.get('height', '-') if bio1 else '-'
                        p1_weight = bio1.get('weight', '-') if bio1 else '-'
                        p1_age = bio1.get('age', '-') if bio1 else '-'
                        p1_draft_year = bio1.get('draft_year', '-') if bio1 else '-'
                        p1_draft_round = bio1.get('draft_round', '') if bio1 else ''
                        p1_draft_num = bio1.get('draft_number', '') if bio1 else ''
                        p1_draft_display = p1_draft_year
                        if p1_draft_round and p1_draft_num and p1_draft_year != 'Undrafted':
                            p1_draft_display = f"{p1_draft_year} R{p1_draft_round}, #{p1_draft_num}"

                        # Calculate individual record using correct column name
                        p1_wins = (player1_df['W/L'] == 'W').sum() if 'W/L' in player1_df.columns else 0
                        p1_losses = (player1_df['W/L'] == 'L').sum() if 'W/L' in player1_df.columns else 0
                        p1_ind_record = f"{p1_wins}-{p1_losses}"

                        st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
<div style="color: #FAFAFA; font-weight: bold; font-size: 1.1rem; margin-bottom: 0px;">{player1_name}{p1_pos}</div>
<div style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 5px;">{team_full_name1} <span style="color: #9CA3AF; font-size: 0.85rem;">({team_record1} | #{team_rank1} in {team_conf1})</span></div>
<div style="color: #9CA3AF; font-size: 0.8rem; margin-bottom: 10px;">
<span style="color: #9CA3AF;">HT:</span> <span style="color: #FAFAFA; font-weight: bold;">{p1_height}</span> • 
<span style="color: #9CA3AF;">WT:</span> <span style="color: #FAFAFA; font-weight: bold;">{p1_weight} lbs</span> • 
<span style="color: #9CA3AF;">Age:</span> <span style="color: #FAFAFA; font-weight: bold;">{p1_age}</span> • 
<span style="color: #9CA3AF;">Draft:</span> <span style="color: #FAFAFA; font-weight: bold;">{p1_draft_display}</span>
</div>
<div style="display: flex; justify-content: center; gap: 6px; font-size: 0.72rem;">
<div style="background: #374151; padding: 2px 6px; border-radius: 4px;">
<span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{len(player1_df)}</span>
</div>
<div style="background: #374151; padding: 2px 6px; border-radius: 4px;">
<span style="color: #9CA3AF;">IND:</span> <span style="color: #FAFAFA; font-weight: bold;">{p1_ind_record}</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
                    
                    with col2:
                        # Player 2 card
                        p2_photo = get_player_photo_url(player2_name)
                        p2_logo = get_team_logo_url(player2_team)
                        
                        # Calculate stats
                        ppg2 = player2_df['Points'].mean() if 'Points' in player2_df.columns else 0
                        rpg2 = player2_df['Rebounds'].mean() if 'Rebounds' in player2_df.columns else 0
                        apg2 = player2_df['Assists'].mean() if 'Assists' in player2_df.columns else 0
                        stats2 = f"{ppg2:.1f} PPG, {rpg2:.1f} RPG, {apg2:.1f} APG"
                        
                        # Shooting splits
                        total_fgm2 = player2_df['FGM'].sum() if 'FGM' in player2_df.columns else 0
                        total_fga2 = player2_df['FGA'].sum() if 'FGA' in player2_df.columns else 0
                        fg_pct2 = (total_fgm2 / total_fga2 * 100) if total_fga2 > 0 else 0
                        total_3pm2 = player2_df['3PM'].sum() if '3PM' in player2_df.columns else 0
                        total_3pa2 = player2_df['3PA'].sum() if '3PA' in player2_df.columns else 0
                        three_pct2 = (total_3pm2 / total_3pa2 * 100) if total_3pa2 > 0 else 0
                        total_ftm2 = player2_df['FTM'].sum() if 'FTM' in player2_df.columns else 0
                        total_fta2 = player2_df['FTA'].sum() if 'FTA' in player2_df.columns else 0
                        ft_pct2 = (total_ftm2 / total_fta2 * 100) if total_fta2 > 0 else 0
                        shooting2 = f"{format_pct(fg_pct2)} FG% • {format_pct(three_pct2)} 3P% • {format_pct(ft_pct2)} FT%"
                        
                        # Team record, rank, and full name
                        team_record2 = "N/A"
                        team_rank2 = "N/A"
                        team_conf2 = "N/A"
                        team_full_name2 = player2_team
                        if not standings_df.empty:
                            for _, row in standings_df.iterrows():
                                if get_team_abbrev(row['TeamCity']) == player2_team:
                                    team_record2 = row['Record']
                                    team_rank2 = row['PlayoffRank']
                                    team_conf2 = row['Conference']
                                    team_full_name2 = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                                    break
                        
                        # Photo and logo (Centered standardized layout) - Slightly shifted right
                        spacer1, photo_col2, logo_col2, spacer2 = st.columns([1.35, 1, 1, 0.65])
                        with photo_col2:
                            if p2_photo:
                                st.image(p2_photo, width=120)
                        with logo_col2:
                            if p2_logo:
                                st.image(p2_logo, width=100)
                        
                        p2_pos = ""
                        if bio2 and bio2.get('position'):
                            abbrev_p2 = abbreviate_position(bio2['position'], player2_name)
                            p2_pos = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 1.1rem;'>({abbrev_p2})</span>"
                            
                        # Bio for player 2 (Muted labels, bold values)
                        p2_height = bio2.get('height', '-') if bio2 else '-'
                        p2_weight = bio2.get('weight', '-') if bio2 else '-'
                        p2_age = bio2.get('age', '-') if bio2 else '-'
                        p2_draft_year = bio2.get('draft_year', '-') if bio2 else '-'
                        p2_draft_round = bio2.get('draft_round', '') if bio2 else ''
                        p2_draft_num = bio2.get('draft_number', '') if bio2 else ''
                        p2_draft_display = p2_draft_year
                        if p2_draft_round and p2_draft_num and p2_draft_year != 'Undrafted':
                            p2_draft_display = f"{p2_draft_year} R{p2_draft_round}, #{p2_draft_num}"

                        # Calculate individual record using correct column name
                        p2_wins = (player2_df['W/L'] == 'W').sum() if 'W/L' in player2_df.columns else 0
                        p2_losses = (player2_df['W/L'] == 'L').sum() if 'W/L' in player2_df.columns else 0
                        p2_ind_record = f"{p2_wins}-{p2_losses}"

                        st.markdown(f"""<div style="text-align: center; margin-top: 10px;">
<div style="color: #FAFAFA; font-weight: bold; font-size: 1.1rem; margin-bottom: 0px;">{player2_name}{p2_pos}</div>
<div style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 5px;">{team_full_name2} <span style="color: #9CA3AF; font-size: 0.85rem;">({team_record2} | #{team_rank2} in {team_conf2})</span></div>
<div style="color: #9CA3AF; font-size: 0.8rem; margin-bottom: 10px;">
<span style="color: #9CA3AF;">HT:</span> <span style="color: #FAFAFA; font-weight: bold;">{p2_height}</span> • 
<span style="color: #9CA3AF;">WT:</span> <span style="color: #FAFAFA; font-weight: bold;">{p2_weight} lbs</span> • 
<span style="color: #9CA3AF;">Age:</span> <span style="color: #FAFAFA; font-weight: bold;">{p2_age}</span> • 
<span style="color: #9CA3AF;">Draft:</span> <span style="color: #FAFAFA; font-weight: bold;">{p2_draft_display}</span>
</div>
<div style="display: flex; justify-content: center; gap: 6px; font-size: 0.72rem;">
<div style="background: #374151; padding: 2px 6px; border-radius: 4px;">
<span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{len(player2_df)}</span>
</div>
<div style="background: #374151; padding: 2px 6px; border-radius: 4px;">
<span style="color: #9CA3AF;">IND:</span> <span style="color: #FAFAFA; font-weight: bold;">{p2_ind_record}</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center;'>Season Averages Comparison</h3>", unsafe_allow_html=True)
                    
                    # Calculate averages for both players
                    def calc_player_averages(df):
                        stats = {}
                        stats['PPG'] = round(df['Points'].mean(), 1)
                        stats['RPG'] = round(df['Rebounds'].mean(), 1)
                        stats['APG'] = round(df['Assists'].mean(), 1)
                        stats['SPG'] = round(df['Steals'].mean(), 1)
                        stats['BPG'] = round(df['Blocks'].mean(), 1)
                        stats['TPG'] = round(df['Turnovers'].mean(), 1)
                        
                        # Shooting
                        if 'FGM' in df.columns and 'FGA' in df.columns:
                            total_fgm = df['FGM'].sum()
                            total_fga = df['FGA'].sum()
                            stats['FG%'] = round((total_fgm / total_fga * 100), 1) if total_fga > 0 else 0
                        else:
                            stats['FG%'] = 0
                        
                        if '3PM' in df.columns and '3PA' in df.columns:
                            total_3pm = df['3PM'].sum()
                            total_3pa = df['3PA'].sum()
                            stats['3P%'] = round((total_3pm / total_3pa * 100), 1) if total_3pa > 0 else 0
                        else:
                            stats['3P%'] = 0
                        
                        if 'FTM' in df.columns and 'FTA' in df.columns:
                            total_ftm = df['FTM'].sum()
                            total_fta = df['FTA'].sum()
                            stats['FT%'] = round((total_ftm / total_fta * 100), 1) if total_fta > 0 else 0
                        else:
                            stats['FT%'] = 0
                        
                        # TS%
                        if 'FGA' in df.columns and 'FTA' in df.columns:
                            total_pts = df['Points'].sum()
                            total_fga = df['FGA'].sum()
                            total_fta = df['FTA'].sum()
                            stats['TS%'] = round((total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100), 1) if (total_fga + 0.44 * total_fta) > 0 else 0
                        else:
                            stats['TS%'] = 0
                            
                        # IND REC
                        wins = (df['W/L'] == 'W').sum() if 'W/L' in df.columns else 0
                        losses = (df['W/L'] == 'L').sum() if 'W/L' in df.columns else 0
                        stats['IND REC'] = f"{wins}-{losses}"
                        # Calculate win pct for comparison logic
                        stats['Win%'] = round((wins / (wins + losses) * 100), 1) if (wins + losses) > 0 else 0
                        
                        # Games and minutes
                        stats['Games'] = len(df)
                        if 'MIN' in df.columns:
                            def parse_min(m):
                                if pd.isna(m): return 0
                                if ':' in str(m): return int(str(m).split(':')[0])
                                try: return float(m)
                                except: return 0
                            stats['MPG'] = round(df['MIN'].apply(parse_min).mean(), 1)
                        else:
                            stats['MPG'] = 0
                        
                        return stats
                    
                    p1_stats = calc_player_averages(player1_df)
                    p2_stats = calc_player_averages(player2_df)
                    
                    # Stats where higher is better
                    higher_is_better = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'FG%', '3P%', 'FT%', 'TS%', 'Games', 'MPG']
                    # Stats where lower is better
                    lower_is_better = ['TPG']
                    
                    # Create comparison table
                    comparison_data = []
                    for stat in ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'FT%', 'TS%', 'MPG', 'Games', 'IND REC']:
                        v1 = p1_stats.get(stat, 0)
                        v2 = p2_stats.get(stat, 0)
                        
                        # Determine winner
                        if stat == 'IND REC':
                            # Compare win percentages for ID REC
                            p1_better = p1_stats.get('Win%', 0) > p2_stats.get('Win%', 0)
                            p2_better = p2_stats.get('Win%', 0) > p1_stats.get('Win%', 0)
                        elif stat in higher_is_better:
                            p1_better = v1 > v2
                            p2_better = v2 > v1
                        else:  # lower is better (like turnovers)
                            p1_better = v1 < v2
                            p2_better = v2 < v1
                        
                        comparison_data.append({
                            'Stat': stat,
                            'p1_value': v1,
                            'p2_value': v2,
                            'p1_better': p1_better,
                            'p2_better': p2_better
                        })
                    
                    # Display comparison with highlighting
                    for item in comparison_data:
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        # Determine colors
                        p1_color = "#10B981" if item['p1_better'] else "#FAFAFA"  # Green if better
                        p2_color = "#10B981" if item['p2_better'] else "#FAFAFA"
                        
                        with col1:
                            value = format_pct(item['p1_value']) if '%' in item['Stat'] else item['p1_value']
                            st.markdown(f"""
                            <div style="text-align: right; padding: 8px; font-size: 1.2rem; color: {p1_color}; font-weight: {'bold' if item['p1_better'] else 'normal'};">
                                {value}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 8px; font-size: 1rem; color: #9CA3AF;">
                                {item['Stat']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            value = format_pct(item['p2_value']) if '%' in item['Stat'] else item['p2_value']
                            st.markdown(f"""
                            <div style="text-align: left; padding: 8px; font-size: 1.2rem; color: {p2_color}; font-weight: {'bold' if item['p2_better'] else 'normal'};">
                                {value}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    
                    st.caption("Note: 'IND REC' comparison is based on individual player win percentage in games played.")
                    
                    # ==================== HEAD-TO-HEAD MATCHUPS ====================
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center;'>Head-to-Head Matchups</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>Games where these players faced each other</p>", unsafe_allow_html=True)
                    
                    # Find the next matchup between these two teams
                    from datetime import datetime
                    today = get_local_now().date()
                    next_matchup = None
                    
                    if nba_schedule:
                        for game in nba_schedule:
                            try:
                                game_date_str = game['game_date']
                                game_date = datetime.strptime(game_date_str.split(' ')[0], '%m/%d/%Y').date()
                                
                                if game_date < today:
                                    continue
                                if game['game_status'] == 3:  # Already finished
                                    continue
                                
                                # Check if these two teams are playing
                                home = game['home_team']
                                away = game['away_team']
                                if (home == player1_team and away == player2_team) or (home == player2_team and away == player1_team):
                                    is_p1_home = home == player1_team
                                    location = f"@ {player2_team}" if not is_p1_home else f"vs {player2_team}"
                                    next_matchup = {
                                        'date': game_date.strftime('%b %d, %Y'),
                                        'location': location,
                                        'home': home,
                                        'away': away
                                    }
                                    break
                            except:
                                continue
                    
                    if next_matchup:
                        st.markdown(f"<p style='text-align: center; color: #F59E0B; font-size: 1.1rem;'>📅 <strong>Next Matchup:</strong> {next_matchup['date']} — {player1_team} {next_matchup['location']}</p>", unsafe_allow_html=True)
                    
                    # Find games where player 1 played vs player 2's team
                    p1_vs_p2_team = player1_df[player1_df['Opponent'] == player2_team].copy()
                    # Find games where player 2 played vs player 1's team
                    p2_vs_p1_team = player2_df[player2_df['Opponent'] == player1_team].copy()
                    
                    # Filter to only TRUE head-to-head games (both players played on the same date)
                    p1_dates = set(p1_vs_p2_team['GAME_DATE'].tolist()) if len(p1_vs_p2_team) > 0 else set()
                    p2_dates = set(p2_vs_p1_team['GAME_DATE'].tolist()) if len(p2_vs_p1_team) > 0 else set()
                    common_dates = p1_dates & p2_dates  # Games where both played
                    
                    # Filter to only common dates for display
                    p1_h2h = p1_vs_p2_team[p1_vs_p2_team['GAME_DATE'].isin(common_dates)].copy() if len(p1_vs_p2_team) > 0 else p1_vs_p2_team
                    p2_h2h = p2_vs_p1_team[p2_vs_p1_team['GAME_DATE'].isin(common_dates)].copy() if len(p2_vs_p1_team) > 0 else p2_vs_p1_team
                    
                    if len(p1_h2h) == 0 and len(p2_h2h) == 0:
                        st.info(f"No head-to-head matchups found between {player1_name} and {player2_name} this season.")
                    else:
                        # Display both players' game logs vs each other's teams (stacked for more columns)
                        
                        # Helper to calculate Score if not present
                        def add_score_column(df, team_abbrev):
                            if 'Score' not in df.columns:
                                score_lookup = {}
                                team_game_data = get_team_game_log(team_abbrev, season, num_games=82)
                                if team_game_data is not None and len(team_game_data) > 0:
                                    for _, trow in team_game_data.iterrows():
                                        game_id = str(trow.get('GAME_ID', ''))
                                        if 'PTS' in trow and 'PLUS_MINUS' in trow:
                                            team_pts = int(trow['PTS'])
                                            opp_pts = int(trow['PTS'] - trow['PLUS_MINUS'])
                                            score_lookup[game_id] = f"{team_pts} - {opp_pts}"
                                if score_lookup:
                                    df['Score'] = df.apply(
                                        lambda row: score_lookup.get(str(row.get('Game_ID', row.get('GAME_ID', ''))), 'N/A'),
                                        axis=1
                                    )
                                else:
                                    df['Score'] = 'N/A'
                            return df
                        
                        # Player 1 vs Player 2
                        st.markdown(f"#### {player1_name} vs {player2_name}")
                        if len(p1_h2h) > 0:
                            # Add Score if not present
                            p1_h2h = add_score_column(p1_h2h, player1_team)
                            
                            # Select all game log columns (use correct names)
                            display_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PF', 'FG', '3P', 'FT', 'FG%', '3P%', 'FT%', 'TS%']
                            available_cols = [c for c in display_cols if c in p1_h2h.columns]
                            h2h_display = p1_h2h[available_cols].copy()
                            
                            # Rename columns for cleaner display
                            h2h_display = h2h_display.rename(columns={
                                'GAME_DATE': 'Date',
                                'Points': 'PTS',
                                'Rebounds': 'REB',
                                'Assists': 'AST',
                                'Steals': 'STL',
                                'Blocks': 'BLK',
                                'Turnovers': 'TOV'
                            })
                            
                            # Format percentages (multiply by 100 if they're decimals)
                            for pct_col in ['FG%', '3P%', 'FT%']:
                                if pct_col in h2h_display.columns:
                                    h2h_display[pct_col] = h2h_display[pct_col].apply(
                                        lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x <= 1 else (f"{x:.1f}%" if pd.notnull(x) else "N/A")
                                    )
                            if 'TS%' in h2h_display.columns:
                                h2h_display['TS%'] = h2h_display['TS%'].apply(
                                    lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
                                )
                            
                            # Style W/L column with colors
                            def style_wl(df):
                                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                                if 'W/L' in df.columns:
                                    styles['W/L'] = df['W/L'].apply(
                                        lambda x: 'color: #10B981' if x == 'W' else ('color: #EF4444' if x == 'L' else '')
                                    )
                                return styles
                            
                            styled_df = h2h_display.style.apply(style_wl, axis=None)
                            st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            
                            # Show averages (using H2H games)
                            avg_pts = p1_h2h['Points'].mean()
                            avg_reb = p1_h2h['Rebounds'].mean()
                            avg_ast = p1_h2h['Assists'].mean()
                            
                            # Shooting metrics
                            sum_fgm = p1_h2h['FGM'].sum()
                            sum_fga = p1_h2h['FGA'].sum()
                            avg_fg_pct = (sum_fgm / sum_fga * 100) if sum_fga > 0 else 0
                            
                            sum_3pm = p1_h2h['3PM'].sum()
                            sum_3pa = p1_h2h['3PA'].sum()
                            avg_3p_pct = (sum_3pm / sum_3pa * 100) if sum_3pa > 0 else 0
                            
                            sum_ftm = p1_h2h['FTM'].sum()
                            sum_fta = p1_h2h['FTA'].sum()
                            avg_ft_pct = (sum_ftm / sum_fta * 100) if sum_fta > 0 else 0
                            
                            sum_pts = p1_h2h['Points'].sum()
                            avg_ts_pct = (sum_pts / (2 * (sum_fga + 0.44 * sum_fta)) * 100) if (sum_fga + 0.44 * sum_fta) > 0 else 0
                            
                            st.markdown(f"<p style='color: #9CA3AF; font-size: 0.9rem;'>Avg: <strong>{avg_pts:.1f}</strong> PTS, <strong>{avg_reb:.1f}</strong> REB, <strong>{avg_ast:.1f}</strong> AST • <strong>{avg_fg_pct:.1f}%</strong> FG%, <strong>{avg_3p_pct:.1f}%</strong> 3P%, <strong>{avg_ft_pct:.1f}%</strong> FT%, <strong>{avg_ts_pct:.1f}%</strong> TS% ({len(p1_h2h)} games)</p>", unsafe_allow_html=True)
                        else:
                            st.info(f"No H2H games where both players played")
                        
                        st.markdown("")  # Spacer
                        
                        # Player 2 vs Player 1
                        st.markdown(f"#### {player2_name} vs {player1_name}")
                        if len(p2_h2h) > 0:
                            # Add Score if not present
                            p2_h2h = add_score_column(p2_h2h, player2_team)
                            
                            # Select all game log columns (use correct names)
                            display_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PF', 'FG', '3P', 'FT', 'FG%', '3P%', 'FT%', 'TS%']
                            available_cols = [c for c in display_cols if c in p2_h2h.columns]
                            h2h_display = p2_h2h[available_cols].copy()
                            
                            # Rename columns for cleaner display
                            h2h_display = h2h_display.rename(columns={
                                'GAME_DATE': 'Date',
                                'Points': 'PTS',
                                'Rebounds': 'REB',
                                'Assists': 'AST',
                                'Steals': 'STL',
                                'Blocks': 'BLK',
                                'Turnovers': 'TOV'
                            })
                            
                            # Format percentages (multiply by 100 if they're decimals)
                            for pct_col in ['FG%', '3P%', 'FT%']:
                                if pct_col in h2h_display.columns:
                                    h2h_display[pct_col] = h2h_display[pct_col].apply(
                                        lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x <= 1 else (f"{x:.1f}%" if pd.notnull(x) else "N/A")
                                    )
                            if 'TS%' in h2h_display.columns:
                                h2h_display['TS%'] = h2h_display['TS%'].apply(
                                    lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
                                )
                            
                            # Style W/L column with colors
                            def style_wl(df):
                                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                                if 'W/L' in df.columns:
                                    styles['W/L'] = df['W/L'].apply(
                                        lambda x: 'color: #10B981' if x == 'W' else ('color: #EF4444' if x == 'L' else '')
                                    )
                                return styles
                            
                            styled_df = h2h_display.style.apply(style_wl, axis=None)
                            st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            
                            # Show averages (using H2H games)
                            avg_pts = p2_h2h['Points'].mean()
                            avg_reb = p2_h2h['Rebounds'].mean()
                            avg_ast = p2_h2h['Assists'].mean()
                            
                            # Shooting metrics
                            sum_fgm = p2_h2h['FGM'].sum()
                            sum_fga = p2_h2h['FGA'].sum()
                            avg_fg_pct = (sum_fgm / sum_fga * 100) if sum_fga > 0 else 0
                            
                            sum_3pm = p2_h2h['3PM'].sum()
                            sum_3pa = p2_h2h['3PA'].sum()
                            avg_3p_pct = (sum_3pm / sum_3pa * 100) if sum_3pa > 0 else 0
                            
                            sum_ftm = p2_h2h['FTM'].sum()
                            sum_fta = p2_h2h['FTA'].sum()
                            avg_ft_pct = (sum_ftm / sum_fta * 100) if sum_fta > 0 else 0
                            
                            sum_pts = p2_h2h['Points'].sum()
                            avg_ts_pct = (sum_pts / (2 * (sum_fga + 0.44 * sum_fta)) * 100) if (sum_fga + 0.44 * sum_fta) > 0 else 0
                            
                            st.markdown(f"<p style='color: #9CA3AF; font-size: 0.9rem;'>Avg: <strong>{avg_pts:.1f}</strong> PTS, <strong>{avg_reb:.1f}</strong> REB, <strong>{avg_ast:.1f}</strong> AST • <strong>{avg_fg_pct:.1f}%</strong> FG%, <strong>{avg_3p_pct:.1f}%</strong> 3P%, <strong>{avg_ft_pct:.1f}%</strong> FT%, <strong>{avg_ts_pct:.1f}%</strong> TS% ({len(p2_h2h)} games)</p>", unsafe_allow_html=True)
                        else:
                            st.info(f"No H2H games where both players played")
                    
    else:
        st.info("👆 Enter two player names above to compare their stats")

# ==================== AROUND THE NBA PAGE ====================
elif page == "Around the NBA":
    from datetime import datetime
    import pytz
    
    st.title("🏀 Around the NBA")
    
    # Define timezone for News Wire dates
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    st.markdown("---")
    
    # Fetch data
    with st.spinner("Fetching latest NBA updates..."):
        mvp_ladder, mvp_date = get_mvp_ladder()
        nba_schedule = get_nba_schedule()
        standings_df = get_league_standings(season)
        nba_news = get_nba_news()

    # News Ticker
    if nba_news:
        ticker_items = [f" {item['title']} " for item in nba_news]
        # Join with a nice separator emoji
        ticker_text = " 🔥 ".join(ticker_items)
        # Duplicate for smooth infinite scroll
        combined_text = ticker_text + " 🔥 " + ticker_text
        
        st.markdown(f"""<div style="background: #1e1b4b; color: #e0e7ff; padding: 12px 0; border-radius: 8px; 
margin-bottom: 30px; border: 1px solid #312e81; overflow: hidden; 
white-space: nowrap; position: relative; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
<div style="display: inline-block; padding-left: 20px; animation: ticker 120s linear infinite; font-weight: 500; font-size: 1.1rem;">
{combined_text}
</div>
</div>
<style>
@keyframes ticker {{
0% {{ transform: translateX(0); }}
100% {{ transform: translateX(-50%); }}
}}
</style>""", unsafe_allow_html=True)

    # ===== SECTION 1: 📅 DAILY GAME SLATE =====
    st.markdown("## Today's Games")
    
    # Get today's games using the same function as Predictions page
    # Pass the user_tz defined in sidebar
    todays_games = get_todays_games(nba_schedule, standings_df, tz=user_tz)
    
    # Fetch live/final scores
    scoreboard = get_todays_scoreboard()
    
    if todays_games:
        from datetime import datetime
        tz_label = st.session_state.get('user_timezone', 'US/Pacific').split('/')[-1].replace('_', ' ')
        st.caption(f"**{get_local_now().strftime('%A, %B %d, %Y')}** • {len(todays_games)} game(s) • _All times in {tz_label}_")
        
        # Display games in a grid
        cols_per_row = 3
        for i in range(0, len(todays_games), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(todays_games):
                    game = todays_games[i + j]
                    with col:
                        # Get team logos
                        away_logo = get_team_logo_url(game['away_team'])
                        home_logo = get_team_logo_url(game['home_team'])
                        
                        # Format seeds and conference
                        away_seed = f"#{game['away_rank']}" if game['away_rank'] else ""
                        home_seed = f"#{game['home_rank']}" if game['home_rank'] else ""
                        
                        def fmt_cs(conf, streak):
                            if not conf: return ""
                            c = conf.replace("ern", "")
                            s = f"({streak.replace(' ', '')})" if streak else ""
                            return f"{c} {s}"
                            
                        away_conf = fmt_cs(game.get('away_conference'), game.get('away_streak'))
                        home_conf = fmt_cs(game.get('home_conference'), game.get('home_streak'))
                        
                        game_time = game.get('game_time', 'TBD')
                        if not game_time:
                            game_time = 'TBD'
                        
                        # Get records
                        away_record = game.get('away_record', '')
                        home_record = game.get('home_record', '')
                        
                        # Get live score from scoreboard
                        game_id = game.get('game_id')
                        live_data = scoreboard.get(game_id, {})
                        game_status = live_data.get('game_status', 1)
                        status_text = live_data.get('game_status_text', '')
                        home_score = live_data.get('home_score', 0)
                        away_score = live_data.get('away_score', 0)
                        
                        # Determine display based on game status
                        if game_status == 3:  # Final
                            status_display = f"<span style='color: #10B981;'>FINAL</span>"
                            away_score_display = f"<span style='font-size: 1.3rem; font-weight: bold; color: {'#10B981' if away_score > home_score else '#FAFAFA'};'>{away_score}</span>"
                            home_score_display = f"<span style='font-size: 1.3rem; font-weight: bold; color: {'#10B981' if home_score > away_score else '#FAFAFA'};'>{home_score}</span>"
                        elif game_status == 2:  # In Progress
                            status_display = f"<span style='color: #F59E0B;'>{status_text}</span>"
                            away_score_display = f"<span style='font-size: 1.3rem; font-weight: bold;'>{away_score}</span>"
                            home_score_display = f"<span style='font-size: 1.3rem; font-weight: bold;'>{home_score}</span>"
                        else:  # Scheduled - show 'Home' label for home team
                            status_display = f"<span style='color: #FF6B35;'>{game_time}</span>"
                            away_score_display = ""
                            home_score_display = "<span style='color: #9CA3AF; font-size: 0.75rem;'>Home</span>"
                        
                        # Create a more visual card with logos and scores
                        channel_display = game.get('channel', '')
                        box_score_url = f"https://www.nba.com/game/{game_id}" if game_id else "https://www.nba.com/games"
                        box_score_html = f"<div style='position: absolute; top: 12px; left: 12px; font-size: 0.72rem; font-weight: 700;'><a href='{box_score_url}' target='_blank' style='color: #9CA3AF; text-decoration: none; border: 1px solid #374151; padding: 2px 6px; border-radius: 4px;'>BOX SCORE</a></div>"
                        channel_html = f"<div style='position: absolute; top: 12px; right: 12px; color: #9CA3AF; font-size: 0.72rem; font-weight: 600;'>{channel_display}</div>" if channel_display else ""
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1F2937 0%, #111827 100%); 
                                    border-radius: 10px; padding: 15px; text-align: center; 
                                    border: 1px solid #374151; margin-bottom: 10px; position: relative;">
                            {box_score_html}
                            {channel_html}
                            <div style="font-size: 0.8rem; font-weight: bold; margin-bottom: 8px;">{status_display}</div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <img src="{away_logo}" width="38" height="38" style="filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['away_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{away_record}, {away_seed} {away_conf}</div>
                                    </div>
                                </div>
                                <div>{away_score_display}</div>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 10px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <img src="{home_logo}" width="38" height="38" style="filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['home_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{home_record}, {home_seed} {home_conf}</div>
                                    </div>
                                </div>
                                <div>{home_score_display}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("No games scheduled for today.")
    
    st.markdown("---")
    
    # ===== SECTION 3: 📰 TOP HEADLINES =====
    col_h, col_r = st.columns([4, 1])
    with col_h:
        st.markdown("## 📰 NBA News Wire")
        st.caption("Real-time updates from NBA.com")
    with col_r:
        if st.button("🔄 Refresh News", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if nba_news:
        featured = nba_news[0]
        if featured:
            # Derived date: if it says "ago", it's from current date in ET
            article_time = featured.get('time', '')
            article_date = article_time
            if "ago" in article_time.lower() or not article_time:
                article_date = now_et.strftime("%B %d, %Y")
            
            time_display = f"<span style='color: #818cf8; font-size: 0.8rem; font-weight: 600;'>{article_time}</span>" if article_time else ""
            st.markdown(f"""<div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); 
border-radius: 12px; padding: 30px; margin-bottom: 25px;
border: 1px solid #312e81; position: relative; overflow: hidden;">
<div style="position: absolute; top: 0; right: 0; background: #6366f1; color: white; 
padding: 6px 15px; font-size: 0.75rem; font-weight: bold; border-bottom-left-radius: 12px;">
FEATURED STORY
</div>
<a href="{featured['link']}" target="_blank" style="text-decoration: none;">
<h2 style="margin: 0; color: #fff; font-size: 1.5rem; line-height: 1.3; font-weight: 800;">{featured['title']}</h2>
<div style="margin-top: 15px; display: flex; justify-content: space-between; align-items: center;">
<div style="display: flex; align-items: center; gap: 10px;">
{time_display}
<p style="margin: 0; color: #a5b4fc; font-weight: 500; font-size: 0.9rem;">
Read report →
</p>
</div>
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 500;">{article_date}</div>
</div>
</a>
</div>""", unsafe_allow_html=True)

        # Other Headlines in a cleaner grid or list
        if len(nba_news) > 1:
            cols = st.columns(2)
            for i, item in enumerate(nba_news[1:7]): # Show up to 6 in grid
                with cols[i % 2]:
                    # Derived date for grid items
                    article_time = item.get('time', '')
                    article_date = article_time
                    if "ago" in article_time.lower() or not article_time:
                        article_date = now_et.strftime("%b %d, %Y") # Shorter date for grid
                    
                    time_info = f"<p style='margin: 0; color: #6366f1; font-size: 0.75rem; font-weight: bold;'>{article_time}</p>" if article_time else ""
                    st.markdown(f"""<div style="background: #111827; border-radius: 10px; padding: 18px; 
margin-bottom: 15px; border: 1px solid #1f2937; height: 180px;
display: flex; flex-direction: column; justify-content: space-between;">
<a href="{item['link']}" target="_blank" style="text-decoration: none;">
<h4 style="margin: 0; color: #e5e7eb; font-size: 0.95rem; line-height: 1.4; font-weight: 600;">{item['title']}</h4>
<div style="margin-top: 12px; display: flex; align-items: center; gap: 8px;">
{time_info}
<p style="margin: 0; color: #6366f1; font-weight: 600; font-size: 0.8rem;">Read report →</p>
</div>
</a>
<div style="display: flex; justify-content: flex-end; margin-top: 5px;">
<div style="color: #4b5563; font-size: 0.7rem; font-weight: 600;">{article_date}</div>
</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Latest news currently unavailable. [Visit NBA.com →](https://www.nba.com/news)")
    
    st.markdown("---")
    st.markdown("#### Discover More")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 View Standings", use_container_width=True):
            st.session_state.pending_nav_target = "Standings"
            st.rerun()
    with c2:
        st.link_button("🏥 Injury Report", "https://www.espn.com/nba/injuries", use_container_width=True)
    



# ==================== STANDINGS PAGE ====================
elif page == "Standings":
    from datetime import datetime
    
    st.title("NBA Standings")
    
    # Display current date and refresh button
    col_date, col_refresh = st.columns([3, 1])
    with col_date:
        current_date = get_local_now().strftime("%B %d, %Y")
        st.markdown(f"**As of: {current_date}**")
    with col_refresh:
        if st.button("🔄 Refresh Standings", key="refresh_standings_btn"):
            # Clear cached standings and ratings data
            st.cache_data.clear()
            st.rerun()
    st.markdown("---")
    
    # Get user's favorite teams for highlighting
    user_favorite_teams = []
    if is_authenticated:
        user_favorite_teams = auth.get_favorite_teams() or []
    
    # Fetch standings and team ratings
    with st.spinner("Loading standings..."):
        standings_df = get_league_standings(season)
        team_ratings = get_team_ratings_with_ranks(season)
        nba_schedule = get_nba_schedule()

    
    if standings_df.empty:
        st.error("Could not load standings. Please try again later.")
    else:
        # Create tabs for conferences, divisions, and playoff picture
        tab_west, tab_east, tab_divisions, tab_playoffs = st.tabs(["Western Conference", "Eastern Conference", "Divisions", "Playoff Picture"])
        
        def display_conference_standings(conference_df, favorite_teams, team_ratings, schedule, all_standings):
            """Display standings for a conference with favorite team highlighting and tiebreaker notes."""
            # Sort by playoff rank
            conference_df = conference_df.sort_values('PlayoffRank')
            
            # Detect ties and calculate tiebreakers
            tiebreaker_notes = []
            active_ties = {} # rank -> symbol
            
            df_reset = conference_df.reset_index(drop=True)
            
            # Helper to check H2H record
            def check_h2h_winner(team_a_abbr, team_b_abbr):
                try:
                     # Fetch logs for team A vs team B
                     # Optimization: For now we skip live fetch inside loop or implement strictly
                     # Just return neutral explanation if API heavy.
                     # But user asked for reason.
                     games = get_team_game_log(team_a_abbr, season, num_games=82)
                     if games is not None and not games.empty:
                         wins = 0
                         losses = 0
                         played = False
                         for _, game in games.iterrows():
                             if team_b_abbr in game.get('MATCHUP', ''):
                                 played = True
                                 if game['WL'] == 'W':
                                     wins += 1
                                 else:
                                     losses += 1
                         if played:
                             if wins > losses:
                                 return f"Better Head-to-Head record vs {team_b_abbr} ({wins}-{losses})"
                             elif losses > wins:
                                 return f"Worse Head-to-Head record vs {team_b_abbr} ({wins}-{losses})"
                             else:
                                 return f"Tied Head-to-Head with {team_b_abbr} ({wins}-{losses})"
                except:
                    pass
                return None

            for i in range(len(df_reset) - 1):
                team1 = df_reset.iloc[i]
                team2 = df_reset.iloc[i+1]
                
                if team1['Record'] == team2['Record']:
                    t1_abbr = get_team_abbrev(team1['TeamCity'])
                    t2_abbr = get_team_abbrev(team2['TeamCity'])
                    t1_name = team1['TeamCity']
                    t2_name = team2['TeamCity']
                    
                    rank = int(team1['PlayoffRank'])
                    if rank not in active_ties:
                        active_ties[rank] = "*"
                    
                    reason = "Tiebreaker Rule applied"
                    
                    # 1. H2H check
                    h2h = check_h2h_winner(t1_abbr, t2_abbr)
                    if h2h and "Better" in h2h:
                        reason = h2h
                    else:
                        # 2. Conference Record check
                        try:
                            def parse_pct(s):
                                if '-' in str(s):
                                    w, l = map(int, s.split('-'))
                                    return w/(w+l) if (w+l)>0 else 0
                                return 0
                            c1 = parse_pct(team1.get('ConferenceRecord', '0-0'))
                            c2 = parse_pct(team2.get('ConferenceRecord', '0-0'))
                            
                            if c1 > c2:
                                reason = f"Better Conference Record ({team1['ConferenceRecord']} vs {team2.get('ConferenceRecord','')})"
                            elif c1 < c2:
                                reason = f"Division Leader or other rule" # Should be team2 higher?
                            else:
                                # 3. Division Record
                                d1_slug = get_team_division(t1_abbr)
                                d2_slug = get_team_division(t2_abbr)
                                if d1_slug == d2_slug:
                                    reason = f"Better Division Record ({team1.get('DivisionRecord','')} vs {team2.get('DivisionRecord','')})"
                        except:
                            pass
                    
                    tiebreaker_notes.append(f"*{rank} {t1_name} over {t2_name}: {reason}")

            for idx, row in conference_df.iterrows():
                team_abbrev = get_team_abbrev(row['TeamCity'])
                is_favorite = team_abbrev in favorite_teams
                
                rank = int(row['PlayoffRank'])
                team_name = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                record = row['Record']
                win_pct = f"{row['WinPct']*100:.1f}" if isinstance(row['WinPct'], (int, float)) else row['WinPct']
                l10 = row.get('L10', 'N/A')
                streak = row.get('strCurrentStreak', 'N/A')
                gb = row.get('GB', '-')
                home = row.get('HOME', 'N/A')
                road = row.get('ROAD', 'N/A')
                conf_rec = row.get('ConferenceRecord', 'N/A')
                div_rec = row.get('DivisionRecord', 'N/A')
                logo_url = get_team_logo_url(team_abbrev)
                
                # Calculate GP
                wins_val = row.get('Wins', 0)
                loss_val = row.get('Losses', 0)
                gp = int(wins_val) + int(loss_val) if (wins_val != 'N/A' and loss_val != 'N/A') else 0
                
                # Get team ratings
                team_rtg = team_ratings.get(team_abbrev, {})
                off_rtg = team_rtg.get('off_rtg', 'N/A')
                def_rtg = team_rtg.get('def_rtg', 'N/A')
                off_rank = team_rtg.get('off_rank', 'N/A')
                def_rank = team_rtg.get('def_rank', 'N/A')
                
                
                # Use Streamlit columns for layout with optimized ratios to prevent wrapping
                # Order: Logo, Rank, Team, Record, GB, Win%, Home, Road, Conf, Div, L10, Streak, ORTG, DRTG
                # Order: Identity Col (Rank, Logo, Name), then stats
                col_team, col3, col4, col5, col6, col7, col8, col10, col11, col12, col13 = st.columns([3.0, 0.8, 0.5, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 1.1, 1.1])
                
                with col_team:
                    rank_color = "#FF6B35" if is_favorite else "#FAFAFA"
                    name_color = "#FF6B35" if is_favorite else "#FAFAFA"
                    
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px; height: 52px; white-space: nowrap;">
                            <div style="font-weight: bold; font-size: 1.1rem; color: {rank_color}; min-width: 25px; text-align: center;">
                                {rank}
                            </div>
                            <div style="flex-shrink: 0; width: 52px; display: flex; justify-content: center;">
                                <img src="{logo_url}" style="width: 52px; height: 52px; filter: drop-shadow(0px 2px 3px rgba(0,0,0,0.5));">
                            </div>
                            <div style="margin-left: 4px;">
                                <div style="font-weight: bold; color: {name_color}; font-size: 0.95rem; line-height: 1.2;">
                                    {team_name}
                                </div>
                                <div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 2px;">
                                    (GP: {gp})
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                def st_header(text):
                    st.markdown(f'<div style="color: #9CA3AF; font-size: 0.75rem; font-weight: 600; white-space: nowrap; margin-bottom: 4px;">{text}</div>', unsafe_allow_html=True)

                def st_value(text, color="#FAFAFA"):
                    st.markdown(f'<div style="font-size: 1rem; white-space: nowrap; color: {color};">{text}</div>', unsafe_allow_html=True)

                with col3:
                    st_header("RECORD")
                    st_value(record)
                
                with col4:
                    st_header("GB")
                    st_value(gb if gb != 0 else "-")
                
                with col5:
                    st_header("WIN %")
                    st_value(win_pct)
                
                with col6:
                    st_header("HOME")
                    st_value(home, color=get_record_color(home))
                
                with col7:
                    st_header("ROAD")
                    st_value(road, color=get_record_color(road))
                
                with col8:
                    st_header("CONF")
                    st_value(conf_rec, color=get_record_color(conf_rec))
                
                
                with col10:
                    st_header("L10")
                    st_value(l10, color=get_record_color(l10))
                
                with col11:
                    st_header("STREAK")
                    st_value(streak, color=get_streak_color(streak))
                
                with col12:
                    st_header("ORTG")
                    if off_rtg != 'N/A':
                        st_value(f"{off_rtg} (#{off_rank})")
                    else:
                        st_value("N/A")
                
                with col13:
                    st_header("DRTG")
                    if def_rtg != 'N/A':
                        st_value(f"{def_rtg} (#{def_rank})")
                    else:
                        st_value("N/A")
                
                # Boundary markers - replace standard divider at tier transitions
                if rank == 6:
                    st.markdown("""
                    <div style="border-top: 2px dashed #F59E0B; margin: 15px 0 10px 0; padding-top: 5px;">
                        <span style="color: #F59E0B; font-size: 0.8rem; font-weight: bold;">PLAY-IN TOURNAMENT</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif rank == 10:
                    st.markdown("""
                    <div style="border-top: 2px dashed #EF4444; margin: 15px 0 10px 0; padding-top: 5px;">
                        <span style="color: #EF4444; font-size: 0.8rem; font-weight: bold;">OUT OF PLAYOFF RACE</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif rank < 15:
                    st.divider()
            
            # Display collected tiebreaker notes with orange legend
            st.markdown("---")
            st.markdown("""
            <div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 10px;">
                <span style="color: #FF6B35; font-weight: bold;">ORANGE</span> indicates a favorite team.
            </div>
            """, unsafe_allow_html=True)
            
            if tiebreaker_notes:
                st.caption(f"**Tiebreaker Notes:**")
                for note in tiebreaker_notes:
                    st.caption(note)
        
        with tab_west:
            st.markdown("### Western Conference")
            west_df = standings_df[standings_df['Conference'] == 'West']
            if not west_df.empty:
                display_conference_standings(west_df, user_favorite_teams, team_ratings, nba_schedule, standings_df)
            else:
                st.info("No Western Conference standings available.")
        
        with tab_east:
            st.markdown("### Eastern Conference")
            east_df = standings_df[standings_df['Conference'] == 'East']
            if not east_df.empty:
                display_conference_standings(east_df, user_favorite_teams, team_ratings, nba_schedule, standings_df)
            else:
                st.info("No Eastern Conference standings available.")
        
        with tab_playoffs:
            #st.markdown("### Playoff Picture (If Season Ended Today)")
            
            # Helper to get team by seed (defined once for the whole tab)
            def get_team_info_by_seed(conf_df, seed):
                team_row = conf_df[conf_df['PlayoffRank'] == seed]
                if not team_row.empty:
                    row = team_row.iloc[0]
                    city = row['TeamCity']
                    name = row['TeamName']
                    full_name = city if name in city else f"{city} {name}".strip()
                    if city == 'LA' and name == 'Clippers': full_name = "LA Clippers"
                        
                    abbrev = get_team_abbrev(city)
                    record = row['Record']
                    streak = str(row.get('strCurrentStreak', '')).replace(' ', '')
                    logo_url = get_team_logo_url(abbrev)
                    return {
                        'abbrev': abbrev,
                        'name': name,
                        'full_name': full_name,
                        'city': city,
                        'record': record,
                        'streak': streak,
                        'logo_url': logo_url,
                        'seed': seed
                    }
                return None

            west_df = standings_df[standings_df['Conference'] == 'West']
            east_df = standings_df[standings_df['Conference'] == 'East']

            # 1. PLAY-IN TOURNAMENT
            st.markdown("### Play-In Tournament")
            st.caption("_7 seed hosts 8 seed (winner gets #7). 9 seed hosts 10 seed (winner plays loser of 7/8 for #8)._")

            def render_play_in_card(team1, team2):
                """Render a play-in matchup card. Team1 is away, Team2 is home."""
                if not team1 or not team2: return
                # Use padding columns to center the @ symbol - increased c3 left padding
                pad1, c1, c2, c3, pad2 = st.columns([0.2, 1, 0.15, 1.2, 0.2])
                with c1:
                    logo = team1['logo_url']
                    cc1, cc2 = st.columns([0.35, 1])
                    if logo: cc1.image(logo, width=55)
                    cc2.markdown(f"**#{team1['seed']} {team1['full_name']}**")
                    cc2.caption(f"{team1['record']}")
                with c2:
                    st.markdown("<div style='margin-top: 25px; font-weight: bold; text-align: center;'>@</div>", unsafe_allow_html=True)
                with c3:
                    logo = team2['logo_url']
                    # Add spacer to shift logo+text right together
                    spacer, cc1, cc2 = st.columns([0.4, 0.35, 1])
                    if logo: cc1.image(logo, width=55)
                    cc2.markdown(f"**#{team2['seed']} {team2['full_name']}**")
                    cc2.caption(f"{team2['record']}")

            # Western Conference Play-In
            st.markdown("#### Western Conference")
            render_play_in_card(get_team_info_by_seed(west_df, 8), get_team_info_by_seed(west_df, 7))
            render_play_in_card(get_team_info_by_seed(west_df, 10), get_team_info_by_seed(west_df, 9))
            
            st.markdown("---")
            
            # Eastern Conference Play-In
            st.markdown("#### Eastern Conference")
            render_play_in_card(get_team_info_by_seed(east_df, 8), get_team_info_by_seed(east_df, 7))
            render_play_in_card(get_team_info_by_seed(east_df, 10), get_team_info_by_seed(east_df, 9))

            st.markdown("---")


            # 2. BRACKETS
            def render_playoff_bracket_html(conference_df, is_flipped=False):
                """Render a premium interactive playoff bracket for a conference."""
                teams = {s: get_team_info_by_seed(conference_df, s) for s in range(1, 11)}
                
                # CSS for the bracket
                flip_css = "flex-direction: row-reverse;" if is_flipped else ""
                align_css = "text-align: right; padding-right: 14px;" if is_flipped else "text-align: left; padding-left: 14px;"
                order_css = "flex-direction: row-reverse;" if is_flipped else "flex-direction: row;"

                bracket_css = f"""
                <style>
                    .bracket-wrapper {{
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 0;
                        width: 100%;
                        background-color: transparent;
                        font-family: 'Inter', -apple-system, sans-serif;
                    }}
                    .bracket-container {{
                        display: flex;
                        gap: 25px;
                        align-items: flex-start;
                        position: relative;
                        padding: 10px 0;
                        justify-content: center;
                        width: 100%;
                        {flip_css}
                    }}
                    .round {{
                        display: flex;
                        flex-direction: column;
                        gap: 15px;
                        justify-content: flex-start;
                        position: relative;
                    }}
                    .round-title {{
                        text-align: center;
                        font-size: 0.65rem;
                        color: #FF6B35;
                        font-weight: 700;
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        background: rgba(255, 107, 53, 0.1);
                        padding: 3px 6px;
                        border-radius: 4px;
                    }}
                    .matchup {{
                        width: 200px;
                        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
                        border: 1px solid #374151;
                        border-radius: 6px;
                        overflow: hidden;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                    }}
                    .team {{
                        display: flex;
                        align-items: center;
                        padding: 5px 8px;
                        gap: 8px;
                        height: 40px;
                        {order_css}
                    }}
                    .team:first-child {{
                        border-bottom: 1px solid #374151;
                    }}
                    .seed {{
                        font-size: 0.7rem;
                        color: #9CA3AF;
                        width: 12px;
                        font-weight: bold;
                        text-align: center;
                    }}
                    .logo-img {{
                        width: 28px;
                        height: 28px;
                        filter: drop-shadow(0px 1px 2px rgba(0,0,0,0.5));
                    }}
                    .team-info {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        flex-grow: 1;
                        {align_css}
                    }}
                    .team-name-primary {{
                        font-weight: 700;
                        font-size: 0.8rem;
                        color: #FAFAFA;
                    }}
                    .team-rec-small {{
                        font-size: 0.75rem;
                        color: #9CA3AF;
                        margin-left: 5px;
                    }}
                    .tbd-team {{ color: #4B5563; font-style: italic; }}
                </style>
                """

                def team_html(team, is_tbd=False):
                    if is_tbd or not team:
                        return f"""
                            <div class="team">
                                <div class="team-info"><span class="team-name-primary tbd-team">TBD</span></div>
                            </div>
                        """
                    logo = f'<img src="{team["logo_url"]}" class="logo-img">' if team['logo_url'] else '🏀'
                    
                    return f"""
                        <div class="team">
                            <span class="seed">{team['seed']}</span>
                            <div>{logo}</div>
                            <div class="team-info">
                                <div><span class="team-name-primary">{team['abbrev']}</span><span class="team-rec-small">({team['record']})</span></div>
                            </div>
                        </div>
                    """

                m1_8 = f'<div class="matchup">{team_html(teams.get(1))}{team_html(teams.get(8))}</div>'
                m4_5 = f'<div class="matchup">{team_html(teams.get(4))}{team_html(teams.get(5))}</div>'
                m2_7 = f'<div class="matchup">{team_html(teams.get(2))}{team_html(teams.get(7))}</div>'
                m3_6 = f'<div class="matchup">{team_html(teams.get(3))}{team_html(teams.get(6))}</div>'
                m_semi_1 = f'<div class="matchup">{team_html(None, True)}{team_html(None, True)}</div>'
                m_semi_2 = f'<div class="matchup">{team_html(None, True)}{team_html(None, True)}</div>'
                m_finals = f'<div class="matchup">{team_html(None, True)}{team_html(None, True)}</div>'

                full_html = f"""
                <div class="bracket-wrapper">
                    {bracket_css}
                    <div class="bracket-container">
                        <div class="round">
                            <div class="round-title">First Round</div>
                            {m1_8}{m4_5}{m2_7}{m3_6}
                        </div>
                        <div class="round">
                            <div class="round-title">Conf. Semifinals</div>
                            <div style="margin-top: 55px;">{m_semi_1}</div>
                            <div style="margin-top: 100px;">{m_semi_2}</div>
                        </div>
                        <div class="round">
                            <div class="round-title">Conf. Finals</div>
                            <div style="margin-top: 130px;">{m_finals}</div>
                        </div>
                    </div>
                </div>
                """
                return full_html

            # Render Brackets using Streamlit Components for better HTML isolation
            import streamlit.components.v1 as components
            
            # Western Bracket Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: #FF6B35; letter-spacing: 1px;'>WESTERN CONFERENCE BRACKET</h3>", unsafe_allow_html=True)
            components.html(render_playoff_bracket_html(west_df, is_flipped=False), height=520, scrolling=False)
            
            # Eastern Bracket Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<h3 style='text-align: center; color: #FF6B35; letter-spacing: 1px;'>EASTERN CONFERENCE BRACKET</h3>", unsafe_allow_html=True)
            components.html(render_playoff_bracket_html(east_df, is_flipped=True), height=520, scrolling=False)


        
        with tab_divisions:
            # ===== DIVISION STANDINGS =====
            st.markdown("## Division Standings")
            
            # Define team divisions
            divisions = {
                # Western Conference
                "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
                "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
                "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
                # Eastern Conference
                "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
                "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
                "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"]
            }
            
            def get_team_division(team_abbrev):
                """Get division name for a team."""
                for div, teams in divisions.items():
                    if team_abbrev in teams:
                        return div
                return None
            
            def display_division_standings(division_name, division_teams, all_standings, favorite_teams):
                """Display standings for a single division."""
                # Filter standings for this division's teams
                division_df = all_standings[all_standings.apply(
                    lambda row: get_team_abbrev(row['TeamCity']) in division_teams, axis=1
                )]
                
                if division_df.empty:
                    st.info(f"No standings available for {division_name} division.")
                    return
                
                # Sort by PlayoffRank (same as conference standings for consistent tiebreakers)
                division_df = division_df.sort_values('PlayoffRank', ascending=True)
                
                # Header row - using HTML for precise margin control
                header_logo, header1, header2, header3, header4, header5, header6 = st.columns([0.5, 0.3, 2.0, 0.7, 0.7, 0.7, 0.7])
                header_style = "color: #9CA3AF; font-size: 0.75rem; font-weight: 600; margin: 0; padding: 0;"
                with header1:
                    st.markdown(f"<p style='{header_style}'>#</p>", unsafe_allow_html=True)
                with header2:
                    st.markdown(f"<p style='{header_style}'>TEAM</p>", unsafe_allow_html=True)
                with header3:
                    st.markdown(f"<p style='{header_style}'>RECORD</p>", unsafe_allow_html=True)
                with header4:
                    st.markdown(f"<p style='{header_style}'>DIV REC</p>", unsafe_allow_html=True)
                with header5:
                    st.markdown(f"<p style='{header_style}'>SEED</p>", unsafe_allow_html=True)
                with header6:
                    st.markdown(f"<p style='{header_style}'>WIN %</p>", unsafe_allow_html=True)
                
                st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
                
                for idx, (_, row) in enumerate(division_df.iterrows(), 1):
                    team_abbrev = get_team_abbrev(row['TeamCity'])
                    is_favorite = team_abbrev in favorite_teams
                    
                    team_name = row['TeamCity'] if row['TeamName'] in row['TeamCity'] else f"{row['TeamCity']} {row['TeamName']}".strip()
                    record = row['Record']
                    div_rec = row.get('DivisionRecord', 'N/A')
                    win_pct = f"{row['WinPct']*100:.1f}" if isinstance(row['WinPct'], (int, float)) else row['WinPct']
                    conf_seed = int(row['PlayoffRank']) if 'PlayoffRank' in row else 'N/A'
                    logo_url = get_team_logo_url(team_abbrev)
                    
                    # Calculate GP
                    wins_val = row.get('Wins', 0)
                    loss_val = row.get('Losses', 0)
                    gp = int(wins_val) + int(loss_val) if (wins_val != 'N/A' and loss_val != 'N/A') else 0
                    
                    col_logo, col1, col2, col3, col4, col5, col6 = st.columns([0.5, 0.3, 2.0, 0.7, 0.7, 0.7, 0.7])
                    
                    with col_logo:
                        if logo_url:
                            st.image(logo_url, width=35)
                    
                    with col1:
                        if is_favorite:
                            st.markdown(f"**:orange[{idx}]**")
                        else:
                            st.markdown(f"**{idx}**")
                    
                    with col2:
                        # Color coding based on seed
                        seed_val = 99
                        if isinstance(conf_seed, int):
                            seed_val = conf_seed
                        
                        if seed_val <= 6:
                            # Playoff spots - Green
                            color = "#10B981"
                        elif seed_val <= 10:
                            # Play-in spots - Yellow/Orange
                            color = "#F59E0B" 
                        else:
                            color = "#FAFAFA"
                            
                        # If favorite, force bold/orange or mix? 
                        # User priority: "put teams in green/yellow". 
                        # We will make favorites Orange still to distinguish them, or maybe just bold?
                        # Let's check session state: favorite users prefer visibility. 
                        # Compromise: Use Seed Color, but add Star for Favorite.
                        
                        if is_favorite:
                            color = "#FF6B35" # Orange for favorites
                            
                        display_name = f"<span style='color: {color}; font-weight: bold; font-size: 0.95rem;'>{team_name}</span> <span style='color: #9CA3AF; font-size: 0.75rem;'>({gp} GP)</span>"
                        
                        st.markdown(display_name, unsafe_allow_html=True)
                    
                    with col3:
                        color = get_record_color(record)
                        st.markdown(f"<span style='color: white; font-weight: bold;'>{record}</span>", unsafe_allow_html=True)
                    
                    with col4:
                        color = get_record_color(div_rec)
                        st.markdown(f"<span style='color: {color};'>{div_rec}</span>", unsafe_allow_html=True)
                    
                    with col5:
                        st.write(f"#{conf_seed}")
                    
                    with col6:
                        st.write(win_pct)
                
                # Legend at bottom
                st.markdown("""
                <div style="margin-top: 15px; font-size: 0.8rem; color: #9CA3AF;">
                    <span style="color: #10B981; font-weight: bold;">●</span> Playoffs (1-6) &nbsp;&nbsp;
                    <span style="color: #F59E0B; font-weight: bold;">●</span> Play-In Tournament (7-10) &nbsp;&nbsp;
                    <span style="color: #FF6B35; font-weight: bold;">●</span> Favorite Team
                </div>
                """, unsafe_allow_html=True)
            
            # Create division tabs - all in one row
            all_divs = st.tabs(["Northwest", "Pacific", "Southwest", "Atlantic", "Central", "Southeast"])
            
            with all_divs[0]:
                st.markdown("### Northwest Division")
                display_division_standings("Northwest", divisions["Northwest"], standings_df, user_favorite_teams)
            
            with all_divs[1]:
                st.markdown("### Pacific Division")
                display_division_standings("Pacific", divisions["Pacific"], standings_df, user_favorite_teams)
            
            with all_divs[2]:
                st.markdown("### Southwest Division")
                display_division_standings("Southwest", divisions["Southwest"], standings_df, user_favorite_teams)
            
            with all_divs[3]:
                st.markdown("### Atlantic Division")
                display_division_standings("Atlantic", divisions["Atlantic"], standings_df, user_favorite_teams)
            
            with all_divs[4]:
                st.markdown("### Central Division")
                display_division_standings("Central", divisions["Central"], standings_df, user_favorite_teams)
            
            with all_divs[5]:
                st.markdown("### Southeast Division")
                display_division_standings("Southeast", divisions["Southeast"], standings_df, user_favorite_teams)


# ==================== AWARDS PAGE ====================
elif st.session_state.current_page == "Awards":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("NBA Awards")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Odds", use_container_width=True):
            with st.spinner("Refreshing DraftKings odds..."):
                import subprocess
                import sys
                
                # Command to run the script
                # We try to use sys.executable to stay in the same environment
                script_cmd = [sys.executable, "scripts/update_awards_odds.py", "--force"]
                
                try:
                    result = subprocess.run(
                        script_cmd, 
                        check=True, 
                        capture_output=True, 
                        text=True
                    )
                    st.cache_data.clear()
                    st.success("Odds refreshed!")
                    if result.stdout:
                        with st.expander("Show Log"):
                            st.code(result.stdout)
                    time.sleep(1)
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    # If it's a browser error, try to install playwright browsers
                    if "playwright install" in (e.stdout + e.stderr).lower():
                        st.info("Browser missing. Attempting to install Playwright Chromium specialized for this environment...")
                        try:
                            # Try both ways to be sure
                            subprocess.run(["playwright", "install", "chromium"], check=True, capture_output=True)
                            st.success("Browser binaries downloaded! Retrying refresh...")
                            # Retry the script
                            result = subprocess.run(script_cmd, check=True, capture_output=True, text=True)
                            st.cache_data.clear()
                            st.success("Odds refreshed!")
                            if result.stdout:
                                with st.expander("Show Log"):
                                    st.code(result.stdout)
                            time.sleep(1)
                            st.rerun()
                        except Exception as e2:
                            try:
                                # Fallback to sys.executable method
                                subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True, capture_output=True)
                                result = subprocess.run(script_cmd, check=True, capture_output=True, text=True)
                                st.cache_data.clear()
                                st.success("Odds refreshed (via fallback install)!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as install_err:
                                st.error(f"Failed to auto-install browser: {install_err}")
                                st.info("This is common on some cloud platforms. Please try clicking the button again as the first attempt may have initialized the path.")
                    
                    st.error(f"Error refreshing: {e}")
                    if e.stderr:
                        st.error("Error details:")
                        st.code(e.stderr)
                    if e.stdout:
                        st.info("Output before failure:")
                        st.code(e.stdout)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    
    # Auto-refresh odds when page loads if not fresh or already checked in this session
    if 'odds_auto_checked' not in st.session_state:
        st.session_state.odds_auto_checked = True
        with st.status("🔍 Checking for fresh odds...", expanded=False) as status:
            import subprocess
            import sys
            # Run without --force so it respects the script's internal 12h TTL
            script_cmd = [sys.executable, "scripts/update_awards_odds.py"]
            try:
                subprocess.run(script_cmd, check=True, capture_output=True, text=True)
                st.cache_data.clear()
                status.update(label="✅ Odds checked and synced!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="⚠️ Background refresh skipped.", state="complete", expanded=False)

    render_section_header("NBA Awards", "Season award favorites and betting odds")
    
    # Get required data for MVP Ladder
    standings_df = get_league_standings(season)
    mvp_ladder, mvp_date = get_mvp_ladder()
    
    # Pre-fetch stats for MVP candidates (and later awards)
    bulk_player_stats = get_bulk_player_stats()

    
    # ===== SECTION 1: MVP LADDER (Moved from Around the NBA) =====
    st.markdown("## 🏆 KIA MVP Ladder")
    st.caption(f"The latest rankings in the race for the 2025-26 MVP award (as of {mvp_date}). Updated weekly.")
    st.caption("Disclaimer: REC refers to the team record, not the player's individual record. For individual record, view season stats.")
    
    if mvp_ladder:
        # Helper function to render a single MVP card
        def render_mvp_card(player, rank_idx):
            rank_color = "#FFD700" if rank_idx == 0 else "#C0C0C0" if rank_idx == 1 else "#CD7F32" if rank_idx == 2 else "#667eea"
            
            st.markdown(f"""
            <div style="text-align: center; font-size: 1.2rem; font-weight: bold; color: {rank_color}; margin-bottom: 5px;">#{player['rank']}</div>
            """, unsafe_allow_html=True)
            
            raw_full = player.get('name', '')
            player_name = raw_full
            team_abbrev = player.get('team_abbrev')
            team_name = player.get('team', '')
            
            if ',' in raw_full:
                parts = raw_full.split(',')
                player_name = parts[0].strip()
                team_part = parts[1].strip()
                if team_part:
                    team_name = team_part
                sorted_cities = sorted(TEAM_ABBREV_MAP.keys(), key=len, reverse=True)
                for city in sorted_cities:
                    if city.lower() == team_part.lower():
                        team_abbrev = TEAM_ABBREV_MAP[city]
                        break

            player_photo_url = get_player_photo_url(player_name)
            team_logo_url = get_team_logo_url(team_abbrev) if team_abbrev else None
            
            games_played = player.get('games_played', 'N/A')
            player_stats = player.get('stats', 'N/A')
            shooting_stats = ""
            team_record = "N/A"
            team_rank = "N/A"
            team_conf = ""
            
            if player_name:
                try:
                    # Optimized lookup in bulk stats
                    found_stats = None
                    if bulk_player_stats is not None and not bulk_player_stats.empty:
                        match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower() == player_name.lower()]
                        if not match.empty:
                            found_stats = match.iloc[0]
                        else:
                            match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower().str.contains(player_name.lower())]
                            if not match.empty:
                                found_stats = match.iloc[0]
                    
                    if found_stats is not None:
                        games_played = str(int(found_stats['GP']))
                        ppg = found_stats['PTS']
                        rpg = found_stats['REB']
                        apg = found_stats['AST']
                        
                        fg_pct = found_stats['FG_PCT'] * 100
                        three_pct = found_stats['FG3_PCT'] * 100
                        ft_pct = found_stats['FT_PCT'] * 100
                        
                        shooting_stats = f"{format_pct(fg_pct)} FG% • {format_pct(three_pct)} 3P% • {format_pct(ft_pct)} FT%"
                        player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG"
                    else:
                         # Fallback to slow API
                         df_log, _ = get_player_game_log(player_name)
                         if df_log is not None and not df_log.empty:
                            games_played = str(len(df_log))
                            ppg = df_log['Points'].mean()
                            rpg = df_log['Rebounds'].mean()
                            apg = df_log['Assists'].mean()
                            player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG"
                except:
                    pass
            
            if not standings_df.empty and team_name:
                matching = standings_df[standings_df['TeamCity'].str.contains(team_name.split()[0], case=False, na=False)]
                if not matching.empty:
                    team_record = matching.iloc[0]['Record']
                    team_rank = matching.iloc[0].get('PlayoffRank', 'N/A')
                    team_conf = matching.iloc[0].get('Conference', '')
            
            if team_logo_url:
                # Slightly shifted right for better balance
                spacer1, photo_col, logo_col, spacer2 = st.columns([1.35, 4.5, 4.5, 0.65])
                with photo_col:
                    if player_photo_url:
                        st.image(player_photo_url, width=220)
                with logo_col:
                    if team_logo_url:
                        st.image(team_logo_url, width=200)
            else:
                if player_photo_url:
                    st.image(player_photo_url, width=100)
            
            # Fetch position from bio
            pos_label = ""
            try:
                bio = fetch_player_bio(player_name)
                if bio and bio.get('position'):
                    abbrev_pos = abbreviate_position(bio['position'], player_name)
                    pos_label = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>({abbrev_pos})</span>"
            except:
                pass
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                <div style="color: #FAFAFA; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; line-height: 1.2;">{player_name}{pos_label}</div>
                <div style="color: #9CA3AF; font-size: 0.75rem; margin-bottom: 2px;">{player_stats}</div>
                <div style="color: #6B7280; font-size: 0.75rem; margin-bottom: 8px;">{shooting_stats}</div>
                <div style="display: flex; justify-content: center; gap: 12px; font-size: 0.75rem;">
                    <div style="background: #374151; padding: 2px 8px; border-radius: 4px;">
                        <span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{games_played}</span>
                    </div>
                    <div style="background: #374151; padding: 2px 8px; border-radius: 4px;">
                        <span style="color: #9CA3AF;">REC:</span> <span style="color: #FAFAFA; font-weight: bold;">{team_record}</span> | <span style="color: #FAFAFA; font-weight: bold;">#{team_rank} in {team_conf}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 1: Top 5
        cols = st.columns(5)
        for i, player in enumerate(mvp_ladder[:5]):
            with cols[i]:
                render_mvp_card(player, i)
        
        # Row 2: 6-10 (if available)
        if len(mvp_ladder) > 5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### The Next 5")
            cols2 = st.columns(5)
            for i, player in enumerate(mvp_ladder[5:10]):
                with cols2[i]:
                    render_mvp_card(player, i + 5)
    else:
        st.info("MVP Ladder data currently unavailable.")
    
    st.markdown("---")
    
    # ===== SECTION 2: ROOKIE OF THE YEAR LADDER =====
    st.markdown("## 🌟 Rookie of the Year Ladder")
    rookie_ladder, rookie_date = get_rookie_ladder()
    st.caption(f"The latest rankings in the race for the 2025-26 Kia ROY award (as of {rookie_date}). Updated weekly. ")
    
    if rookie_ladder:
        def render_rookie_card(player, rank_idx):
            rank_color = "#FFD700" if rank_idx == 0 else "#C0C0C0" if rank_idx == 1 else "#CD7F32" if rank_idx == 2 else "#667eea"
            player_name = player['name']
            team_abbrev = player.get('team_abbrev', 'N/A')
            draft_pick = player.get('draft_pick', 'N/A')
            
            # Handle special characters in names for API lookup
            api_lookup_name = player_name
            # Map common special character variations
            name_mappings = {
                'Egor Demin': 'Egor Dëmin',
                'Egor Dëmin': 'Egor Dëmin',
            }
            api_lookup_name = name_mappings.get(player_name, player_name)
            
            player_photo_url = get_player_photo_url(api_lookup_name)
            team_logo_url = get_team_logo_url(team_abbrev)
            
            # Fetch stats from bulk data (overrides scraped stats with current API data)
            player_stats = ""
            shooting_stats = ""
            games_played = "N/A"
            team_record = "N/A"
            team_rank = "N/A"
            team_conf = ""
            
            try:
                if bulk_player_stats is not None and not bulk_player_stats.empty:
                    # Try exact match first
                    match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower() == api_lookup_name.lower()]
                    # If no match, try without special chars
                    if match.empty:
                        match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower() == player_name.lower()]
                    # Try contains as fallback
                    if match.empty:
                        # Try partial match on last name
                        last_name = player_name.split()[-1] if ' ' in player_name else player_name
                        match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.contains(last_name, case=False, na=False)]
                    
                    if not match.empty:
                        s = match.iloc[0]
                        games_played = s.get('GP', 'N/A')
                        ppg = s.get('PTS', 0)
                        rpg = s.get('REB', 0)
                        apg = s.get('AST', 0)
                        player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG"
                        fg = s.get('FG_PCT', 0) * 100
                        fg3 = s.get('FG3_PCT', 0) * 100
                        ft = s.get('FT_PCT', 0) * 100
                        shooting_stats = f"{fg:.1f}% FG% · {fg3:.1f}% 3P% · {ft:.1f}% FT%"
                        team_abbrev = s.get('TEAM_ABBREVIATION', team_abbrev)
                        team_logo_url = get_team_logo_url(team_abbrev)
                
                # Get team record
                if standings_df is not None and not standings_df.empty:
                    team_stand = standings_df[standings_df.apply(lambda r: get_team_abbrev(r['TeamCity']) == team_abbrev, axis=1)]
                    if not team_stand.empty:
                        team_record = team_stand.iloc[0].get('Record', 'N/A')
                        team_rank = int(team_stand.iloc[0].get('PlayoffRank', 0))
                        team_conf = team_stand.iloc[0].get('Conference', '')[:1]
                        team_conf = "East" if team_conf == "E" else "West" if team_conf == "W" else team_conf
            except:
                pass
            
            # If no stats from API, use scraped stats as fallback
            if not player_stats:
                scraped = player.get('stats', 'N/A')
                if scraped != "N/A":
                    player_stats = scraped.replace(",", " ·")
            
            # Display rank at top center (like MVP card)
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 5px;">
                <span style="color: {rank_color}; font-weight: bold; font-size: 1.5rem;">#{rank_idx + 1}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Photo with team logo side by side (like the MVP card)
            col_photo, col_logo = st.columns([1, 1])
            with col_photo:
                if player_photo_url:
                    st.image(player_photo_url, width=90)
            with col_logo:
                if team_logo_url:
                    st.image(team_logo_url, width=90)
            
            # Fetch position from bio
            pos_label = ""
            try:
                bio = fetch_player_bio(api_lookup_name)
                if not bio:
                    bio = fetch_player_bio(player_name)
                if bio and bio.get('position'):
                    abbrev_pos = abbreviate_position(bio['position'], player_name)
                    pos_label = abbrev_pos
            except:
                pass
            
            # Format draft pick as (Rd1, #1) or (Rd2, #42) etc.
            draft_label = ""
            if draft_pick and draft_pick != "N/A" and draft_pick != "Undrafted":
                pick_num = int(draft_pick) if draft_pick.isdigit() else 0
                if pick_num > 0:
                    if pick_num <= 30:
                        draft_label = f"(Rd1, #{pick_num})"
                    else:
                        draft_label = f"(Rd2, #{pick_num})"
            elif draft_pick == "Undrafted":
                draft_label = "(Undrafted)"
            
            # Combine position and draft pick
            pos_draft = ""
            if pos_label and draft_label:
                pos_draft = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>({pos_label}) {draft_label}</span>"
            elif pos_label:
                pos_draft = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>({pos_label})</span>"
            elif draft_label:
                pos_draft = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>{draft_label}</span>"
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                <div style="color: #FAFAFA; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; line-height: 1.2;">{player_name}{pos_draft}</div>
                <div style="color: #9CA3AF; font-size: 0.75rem; margin-bottom: 2px;">{player_stats}</div>
                <div style="color: #6B7280; font-size: 0.75rem; margin-bottom: 8px;">{shooting_stats}</div>
                <div style="display: flex; flex-wrap: nowrap; justify-content: center; gap: 8px; font-size: 0.7rem;">
                    <div style="background: #374151; padding: 2px 6px; border-radius: 4px; white-space: nowrap;">
                        <span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{games_played}</span>
                    </div>
                    <div style="background: #374151; padding: 2px 6px; border-radius: 4px; white-space: nowrap;">
                        <span style="color: #9CA3AF;">REC:</span> <span style="color: #FAFAFA; font-weight: bold;">{team_record}</span> | <span style="color: #FAFAFA; font-weight: bold;">#{team_rank} in {team_conf}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Render top 5 rookies in a row
        cols = st.columns(5)
        for i, player in enumerate(rookie_ladder[:5]):
            with cols[i]:
                render_rookie_card(player, i)
        
        # Show "Next 5" if available
        if len(rookie_ladder) > 5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### The Next 5")
            cols2 = st.columns(5)
            for i, player in enumerate(rookie_ladder[5:10]):
                with cols2[i]:
                    render_rookie_card(player, i + 5)
    else:
        st.info("Rookie Ladder data currently unavailable.")
    
    st.markdown("---")
    
    # ===== SECTION 3: BETTING ODDS FOR OTHER AWARDS =====
    st.markdown("## 🎲 Other Award Favorites")
    st.caption("Top 5 betting favorites per award. Odds may be affected by 65-game rule.")
    
    data_path = "data/awards_odds.json"
    
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as f:
                awards_data = json.load(f)
            
            last_modified = os.path.getmtime(data_path)
            user_tz = pytz.timezone(st.session_state.get('user_timezone', 'US/Pacific'))
            local_dt = datetime.fromtimestamp(last_modified, tz=user_tz)
            last_updated_str = local_dt.strftime('%Y-%m-%d %I:%M %p')
            st.caption(f"Odds Updated: {last_updated_str} | Source: DraftKings")
            
            # Load Coach-to-Team mapping from CSV
            coach_team_map = {}
            try:
                coaches_df = pd.read_csv('nbacoaches.csv')
                for _, row in coaches_df.iterrows():
                    coach_name = row['NAME'].strip()
                    team_name = row['TEAM'].strip()
                    team_abbrev = row['ABBREV'].strip()  # Use abbreviation from CSV
                    coach_team_map[coach_name] = {"team": team_name, "abbrev": team_abbrev}
            except Exception as e:
                # Fallback to empty if CSV not found
                pass
            

            
            # Helper function to render a PLAYER award card with stats (MVP Ladder style)
            def render_player_award_card(candidate, rank_idx, odds, award_name=""):
                rank_color = "#FFD700" if rank_idx == 0 else "#C0C0C0" if rank_idx == 1 else "#CD7F32" if rank_idx == 2 else "#667eea"
                player_name = candidate
                
                player_photo_url = get_player_photo_url(player_name)
                team_abbrev = None
                team_logo_url = None
                
                # Fetch player stats, shooting splits, GP, and team
                player_stats = ""
                shooting_stats = ""
                games_played = "N/A"
                team_record = "N/A"
                team_rank = "N/A"
                team_conf = ""
                
                try:
                    # Optimized lookup in bulk stats
                    found_stats = None
                    if bulk_player_stats is not None and not bulk_player_stats.empty:
                        # Try exact match first
                        match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower() == player_name.lower()]
                        if not match.empty:
                            found_stats = match.iloc[0]
                        else:
                            # Try fuzzy match if needed (simple contains)
                            match = bulk_player_stats[bulk_player_stats['PLAYER_NAME'].str.lower().str.contains(player_name.lower())]
                            if not match.empty:
                                found_stats = match.iloc[0]
                    
                    if found_stats is not None:
                        games_played = str(int(found_stats['GP']))
                        
                        ppg = found_stats['PTS']
                        rpg = found_stats['REB']
                        apg = found_stats['AST']
                        
                        if "Defensive Player" in award_name:
                            spg = found_stats['STL']
                            bpg = found_stats['BLK']
                            player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG · {spg:.1f} SPG · {bpg:.1f} BPG"
                        else:
                            player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG"
                        
                        # Stats are decimals in leaguedashplayerstats, need to multiply by 100 for display
                        fg_pct = found_stats['FG_PCT'] * 100
                        three_pct = found_stats['FG3_PCT'] * 100
                        ft_pct = found_stats['FT_PCT'] * 100
                        
                        shooting_stats = f"{format_pct(fg_pct)} FG% • {format_pct(three_pct)} 3P% • {format_pct(ft_pct)} FT%"
                        
                        # Team info
                        team_abbrev = found_stats['TEAM_ABBREVIATION']
                        team_logo_url = get_team_logo_url(team_abbrev) if team_abbrev else None
                        
                        # Get team record from standings
                        if team_abbrev and not standings_df.empty:
                            for _, row in standings_df.iterrows():
                                row_abbrev = get_team_abbrev(row['TeamCity'])
                                if row_abbrev == team_abbrev:
                                    team_record = row['Record']
                                    team_rank = row.get('PlayoffRank', 'N/A')
                                    team_conf = row.get('Conference', '')
                                    break
                    else:
                        # Fallback to old slow method if not found in bulk for some reason
                         df_log, _ = get_player_game_log(player_name)
                         if df_log is not None and not df_log.empty:
                            games_played = str(len(df_log))
                            ppg = df_log['Points'].mean()
                            rpg = df_log['Rebounds'].mean()
                            apg = df_log['Assists'].mean()
                            player_stats = f"{ppg:.1f} PPG · {rpg:.1f} RPG · {apg:.1f} APG"
                except Exception as e:
                    # st.error(f"Error: {e}") 
                    pass
                
                # Rank badge
                st.markdown(f"""
                <div style="text-align: center; font-size: 1.2rem; font-weight: bold; color: {rank_color}; margin-bottom: 5px;">#{rank_idx + 1}</div>
                """, unsafe_allow_html=True)
                
                # Show player photo and team logo side by side - Slightly shifted right
                if team_logo_url and player_photo_url:
                    spacer1, photo_col, logo_col, spacer2 = st.columns([1.35, 4.5, 4.5, 0.65])
                    with photo_col:
                        st.image(player_photo_url, width=220)
                    with logo_col:
                        st.image(team_logo_url, width=200)
                elif player_photo_url:
                    st.image(player_photo_url, width=220)
                
                # Fetch position from bio (This is still an API call, but we can't easily avoid it without bulk bio. 
                # Optimization: Could cache bio or skip position if slow. For now we leave it as it's just 1 call instead of 2)
                pos_label = ""
                try:
                    # Try to use position from bulk stats if available? 
                    # leaguedashplayerstats doesn't have position. leaguedashplayerbiostats or commonallplayers does.
                    # For now, we'll keep the bio fetch but it's much faster than game log.
                    bio = fetch_player_bio(player_name)
                    if bio and bio.get('position'):
                        abbrev_pos = abbreviate_position(bio['position'], player_name)
                        pos_label = f" <span style='color: #9CA3AF; font-weight: normal; font-size: 0.8rem;'>({abbrev_pos})</span>"
                except:
                    pass
                
                # Further reduce font size for defensive stats to prevent overflow
                stat_font_size = "0.72rem"
                if "Defensive Player" in award_name:
                    stat_font_size = "0.62rem"

                # Player name, stats, shooting splits, GP/REC badge
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <div style="color: #FAFAFA; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px; line-height: 1.2; display: flex; justify-content: center; align-items: center; gap: 8px;">
                        {player_name}{pos_label}
                        <span style="background: #10B981; color: #FFF; font-weight: bold; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; display: inline-block;">{odds}</span>
                    </div>
                    <div style="color: #9CA3AF; font-size: {stat_font_size}; margin-bottom: 2px;">{player_stats}</div>
                    <div style="color: #6B7280; font-size: 0.75rem; margin-bottom: 8px;">{shooting_stats}</div>
                    <div style="display: flex; justify-content: center; gap: 8px; font-size: 0.75rem; flex-wrap: wrap;">
                        <div style="background: #374151; padding: 2px 8px; border-radius: 4px;">
                            <span style="color: #9CA3AF;">GP:</span> <span style="color: #FAFAFA; font-weight: bold;">{games_played}</span>
                        </div>
                        <div style="background: #374151; padding: 2px 8px; border-radius: 4px;">
                            <span style="color: #9CA3AF;">REC:</span> <span style="color: #FAFAFA; font-weight: bold;">{team_record}</span> | <span style="color: #FAFAFA; font-weight: bold;">#{team_rank} in {team_conf}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Helper function to render a COACH award card (centered logo, no photo)
            def render_coach_award_card(coach_name, rank_idx, odds, standings_df):
                rank_color = "#FFD700" if rank_idx == 0 else "#C0C0C0" if rank_idx == 1 else "#CD7F32" if rank_idx == 2 else "#667eea"
                
                team_info = coach_team_map.get(coach_name, None)
                team_name = team_info['team'] if team_info else "Unknown Team"
                team_abbrev = team_info['abbrev'] if team_info else None
                team_record = "N/A"
                team_rank = "N/A"
                team_conf = ""
                
                if team_abbrev and not standings_df.empty:
                    matching = standings_df[standings_df['TeamCity'].str.contains(team_name.split()[0], case=False, na=False)]
                    if not matching.empty:
                        team_record = matching.iloc[0]['Record']
                        team_rank = matching.iloc[0].get('PlayoffRank', 'N/A')
                        team_conf = matching.iloc[0].get('Conference', '')
                
                team_logo_url = get_team_logo_url(team_abbrev) if team_abbrev else None
                
                # Rank badge
                st.markdown(f"""
                <div style="text-align: center; font-size: 1.2rem; font-weight: bold; color: {rank_color}; margin-bottom: 5px;">#{rank_idx + 1}</div>
                """, unsafe_allow_html=True)
                
                # Centered team logo - Slightly shifted right
                if team_logo_url:
                    spacer1, logo_col, spacer2 = st.columns([1.35, 4, 0.65])
                    with logo_col:
                        st.image(team_logo_url, width=220)
                
                # Coach name, team, record, odds badge
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <div style="color: #FAFAFA; font-weight: bold; font-size: 0.95rem; margin-bottom: 5px; display: flex; justify-content: center; align-items: center; gap: 8px;">
                        {coach_name}
                        <span style="background: #10B981; color: #FFF; font-weight: bold; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; display: inline-block;">{odds}</span>
                    </div>
                    <div style="color: #9CA3AF; font-size: 0.8rem; margin-bottom: 2px;">{team_name}</div>
                    <div style="display: flex; justify-content: center; gap: 12px; font-size: 0.75rem; margin-top: 8px;">
                        <div style="background: #374151; padding: 2px 8px; border-radius: 4px;">
                            <span style="color: #9CA3AF;">REC:</span> <span style="color: #FAFAFA; font-weight: bold;">{team_record}</span> | <span style="color: #FAFAFA; font-weight: bold;">#{team_rank} in {team_conf}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Separate Coach of the Year from player awards, and exclude MVP odds
            player_awards = [a for a in awards_data if "Coach" not in a['award_name'] and "MVP" not in a['award_name'] and "Rookie" not in a['award_name']]
            coach_awards = [a for a in awards_data if "Coach" in a['award_name']]
            
            # Logic to collect ties for footnotes
            ties_list = []
            
            def collect_award_ties(candidates, award_name, is_coach=False):
                """Collect ties among top 10 candidates, ensuring at least one is in top 5."""
                # Use top 10 for tie checking
                check_cands = candidates[:10]
                odds_map = {}
                for i, cand in enumerate(check_cands):
                    # Normalize odds string (strip and replace special minus)
                    o = cand['odds'].strip().replace('\u2212', '-')
                    if o in odds_map:
                        odds_map[o].append({"name": cand['player'], "rank": i})
                    else:
                        odds_map[o] = [{"name": cand['player'], "rank": i}]
                
                award_ties = []
                for o, hitters in odds_map.items():
                    if len(hitters) > 1:
                        # Only add if at least one player in the tie is in the top 5
                        if any(h['rank'] < 5 for h in hitters):
                            # Add all unique pairs
                            for i in range(len(hitters)):
                                for j in range(i + 1, len(hitters)):
                                    award_ties.append({
                                        "award": award_name,
                                        "p1": hitters[i]['name'],
                                        "p2": hitters[j]['name'],
                                        "odds": o,
                                        "is_coach": is_coach
                                    })
                return award_ties

            # Render PLAYER awards first
            for award in player_awards:
                st.markdown(f"### {award['award_name']}")
                
                # Use helper for more inclusive tie detection
                ties_list.extend(collect_award_ties(award['candidates'], award['award_name']))
                
                top_candidates = award['candidates'][:5]
                cols = st.columns(5)
                for idx, cand in enumerate(top_candidates):
                    with cols[idx]:
                        render_player_award_card(cand['player'], idx, cand['odds'], award['award_name'])
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Render COACH awards at the bottom
            for award in coach_awards:
                st.markdown(f"### {award['award_name']}")
                
                # Use helper for more inclusive tie detection
                ties_list.extend(collect_award_ties(award['candidates'], award['award_name'], is_coach=True))
                
                top_candidates = award['candidates'][:5]
                cols = st.columns(5)
                for idx, cand in enumerate(top_candidates):
                    with cols[idx]:
                        render_coach_award_card(cand['player'], idx, cand['odds'], standings_df)
                
                st.markdown("<br>", unsafe_allow_html=True)

            # Footer section for Ties
            if ties_list:
                st.markdown("---")
                st.markdown("#### 🔗 Tied Favorites")
                for tie in ties_list:
                    p1 = tie['p1']
                    p2 = tie['p2']
                    
                    # Get teams for both players/coaches
                    p1_team_info = ""
                    p2_team_info = ""
                    
                    if tie.get('is_coach'):
                        t_info1 = coach_team_map.get(p1)
                        p1_team_info = f" ({t_info1['abbrev']})" if t_info1 else ""
                        t_info2 = coach_team_map.get(p2)
                        p2_team_info = f" ({t_info2['abbrev']})" if t_info2 else ""
                    else:
                        try:
                            # Use cached bio fetch for quick lookup
                            p1_bio = fetch_player_bio(p1)
                            p1_team_info = f" ({p1_bio.get('team_abbrev', 'N/A')})" if p1_bio else ""
                            p2_bio = fetch_player_bio(p2)
                            p2_team_info = f" ({p2_bio.get('team_abbrev', 'N/A')})" if p2_bio else ""
                        except:
                            pass
                    
                    col_text, col_btn = st.columns([4, 1])
                    with col_text:
                        st.markdown(f"**{p1}**{p1_team_info} is tied with **{p2}**{p2_team_info} for {tie['award']} at **{tie['odds']}**")
                    with col_btn:
                        # Use unique key for button
                        btn_key = f"compare_{p1}_{p2}_{tie['award']}".replace(" ", "_")
                        st.button(f"Compare Players", key=btn_key, type="secondary", use_container_width=True,
                                  on_click=nav_to_compare, args=(p1, p2))
            
            st.markdown("---")
            # st.info("💡 To update odds, run `python scripts/update_awards_odds.py`")
            
        except Exception as e:
            st.error(f"Error loading awards data: {e}")
    else:
        st.warning("Awards data not found. Please run the data update script.")


# ==================== ABOUT PAGE ====================
elif st.session_state.current_page == "About":
    st.title("About NBA Live Stats Predictor")
    
    st.markdown("""
    ### Overview
    
    This application uses **Hidden Markov Models (HMM)** combined with real-time NBA data 
    to predict player performance in upcoming games.
    
    ### How It Works
    
    1. **Data Collection**: Fetches live game logs and defensive ratings from NBA.com API
    2. **Model Training**: Trains an HMM on player's historical performance
    3. **Prediction**: Uses recent games and opponent defense to forecast stats
    4. **Adjustment**: Applies defensive rating factor to offensive statistics
    
    ### Key Features
    
    - **Real-time Data**: Always uses the latest NBA statistics
    - **Opponent-Aware**: Considers opponent's defensive strength
    - **Recent Form**: Weights recent games more heavily
    - **Personalized**: Save your favorite players and teams (requires login)
    
    ### Statistics Predicted
    
    - Points (PTS)
    - Assists (AST)
    - Rebounds (REB)
    - Steals (STL)
    - Blocks (BLK)
    - Turnovers (TOV)
    
    ### Technologies
    
    - **Streamlit**: Interactive web interface
    - **nba_api**: Official NBA data source
    - **hmmlearn**: Hidden Markov Model implementation
    - **SQLite**: User preferences storage
    - **Google OAuth**: Secure authentication
    
    ### Disclaimer
    
    This tool is for entertainment and educational purposes. Predictions are based on 
    statistical models and may not reflect actual game outcomes.
    """)
