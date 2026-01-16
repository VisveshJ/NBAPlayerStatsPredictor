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
    'Los Angeles': 'LAL', 'Golden State': 'GSW', 'Phoenix': 'PHX', 'Denver': 'DEN',
    'Memphis': 'MEM', 'Sacramento': 'SAC', 'Dallas': 'DAL', 'New Orleans': 'NOP',
    'LA': 'LAC', 'Minnesota': 'MIN', 'Oklahoma City': 'OKC', 'Portland': 'POR',
    'Utah': 'UTA', 'San Antonio': 'SAS', 'Houston': 'HOU',
    'Boston': 'BOS', 'Milwaukee': 'MIL', 'Philadelphia': 'PHI', 'Cleveland': 'CLE',
    'New York': 'NYK', 'Brooklyn': 'BKN', 'Miami': 'MIA', 'Atlanta': 'ATL',
    'Chicago': 'CHI', 'Toronto': 'TOR', 'Indiana': 'IND', 'Washington': 'WAS',
    'Orlando': 'ORL', 'Charlotte': 'CHA', 'Detroit': 'DET'
}

def get_team_abbrev(city):
    """Get team abbreviation from city name."""
    for key in TEAM_ABBREV_MAP:
        if key in city:
            return TEAM_ABBREV_MAP[key]
    return city[:3].upper()

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
                'TeamAbbrev': row.get('TeamCity', '')[:3].upper() if 'TeamCity' in row else '',
                'TeamName': row.get('TeamName', ''),
                'TeamCity': row.get('TeamCity', ''),
                'Conference': row.get('Conference', ''),
                'Division': row.get('Division', ''),
                'ConferenceRecord': row.get('ConferenceRecord', ''),
                'DivisionRecord': row.get('DivisionRecord', ''),
                'PlayoffRank': row.get('PlayoffRank', 0),
                'Wins': row.get('WINS', 0),
                'Losses': row.get('LOSSES', 0),
                'WinPct': row.get('WinPCT', 0),
                'Record': f"{row.get('WINS', 0)}-{row.get('LOSSES', 0)}",
                'HOME': row.get('HOME', ''),
                'ROAD': row.get('ROAD', ''),
                'L10': row.get('L10', ''),
                'strCurrentStreak': row.get('strCurrentStreak', ''),
                'PointsPG': row.get('PointsPG', 0),
                'OppPointsPG': row.get('OppPointsPG', 0),
                'GB': row.get('ConferenceGamesBack', '-'),
            }
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
    
    today = datetime.now().date()
    
    # Build playoff rank lookup from standings
    team_ranks = {}
    for _, row in standings_df.iterrows():
        # Map city to abbreviation
        city = row.get('TeamCity', '')
        abbrev = None
        abbrev_map = {
            'Los Angeles': 'LAL', 'Golden State': 'GSW', 'Phoenix': 'PHX', 'Denver': 'DEN',
            'Memphis': 'MEM', 'Sacramento': 'SAC', 'Dallas': 'DAL', 'New Orleans': 'NOP',
            'LA': 'LAC', 'Minnesota': 'MIN', 'Oklahoma City': 'OKC', 'Portland': 'POR',
            'Utah': 'UTA', 'San Antonio': 'SAS', 'Houston': 'HOU',
            'Boston': 'BOS', 'Milwaukee': 'MIL', 'Philadelphia': 'PHI', 'Cleveland': 'CLE',
            'New York': 'NYK', 'Brooklyn': 'BKN', 'Miami': 'MIA', 'Atlanta': 'ATL',
            'Chicago': 'CHI', 'Toronto': 'TOR', 'Indiana': 'IND', 'Washington': 'WAS',
            'Orlando': 'ORL', 'Charlotte': 'CHA', 'Detroit': 'DET'
        }
        for key in abbrev_map:
            if key in city:
                abbrev = abbrev_map[key]
                break
        if abbrev:
            team_ranks[abbrev] = {
                'rank': int(row.get('PlayoffRank', 0)),
                'name': row.get('TeamName', ''),
                'conference': row.get('Conference', '')
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
            opponent_info = team_ranks.get(opponent, {'rank': 0, 'name': opponent, 'conference': ''})
            
            upcoming.append({
                'date': game_date.strftime('%b %d'),
                'opponent': opponent,
                'opponent_name': opponent_info['name'],
                'opponent_rank': opponent_info['rank'],
                'opponent_conference': opponent_info['conference'],
                'is_home': is_home,
            })
            
            if len(upcoming) >= num_games:
                break
    
    return upcoming


def get_todays_games(schedule, standings_df):
    """Get games scheduled for today with team seeds."""
    from datetime import datetime
    
    if not schedule:
        return []
    
    today = datetime.now().date()
    
    # Build playoff rank lookup from standings
    team_ranks = {}
    for _, row in standings_df.iterrows():
        abbrev = get_team_abbrev(row.get('TeamCity', ''))
        if abbrev:
            team_ranks[abbrev] = {
                'rank': int(row.get('PlayoffRank', 0)),
                'name': row.get('TeamName', ''),
                'conference': row.get('Conference', ''),
                'record': row.get('Record', '')
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
            home_info = team_ranks.get(home_team, {'rank': 0, 'name': home_team, 'conference': ''})
            away_info = team_ranks.get(away_team, {'rank': 0, 'name': away_team, 'conference': ''})
            
            # Parse game time and convert to PST
            game_time_pst = ""
            try:
                game_time_utc = game.get('game_time_utc', '')
                if game_time_utc:
                    # Parse UTC time (format: 2026-01-15T00:30:00Z)
                    utc_dt = datetime.strptime(game_time_utc, '%Y-%m-%dT%H:%M:%SZ')
                    # PST is UTC-8
                    from datetime import timedelta
                    pst_dt = utc_dt - timedelta(hours=8)
                    game_time_pst = pst_dt.strftime('%I:%M %p').lstrip('0')
            except:
                pass
            
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
                'game_status': game.get('game_status', 1),
                'game_time': game_time_pst,
                'game_time_sort': game.get('game_time_utc', ''),  # For sorting
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

@st.cache_data(ttl=1800)
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


@st.cache_data(ttl=86400)
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
                today = datetime.now()
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

# Navigation options based on auth status
if is_authenticated:
    nav_options = ["Home", "Predictions", "Player Stats", "Compare Players", "Standings", "Favorites", "About"]
else:
    nav_options = ["Home", "Predictions", "Player Stats", "Compare Players", "Standings", "About"]

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
st.session_state.current_page = page

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


# ==================== HOME PAGE ====================
if page == "Home":
    if not is_authenticated:
        # Show login page for unauthenticated users
        render_login_page()
        
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #9CA3AF;'>Sign in to save your favorites</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            auth.show_login_button()
        
        st.markdown("---")
        st.info("👈 You can still use **Live Predictions** and **Player Stats** without logging in!")
    
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
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        st.write(f"**{player_name}**")
                    with col_b:
                        if st.button("View", key=f"home_view_{player_name}", help="View stats"):
                            st.session_state["redirect_to_predictions"] = player_name
                            st.session_state["auto_load_player"] = player_name
                            st.session_state['pending_nav_target'] = "Live Predictions"
                            st.rerun()
                    with col_c:
                        if st.button("X", key=f"home_remove_{player_name}", help="Remove"):
                            auth.remove_favorite_player(player_name)
                            st.rerun()
            else:
                render_empty_state(
                    "No favorite players yet! Go to Live Predictions to add some.",
                    ""
                )
        
        # ===== FAVORITE TEAMS SECTION =====
        with col2:
            render_section_header("Your Watched Teams", "")
            
            favorite_teams = auth.get_favorite_teams()
            team_def_ratings = get_current_defensive_ratings(season)
            
            if favorite_teams:
                for team_abbrev in favorite_teams[:6]:
                    col_a, col_b = st.columns([4, 1])
                    rating = team_def_ratings.get(team_abbrev, "N/A")
                    with col_a:
                        st.write(f"**{team_abbrev}** - DEF RTG: {rating}")
                    with col_b:
                        if st.button("X", key=f"home_remove_team_{team_abbrev}", help="Remove"):
                            auth.remove_favorite_team(team_abbrev)
                            st.rerun()
            else:
                render_empty_state(
                    "No watched teams yet! Add teams from the Live Predictions page.",
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
                        if st.button("View", key=f"quick_{player}"):
                            st.session_state["redirect_to_predictions"] = player
                            st.session_state["auto_load_player"] = player
                            st.session_state['pending_nav_target'] = "Live Predictions"
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
    todays_games = get_todays_games(nba_schedule, standings_df)
    
    # Fetch live/final scores
    scoreboard = get_todays_scoreboard()
    
    if todays_games:
        from datetime import datetime
        st.caption(f"**{datetime.now().strftime('%A, %B %d, %Y')}** • {len(todays_games)} game(s) • _All times in PST_")
        
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
                        away_conf = f"({game['away_conference'][0]})" if game['away_conference'] else ""
                        home_conf = f"({game['home_conference'][0]})" if game['home_conference'] else ""
                        
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
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1F2937 0%, #111827 100%); 
                                    border-radius: 10px; padding: 15px; text-align: center; 
                                    border: 1px solid #374151; margin-bottom: 10px;">
                            <div style="font-size: 0.8rem; font-weight: bold; margin-bottom: 8px;">{status_display}</div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <img src="{away_logo}" width="35" height="35" style="vertical-align: middle;" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['away_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{away_record} {away_seed}</div>
                                    </div>
                                </div>
                                <div>{away_score_display}</div>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <img src="{home_logo}" width="35" height="35" style="vertical-align: middle;" onerror="this.style.display='none'"/>
                                    <div style="text-align: left;">
                                        <div style="font-weight: bold; color: #FAFAFA;">{game['home_team']}</div>
                                        <div style="color: #9CA3AF; font-size: 0.75rem;">{home_record} {home_seed}</div>
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

        # Show player photo and team logo centered
        spacer1, photo_col, logo_col, spacer2 = st.columns([1.5, 0.5, 0.5, 1.5])
        with photo_col:
            player_photo = get_player_photo_url(selected_player)
            if player_photo:
                st.image(player_photo, width=100)
        with logo_col:
            team_logo = get_team_logo_url(player_team)
            if team_logo:
                st.image(team_logo, width=80)
        
        # Player name and info centered
        st.markdown(f"<h3 style='text-align: center;'>{selected_player}</h3>", unsafe_allow_html=True)
        
        bio_info = ""
        if bio:
            height = bio.get('height', '')
            weight = bio.get('weight', '')
            draft = bio.get('draft_year', '')
        bio_info = ""
        if bio:
            height = bio.get('height', '')
            weight = bio.get('weight', '')
            age = bio.get('age', '')
            draft = bio.get('draft_year', '')
            draft_round = bio.get('draft_round', '')
            draft_num = bio.get('draft_number', '')
            
            parts = []
            if height and weight:
                parts.append(f"{height}, {weight} lbs")
            if age and age != 'N/A':
                parts.append(f"Age: {age}")
            if draft and draft != 'Undrafted':
                draft_str = f"Drafted {draft}"
                if draft_round and draft_num and draft_round != 'Undrafted':
                     draft_str += f" (R{draft_round}, #{draft_num})"
                parts.append(draft_str)
            
            bio_info = " • ".join(parts)
            if bio_info:
                bio_info += " • "
        
        st.markdown(f"<p style='text-align: center; color: #9CA3AF;'>{bio_info}{player_team} • {len(player_df)} games loaded</p>", unsafe_allow_html=True)
        
        # Get team seed for the button
        team_seed_suffix = ""
        if standings_df is not None:
            for _, row in standings_df.iterrows():
                if get_team_abbrev(row['TeamCity']) == player_team:
                    team_seed_suffix = f" (#{int(row['PlayoffRank'])})"
                    break
        
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
                if st.button(f"👀 Watch {player_team}{team_seed_suffix}", use_container_width=True):
                    if auth.add_favorite_team(player_team):
                        st.toast(f"Added {player_team} to watched teams!")
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
            'FG': f"{fg_pct:.1f}%" if fg_pct != "N/A" else "N/A",
            '3P': f"{three_pct:.1f}%" if three_pct != "N/A" else "N/A",
            'FT': f"{ft_pct:.1f}%" if ft_pct != "N/A" else "N/A",
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
                    opp_rank = game['opponent_rank']
                    if opp_rank:
                        label = f"{game['date']}\n{home_away} #{opp_rank} {game['opponent']}"
                    else:
                        label = f"{game['date']}\n{home_away} {game['opponent']}"
                    
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
                    help=f"{selected_player} currently plays for {player_team} (excluded from list)"
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
                st.info(f"**{selected_player}** has not played against **{selected_opponent}** yet this season.")
        
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
                        st.markdown(f"### Predicted Stats: {selected_player} vs {selected_opponent}")
                        
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
                        fig.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#0E1117')
                        
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
    
    if player_search:
        matching_players = search_players(player_search, season)
        if matching_players:
            selected_player = st.selectbox("Select player:", matching_players, key="stats_select")
            
            if st.button("Load Stats", type="primary"):
                with st.spinner(f"Loading {selected_player}'s stats..."):
                    player_df, player_team = get_player_game_log(selected_player, season)
                
                if player_df is None or len(player_df) == 0:
                    st.error(f"No data found for {selected_player} in {season}")
                else:
                    # Get player bio first for better branding
                    bio = fetch_player_bio(selected_player)
                    if bio and bio.get('team_abbrev'):
                        player_team = bio['team_abbrev']

                    # Show player photo and team logo centered
                    spacer1, photo_col, logo_col, spacer2 = st.columns([1.5, 0.5, 0.5, 1.5])
                    with photo_col:
                        player_photo = get_player_photo_url(selected_player)
                        if player_photo:
                            st.image(player_photo, width=100)
                    with logo_col:
                        team_logo = get_team_logo_url(player_team)
                        if team_logo:
                            st.image(team_logo, width=80)
                    
                    # Player name and info centered
                    st.markdown(f"<h3 style='text-align: center;'>{selected_player}</h3>", unsafe_allow_html=True)
                    
                    bio_info = ""
                    if bio:
                        height = bio.get('height', '')
                        weight = bio.get('weight', '')
                        draft = bio.get('draft_year', '')
                    bio_info = ""
                    if bio:
                        height = bio.get('height', '')
                        weight = bio.get('weight', '')
                        age = bio.get('age', '')
                        draft = bio.get('draft_year', '')
                        draft_round = bio.get('draft_round', '')
                        draft_num = bio.get('draft_number', '')
                        
                        parts = []
                        if height and weight:
                            parts.append(f"{height}, {weight} lbs")
                        if age and age != 'N/A':
                            parts.append(f"Age: {age}")
                        if draft and draft != 'Undrafted':
                            draft_str = f"Drafted {draft}"
                            if draft_round and draft_num and draft_round != 'Undrafted':
                                draft_str += f" (R{draft_round}, #{draft_num})"
                            parts.append(draft_str)
                        
                        bio_info = " • ".join(parts)
                        if bio_info:
                            bio_info += " • "
                    
                    st.markdown(f"<p style='text-align: center; color: #9CA3AF;'>{bio_info}{player_team} • {len(player_df)} games loaded</p>", unsafe_allow_html=True)
                    
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
                        st.metric("FG%", f"{fg_pct:.1f}%")
                        st.metric("3P%", f"{three_pct:.1f}%")
                    
                    with col5:
                        st.metric("FT%", f"{ft_pct:.1f}%")
                        st.metric("Games", len(player_df))
                    
                    with col6:
                        # Calculate TS%
                        total_pts = player_df['Points'].sum()
                        ts_pct = (total_pts / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                        st.metric("TS%", f"{ts_pct:.1f}%")
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
                                   'Steals', 'Blocks', 'Turnovers', 'PF', 'FG', '3P', 'FT', 'TS%']
                    available_cols = [col for col in display_cols if col in player_df.columns]
                    
                    display_df = player_df[available_cols].iloc[::-1].copy()
                    
                    if 'TS%' in display_df.columns:
                        display_df['TS%'] = display_df['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                    
                    # Format PF as integer
                    if 'PF' in display_df.columns:
                        display_df['PF'] = display_df['PF'].astype(int)
                    
                    def style_wl(val):
                        if val == 'W':
                            return 'color: #10B981; font-weight: bold'
                        elif val == 'L':
                            return 'color: #EF4444; font-weight: bold'
                        return ''
                    
                    styled_display_df = display_df.style.applymap(style_wl, subset=['W/L']) if 'W/L' in display_df.columns else display_df
                    
                    st.dataframe(
                        styled_display_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Score": st.column_config.TextColumn("Score", width="small")
                        }
                    )
                    
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
        
        tab1, tab2 = st.tabs(["Favorite Players", "Watched Teams"])
        
        with tab1:
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
                        col_img, col_info, col_actions = st.columns([0.8, 2.5, 1])
                        
                        with col_img:
                            if bio and bio.get('player_id'):
                                headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{bio['player_id']}.png"
                                st.image(headshot_url, width=100)
                            else:
                                st.write("👤")
                        
                        with col_info:
                            # Team Logo + Name
                            team_abbrev = bio.get('team_abbrev') if bio else None
                            
                            title_col1, title_col2 = st.columns([0.15, 0.85])
                            with title_col1:
                                if team_abbrev:
                                    logo = get_team_logo_url(team_abbrev)
                                    if logo:
                                        st.image(logo, width=40)
                                    else:
                                        st.write(team_abbrev)
                            with title_col2:
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
                                
                                # Display columns matching Live Predictions
                                display_cols = ['GAME_DATE', 'MATCHUP', 'W/L', 'Score', 'MIN', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PF']
                                available = [c for c in display_cols if c in recent.columns]
                                recent_display = recent[available].copy()
                                
                                # Rename columns
                                rename_map = {
                                    'GAME_DATE': 'Date', 'MATCHUP': 'Matchup', 
                                    'Points': 'PTS', 'Assists': 'AST', 'Rebounds': 'REB',
                                    'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TO'
                                }
                                recent_display = recent_display.rename(columns=rename_map)
                                
                                # Style W/L
                                def color_wl(val):
                                    if val == 'W': return 'color: #10B981; font-weight: bold'
                                    elif val == 'L': return 'color: #EF4444; font-weight: bold'
                                    return ''
                                
                                if 'W/L' in recent_display.columns:
                                    styled = recent_display.style.applymap(color_wl, subset=['W/L'])
                                else:
                                    styled = recent_display
                                    
                                st.dataframe(
                                    styled, 
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
                                            opp_rank = game['opponent_rank']
                                            if opp_rank:
                                                label = f"{game['date']}\n{home_away} #{opp_rank} {game['opponent']}"
                                            else:
                                                label = f"{game['date']}\n{home_away} {game['opponent']}"
                                            
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
        
        with tab2:
            # Header with add button
            header_col, add_col = st.columns([4, 1])
            with header_col:
                st.markdown("### Your Watched Teams")
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
                        new_team = st.selectbox("Select team to add:", available_teams, key="new_team_select")
                        add_col1, add_col2 = st.columns(2)
                        with add_col1:
                            if st.button("Add Team", use_container_width=True, type="primary", key="confirm_add_team"):
                                if auth.add_favorite_team(new_team):
                                    st.toast(f"Added {new_team} to watched teams!")
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
                st.write(f"You have **{len(favorite_teams)}** watched team(s):")
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
                    "LAC": {"name": "Clippers", "city": "LA"},
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
                        # Match by city name
                        matching = standings_df[standings_df['TeamCity'].str.contains(team_city, case=False, na=False)]
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
                    logo_url = get_team_logo_url(team)
                    
                    # Show team logo and card side by side
                    logo_col, card_col = st.columns([0.15, 0.85])
                    
                    with logo_col:
                        if logo_url:
                            st.image(logo_url, width=60)
                    
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
                                    <div style="color: #9CA3AF; font-size: 0.9rem;">Record: <strong style="color: #FAFAFA;">{record}</strong> | {conf}ern Conference #{rank}</div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">OFF RTG</div>
                                    <div style="color: #10B981; font-weight: 600;">{off_rtg} <span style="color: #6B7280; font-size: 0.8rem;">(#{off_rank})</span></div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">DEF RTG</div>
                                    <div style="color: #3B82F6; font-weight: 600;">{def_rtg} <span style="color: #6B7280; font-size: 0.8rem;">(#{def_rank})</span></div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">HOME</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{home}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">ROAD</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{road}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">L10</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{l10}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75rem;">STREAK</div>
                                    <div style="color: #FAFAFA; font-weight: 600;">{streak}</div>
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
                                if opp_rank:
                                    opp_display = f"#{opp_rank} {game['opponent_name']}"
                                else:
                                    opp_display = game['opponent_name']
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
                                        # Navigate to Live Predictions page with this player
                                        st.session_state['auto_load_player'] = player_name
                                        st.session_state['pending_nav_target'] = "Live Predictions"
                                        st.rerun()
                        else:
                            st.info("Could not load roster.")
                    
                    # Remove button
                    if st.button("Remove", key=f"team_remove_{team}", use_container_width=True):
                        auth.remove_favorite_team(team)
                        st.toast(f"Removed {team} from watched teams")
                        st.rerun()
                    
                    st.markdown("---")
            else:
                render_empty_state("No watched teams yet! Add teams from the Live Predictions page.", "")

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
    if selected_player1 and selected_player2:
        if st.button("🔍 Compare Players", type="primary", use_container_width=True):
            with st.spinner("Loading player stats..."):
                # Get player data for both
                player1_df, player1_team = get_player_game_log(selected_player1, season)
                player2_df, player2_team = get_player_game_log(selected_player2, season)
                
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
                    
                    # Player profile cards
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Player 1 card
                        p1_photo = get_player_photo_url(player1_name)
                        p1_logo = get_team_logo_url(player1_team)
                        
                        # Build bio line (no draft info)
                        bio1_parts = []
                        if bio1:
                            if bio1.get('height') and bio1.get('weight'):
                                bio1_parts.append(f"{bio1['height']}, {bio1['weight']} lbs")
                            if bio1.get('age') and bio1['age'] != 'N/A':
                                bio1_parts.append(f"Age: {bio1['age']}")
                        bio1_parts.append(player1_team)
                        bio1_parts.append(f"{len(player1_df)} games loaded")
                        bio1_line = " • ".join(bio1_parts)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: #1F2937; border-radius: 10px;">
                            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 10px;">
                                <img src="{p1_photo}" width="80" onerror="this.style.display='none'"/>
                                <img src="{p1_logo}" width="60" onerror="this.style.display='none'"/>
                            </div>
                            <h2 style="color: #FAFAFA; margin: 10px 0;">{player1_name}</h2>
                            <p style="color: #9CA3AF; margin: 5px 0; font-size: 0.9rem;">{bio1_line}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Player 2 card
                        p2_photo = get_player_photo_url(player2_name)
                        p2_logo = get_team_logo_url(player2_team)
                        
                        # Build bio line (no draft info)
                        bio2_parts = []
                        if bio2:
                            if bio2.get('height') and bio2.get('weight'):
                                bio2_parts.append(f"{bio2['height']}, {bio2['weight']} lbs")
                            if bio2.get('age') and bio2['age'] != 'N/A':
                                bio2_parts.append(f"Age: {bio2['age']}")
                        bio2_parts.append(player2_team)
                        bio2_parts.append(f"{len(player2_df)} games loaded")
                        bio2_line = " • ".join(bio2_parts)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: #1F2937; border-radius: 10px;">
                            <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 10px;">
                                <img src="{p2_photo}" width="80" onerror="this.style.display='none'"/>
                                <img src="{p2_logo}" width="60" onerror="this.style.display='none'"/>
                            </div>
                            <h2 style="color: #FAFAFA; margin: 10px 0;">{player2_name}</h2>
                            <p style="color: #9CA3AF; margin: 5px 0; font-size: 0.9rem;">{bio2_line}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                    for stat in ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TPG', 'FG%', '3P%', 'FT%', 'TS%', 'MPG', 'Games']:
                        v1 = p1_stats.get(stat, 0)
                        v2 = p2_stats.get(stat, 0)
                        
                        # Determine winner
                        if stat in higher_is_better:
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
                            value = f"{item['p1_value']}%" if '%' in item['Stat'] else item['p1_value']
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
                            value = f"{item['p2_value']}%" if '%' in item['Stat'] else item['p2_value']
                            st.markdown(f"""
                            <div style="text-align: left; padding: 8px; font-size: 1.2rem; color: {p2_color}; font-weight: {'bold' if item['p2_better'] else 'normal'};">
                                {value}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    
    else:
        st.info("👆 Enter two player names above to compare their stats")


# ==================== STANDINGS PAGE ====================
elif page == "Standings":
    from datetime import datetime
    
    st.title("NBA Standings")
    
    # Display current date
    current_date = datetime.now().strftime("%B %d, %Y")
    st.markdown(f"**As of: {current_date}**")
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
            """Display standings for a conference with favorite team highlighting."""
            # Sort by playoff rank
            conference_df = conference_df.sort_values('PlayoffRank')
            
            for idx, row in conference_df.iterrows():
                team_abbrev = get_team_abbrev(row['TeamCity'])
                is_favorite = team_abbrev in favorite_teams
                
                rank = int(row['PlayoffRank'])
                team_name = f"{row['TeamCity']} {row['TeamName']}"
                record = row['Record']
                win_pct = f"{row['WinPct']:.3f}" if isinstance(row['WinPct'], float) else row['WinPct']
                l10 = row.get('L10', 'N/A')
                streak = row.get('strCurrentStreak', 'N/A')
                gb = row.get('GB', '-')
                home = row.get('HOME', 'N/A')
                road = row.get('ROAD', 'N/A')
                conf_rec = row.get('ConferenceRecord', 'N/A')
                div_rec = row.get('DivisionRecord', 'N/A')
                logo_url = get_team_logo_url(team_abbrev)
                
                # Get team ratings
                team_rtg = team_ratings.get(team_abbrev, {})
                off_rtg = team_rtg.get('off_rtg', 'N/A')
                def_rtg = team_rtg.get('def_rtg', 'N/A')
                off_rank = team_rtg.get('off_rank', 'N/A')
                def_rank = team_rtg.get('def_rank', 'N/A')
                
                # Play-in indicator - mark boundary before 7 and after 10
                if rank == 7:
                    st.markdown("""
                    <div style="border-top: 2px dashed #FF6B35; margin: 10px 0; padding-top: 5px;">
                        <span style="color: #FF6B35; font-size: 0.8rem; font-weight: bold;">PLAY-IN TOURNAMENT</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Use Streamlit columns for layout with CONF and DIV records
                col_logo, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns([0.35, 0.2, 1.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.45, 0.45, 0.4, 0.4, 0.6, 0.6])
                
                with col_logo:
                    if logo_url:
                        st.image(logo_url, width=35)
                
                with col1:
                    if is_favorite:
                        st.markdown(f"**:orange[{rank}]**")
                    else:
                        st.markdown(f"**{rank}**")
                
                with col2:
                    if is_favorite:
                        st.markdown(f"**:orange[{team_name}]** _(Fav)_")
                    else:
                        st.markdown(f"**{team_name}**")
                
                with col3:
                    st.caption("RECORD")
                    st.write(record)
                
                with col4:
                    st.caption("GB")
                    st.write(gb if gb != 0 else "-")
                
                with col5:
                    st.caption("PCT")
                    st.write(win_pct)
                
                with col6:
                    st.caption("HOME")
                    st.write(home)
                
                with col7:
                    st.caption("ROAD")
                    st.write(road)
                
                with col8:
                    st.caption("CONF")
                    st.write(conf_rec)
                
                with col9:
                    st.caption("DIV")
                    st.write(div_rec)
                
                with col10:
                    st.caption("L10")
                    st.write(l10)
                
                with col11:
                    st.caption("STREAK")
                    st.write(streak)
                
                with col12:
                    st.caption("OFF RTG")
                    if off_rtg != 'N/A':
                        st.write(f"{off_rtg} (#{off_rank})")
                    else:
                        st.write("N/A")
                
                with col13:
                    st.caption("DEF RTG")
                    if def_rtg != 'N/A':
                        st.write(f"{def_rtg} (#{def_rank})")
                    else:
                        st.write("N/A")
                
                # End of play-in indicator after rank 10 and mark lottery teams
                if rank == 10:
                    st.markdown("""
                    <div style="border-bottom: 2px dashed #FF6B35; margin-top: 5px; padding-bottom: 10px;"></div>
                    <div style="margin: 10px 0; padding-top: 5px;">
                        <span style="color: #EF4444; font-size: 0.8rem; font-weight: bold;">OUT OF PLAYOFF RACE (LOTTERY)</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.divider()
        
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
            st.markdown("### Playoff Picture (If Season Ended Today)")
            
            def get_team_by_seed(conference_df, seed):
                """Get team info by playoff seed."""
                team_row = conference_df[conference_df['PlayoffRank'] == seed]
                if not team_row.empty:
                    row = team_row.iloc[0]
                    abbrev = get_team_abbrev(row['TeamCity'])
                    name = f"{row['TeamCity']} {row['TeamName']}"
                    record = row['Record']
                    return {'abbrev': abbrev, 'name': name, 'record': record, 'seed': seed}
                return None
            
            def display_matchup(team1, team2, matchup_type="vs"):
                """Display a single matchup with logos."""
                if team1 and team2:
                    col1, col2, col3, col4, col5 = st.columns([0.3, 2, 0.5, 2, 0.3])
                    
                    logo1 = get_team_logo_url(team1['abbrev'])
                    logo2 = get_team_logo_url(team2['abbrev'])
                    
                    with col1:
                        if logo1:
                            st.image(logo1, width=40)
                    with col2:
                        st.markdown(f"**({team1['seed']}) {team1['name']}**")
                        st.caption(team1['record'])
                    with col3:
                        st.markdown(f"**{matchup_type}**")
                    with col4:
                        st.markdown(f"**({team2['seed']}) {team2['name']}**")
                        st.caption(team2['record'])
                    with col5:
                        if logo2:
                            st.image(logo2, width=40)
            
            west_df = standings_df[standings_df['Conference'] == 'West']
            east_df = standings_df[standings_df['Conference'] == 'East']
            
            # Western Conference Matchups
            st.markdown("---")
            st.markdown('<img src="./app/static/western_conference.png" width="30" style="vertical-align: middle;"/> **Western Conference**', unsafe_allow_html=True)
            
            st.markdown("**Play-In Tournament**")
            w7 = get_team_by_seed(west_df, 7)
            w8 = get_team_by_seed(west_df, 8)
            w9 = get_team_by_seed(west_df, 9)
            w10 = get_team_by_seed(west_df, 10)
            
            display_matchup(w8, w7, "@")
            display_matchup(w10, w9, "@")
            
            st.markdown("**First Round**")
            st.caption("_2 seed plays winner of (7 vs 8). 1 seed plays winner of (winner 9 vs 10) @ (loser 7 vs 8)._")
            w1 = get_team_by_seed(west_df, 1)
            w2 = get_team_by_seed(west_df, 2)
            w3 = get_team_by_seed(west_df, 3)
            w4 = get_team_by_seed(west_df, 4)
            w5 = get_team_by_seed(west_df, 5)
            w6 = get_team_by_seed(west_df, 6)
            
            display_matchup(w8, w1, "@")  # 8 @ 1
            display_matchup(w5, w4, "@")  # 5 @ 4
            st.markdown("---")
            display_matchup(w7, w2, "@")  # 7 @ 2
            display_matchup(w6, w3, "@")  # 6 @ 3
            
            # Eastern Conference Matchups
            st.markdown("---")
            st.markdown('<img src="./app/static/eastern_conference.png" width="30" style="vertical-align: middle;"/> **Eastern Conference**', unsafe_allow_html=True)
            
            st.markdown("**Play-In Tournament**")
            e7 = get_team_by_seed(east_df, 7)
            e8 = get_team_by_seed(east_df, 8)
            e9 = get_team_by_seed(east_df, 9)
            e10 = get_team_by_seed(east_df, 10)
            
            display_matchup(e8, e7, "@")
            display_matchup(e10, e9, "@")
            
            st.markdown("**First Round**")
            st.caption("_2 seed plays winner of (7 vs 8). 1 seed plays winner of (winner 9 vs 10) @ (loser 7 vs 8)._")
            e1 = get_team_by_seed(east_df, 1)
            e2 = get_team_by_seed(east_df, 2)
            e3 = get_team_by_seed(east_df, 3)
            e4 = get_team_by_seed(east_df, 4)
            e5 = get_team_by_seed(east_df, 5)
            e6 = get_team_by_seed(east_df, 6)
            
            display_matchup(e8, e1, "@")  # 8 @ 1
            display_matchup(e5, e4, "@")  # 5 @ 4
            st.markdown("---")
            display_matchup(e7, e2, "@")  # 7 @ 2
            display_matchup(e6, e3, "@")  # 6 @ 3
        
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
                
                # Header row - added DIV REC column
                header_logo, header1, header2, header3, header4, header5, header6 = st.columns([0.5, 0.3, 2.0, 0.7, 0.7, 0.7, 0.7])
                with header1:
                    st.caption("#")
                with header2:
                    st.caption("TEAM")
                with header3:
                    st.caption("RECORD")
                with header4:
                    st.caption("DIV REC")
                with header5:
                    st.caption("SEED")
                with header6:
                    st.caption("PCT")
                st.markdown("<div style='margin-bottom: -25px'></div>", unsafe_allow_html=True)
                for idx, (_, row) in enumerate(division_df.iterrows(), 1):
                    team_abbrev = get_team_abbrev(row['TeamCity'])
                    is_favorite = team_abbrev in favorite_teams
                    
                    team_name = f"{row['TeamCity']} {row['TeamName']}"
                    record = row['Record']
                    div_rec = row.get('DivisionRecord', 'N/A')
                    win_pct = f"{row['WinPct']:.3f}" if isinstance(row['WinPct'], float) else row['WinPct']
                    conf_seed = int(row['PlayoffRank']) if 'PlayoffRank' in row else 'N/A'
                    logo_url = get_team_logo_url(team_abbrev)
                    
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
                        if is_favorite:
                            st.markdown(f"**:orange[{team_name}]**")
                        else:
                            st.markdown(f"**{team_name}**")
                    
                    with col3:
                        st.write(record)
                    
                    with col4:
                        st.write(div_rec)
                    
                    with col5:
                        st.write(f"#{conf_seed}")
                    
                    with col6:
                        st.write(win_pct)
            
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



# ==================== ABOUT PAGE ====================
elif page == "About":
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

