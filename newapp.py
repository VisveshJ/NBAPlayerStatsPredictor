import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import time
import warnings
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Live Stats Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DARK MODE CSS ====================

st.markdown("""
<style>
/* ===== GLOBAL APP BACKGROUND ===== */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
/* ===== TEXT ===== */
html, body, [class*="css"] {
    color: #FAFAFA !important;
}
/* ===== HEADERS ===== */
h1, h2, h3, h4, h5, h6 {
    color: #FAFAFA;
}
/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stSidebar"] * {
    color: #FAFAFA;
}
/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background-color: #1F2937;
    padding: 16px;
    border-radius: 10px;
    color: #FAFAFA;
}
[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}
[data-testid="stMetricValue"] {
    color: #FAFAFA;
    font-size: 1.6rem;
}
/* ===== DATAFRAMES ===== */
.stDataFrame {
    background-color: #1F2937;
}
.stDataFrame td, .stDataFrame th {
    color: #FAFAFA !important;
    background-color: #1F2937 !important;
}
/* ===== INPUTS ===== */
input, textarea, select {
    background-color: #1F2937 !important;
    color: #FAFAFA !important;
}
/* ===== BUTTONS ===== */
.stButton > button {
    background-color: #FF6B35;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #FF8C5A;
}
/* ===== SUCCESS / INFO / WARNING ===== */
.stAlert {
    background-color: #1F2937;
    color: #FAFAFA;
}
/* ===== PLOTS ===== */
svg, text {
    fill: #FAFAFA !important;
}
</style>
""", unsafe_allow_html=True)

import json
import os

# ==================== USER PREFERENCES FUNCTIONS ====================
PREFS_FILE = "user_preferences.json"

def load_preferences():
    """Load saved players and teams from a JSON file."""
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    return {"fav_players": [], "fav_teams": []}

def save_preferences(prefs):
    """Save players and teams to a JSON file."""
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f)

def add_to_favorites(item, category):
    """Add a player or team to the preferences file."""
    prefs = load_preferences()
    if item not in prefs[category]:
        prefs[category].append(item)
        save_preferences(prefs)
        return True
    return False

def remove_from_favorites(item, category):
    """Remove a player or team from the preferences file."""
    prefs = load_preferences()
    if item in prefs[category]:
        prefs[category].remove(item)
        save_preferences(prefs)
        return True
    return False

# ==================== DATA FETCHING FUNCTIONS ====================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_current_defensive_ratings(season="2025-26"):
    """Fetch current defensive ratings for all NBA teams."""
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame"
        )
        
        df = team_stats.get_data_frames()[0]
        all_teams = teams.get_teams()
        valid_nba_teams = {t['abbreviation'] for t in all_teams}
        
        # Create multiple mapping strategies
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
                
                # Try direct abbreviation first
                if team_value in valid_nba_teams:
                    team_abbrev = team_value
                else:
                    # Try mapping from full name or nickname
                    team_abbrev = team_name_to_abbrev.get(team_value)
                
                if team_abbrev and team_abbrev in valid_nba_teams:
                    team_def_ratings[team_abbrev] = round(row['DEF_RATING'], 1)
        
        # Ensure all 30 teams are present - fill missing teams with league average
        if len(team_def_ratings) < 30:
            league_avg = np.mean(list(team_def_ratings.values())) if team_def_ratings else 112.0
            for team in valid_nba_teams:
                if team not in team_def_ratings:
                    team_def_ratings[team] = round(league_avg, 1)
        
        return team_def_ratings
    except Exception as e:
        st.error(f"Error fetching defensive ratings: {str(e)}")
        return {}

# Update the get_player_game_log function to include FT stats and format combined columns
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_player_game_log(player_name, season="2025-26"):
    """Fetch game log for a player from the current season. Returns (df, team_abbrev)"""
    all_players = players.get_players()
    player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    
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
        
        # Extract player's team from most recent game (first row before reversing)
        player_team = df['MATCHUP'].iloc[0][:3]
        
        df['Opponent'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
        df = df.rename(columns={
            'PTS': 'Points',
            'AST': 'Assists',
            'REB': 'Rebounds',
            'STL': 'Steals',
            'BLK': 'Blocks',
            'TOV': 'Turnovers',
            'FGM': 'FGM',
            'FGA': 'FGA',
            'FG_PCT': 'FG%',
            'FG3M': '3PM',
            'FG3A': '3PA',
            'FG3_PCT': '3P%',
            'FTM': 'FTM',
            'FTA': 'FTA',
            'FT_PCT': 'FT%',
            'MIN': 'MIN', 
            'PF': 'PF'
        })
        
        # Calculate True Shooting Percentage (TS%)
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
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

@st.cache_data(ttl=86400)  # Cache for 24 hours
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
    except Exception as e:
        return None


def search_players(query, season="2025-26"):
    """Search for players by name - only returns active players with games this season."""
    
    # Try to get verified active players list
    active_players_list = get_active_players_list(season)
    
    if active_players_list:
        if not query:
            return active_players_list[:30]
        
        query_lower = query.lower()
        matches = [name for name in active_players_list 
                   if query_lower in name.lower()]
        return matches[:20]
    else:
        # Fallback
        all_players = players.get_players()
        active_players = [p for p in all_players if p.get('is_active', True)]
        
        query_lower = query.lower()
        matches = [p['full_name'] for p in active_players 
                   if query_lower in p['full_name'].lower()]
        
        return matches[:20]


# ==================== MODEL FUNCTIONS ====================

def ensure_minimum_transitions(model, min_prob=0.01):
    """Ensure all states have minimum transition probability."""
    n_states = model.n_components
    transmat = model.transmat_.copy()
    
    # Apply minimum probability
    transmat[transmat < min_prob] = min_prob
    
    # Renormalize each row
    for i in range(n_states):
        transmat[i] = transmat[i] / transmat[i].sum()
    
    model.transmat_ = transmat
    return model

def calculate_player_consistency(player_df, stat_cols):
    """Calculate player consistency score (lower = more consistent)."""
    cv_scores = []
    for stat in stat_cols:
        if player_df[stat].mean() > 0:
            cv = player_df[stat].std() / player_df[stat].mean()
            cv_scores.append(cv)
    
    return np.mean(cv_scores) if cv_scores else 1.0

def train_hmm_with_drtg(player_df, team_def_ratings, n_states=3, use_temporal_weighting=True, weight_strength='medium',min_transition_prob=0.01):
    """Train HMM with opponent defense as a feature."""
    player_df = player_df.copy()
    
    # Clean data
    player_df['Opponent_DEF_RTG'] = player_df['Opponent'].map(team_def_ratings)
    player_df = player_df.dropna(subset=['Opponent_DEF_RTG'])
    
    stat_cols = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers']
    player_df = player_df.dropna(subset=stat_cols)
    
    feature_cols = stat_cols + ['Opponent_DEF_RTG']
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(player_df[feature_cols].values)
    
    # Train HMM with better parameters
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=1000,  # More iterations
        random_state=100,
        verbose=True,
        init_params='ste',  # Initialize with stochastic method
        tol=0.001,  # Lower tolerance for convergence
        min_covar=0.001  # Minimum covariance to avoid singular matrices
    )
    model.fit(X)

    if min_transition_prob > 0:
        model = ensure_minimum_transitions(model, min_prob=min_transition_prob)
    
    return model, stat_cols, scaler, player_df  # Return the full player_df

def predict_with_drtg(model, stat_cols, scaler, recent_df, team_def_ratings, target_opponent, full_player_df=None):
    """Generate prediction with defensive rating AND head-to-head performance."""
    target_drtg = team_def_ratings.get(target_opponent)
    if target_drtg is None:
        print(f"⚠️ No DRTG found for opponent: {target_opponent}")
        return None
    
    # Get last 3-5 games for sequence prediction
    last_games = recent_df.tail(5)
    if len(last_games) < 2:
        print(f"⚠️ Not enough games for sequence: {len(last_games)}")
        return None
    
    # Prepare features
    feature_cols = stat_cols + ['Opponent_DEF_RTG']
    
    # Check all columns exist
    for col in feature_cols:
        if col not in last_games.columns:
            print(f"⚠️ Missing column: {col}")
            return None
    
    # Get the sequence of recent games
    X_sequence = last_games[feature_cols].values  # This is 2D: (n_games, n_features)
    
    # Scale the sequence
    X_sequence_scaled = scaler.transform(X_sequence)  # Should be 2D
    
    # Use Viterbi algorithm to find the most likely hidden state sequence
    try:
        log_prob, state_sequence = model.decode(X_sequence_scaled, algorithm="viterbi")
        last_state = state_sequence[-1]
        
        # Predict next state using transition matrix
        next_state_probs = model.transmat_[last_state]
        next_state = np.argmax(next_state_probs)
        
        print(f"📊 Transition probabilities from state {last_state}: {next_state_probs}")
        print(f"📊 Predicted next state: {next_state}")
        
        # Get the mean emission for the predicted next state
        next_state_mean = model.means_[next_state]  # This is 1D: (n_features,)
        
        # Adjust for defensive rating
        avg_drtg = np.mean(list(team_def_ratings.values()))
        defensive_factor =  target_drtg / avg_drtg
        
        print(f"📊 Defensive factor: {defensive_factor:.3f} (Avg DRTG: {avg_drtg:.1f} / Opp DRTG: {target_drtg})")
        
        # Create prediction with DRTG adjustment
        adjusted_prediction = next_state_mean.copy()
        
        # Set DRTG component to scaled target value
        # Create a dummy array with the target DRTG
        dummy_features = np.zeros((1, len(feature_cols)))  # 2D: (1, n_features)
        dummy_features[0, -1] = target_drtg
        scaled_drtg = scaler.transform(dummy_features)[0, -1]
        adjusted_prediction[-1] = scaled_drtg
        
        # Inverse transform to get actual values
        # Reshape to 2D for inverse_transform
        adjusted_prediction_2d = adjusted_prediction.reshape(1, -1)
        predicted_unscaled = scaler.inverse_transform(adjusted_prediction_2d)[0]
        
        print(f"📊 Raw prediction before DRTG adjustment: {predicted_unscaled[:len(stat_cols)]}")
        
        # Apply defensive factor to relevant stats
        result = {}
        defensive_sensitive_stats = ['Points', 'Assists', 'Turnovers']
        
        for i, stat in enumerate(stat_cols):
            value = predicted_unscaled[i]
            if stat in defensive_sensitive_stats:
                old_value = value
                value *= defensive_factor
                print(f"📊 {stat}: {old_value:.1f} → {value:.1f} (x{defensive_factor:.3f})")
            
            # Apply realistic bounds
            if stat == 'Points':
                value = max(0, min(value, 60))
            elif stat == 'Assists':
                value = max(0, min(value, 20))
            elif stat == 'Rebounds':
                value = max(0, min(value, 25))
            elif stat == 'Steals':
                value = max(0, min(value, 8))
            elif stat == 'Blocks':
                value = max(0, min(value, 8))
            elif stat == 'Turnovers':
                value = max(0, min(value, 10))
            
            result[stat] = round(value, 1)
        
        return result
        
    except Exception as e:
        print(f"❌ Viterbi algorithm error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== SIDEBAR ====================

# ==================== SIDEBAR ====================

st.sidebar.title("NBA Stats Predictor")
st.sidebar.markdown("---")

# Fixed season for current season
season = "2025-26"

# Now define the navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Live Predictions", "Player Stats", "Favorites", "About"],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tip:** Predictions update automatically with live NBA data!"
)

# st.sidebar.title("🏀 NBA Stats Predictor")
# st.sidebar.markdown("---")

# page = st.sidebar.radio(
#     "Navigation",
#     ["🏠 Home", "🔮 Live Predictions", "📊 Player Stats", "ℹ️ About"],
#     label_visibility="visible"
# )

# st.sidebar.markdown("---")
# st.sidebar.markdown("### ⚙️ Settings")
# season = st.sidebar.selectbox("Season", ["2025-26", "2024-25"], index=0)

# st.sidebar.markdown("---")
# st.sidebar.info(
#     "💡 **Tip:** Predictions update automatically with live NBA data!"
# )

# ==================== HOME PAGE ====================

if page == "Home":
    st.markdown('<p style="font-size: 3rem; font-weight: bold; text-align: center; color: #FF6B35;">NBA Live Stats Predictor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the NBA Live Stats Prediction System!
    
    This advanced analytics tool uses **Hidden Markov Models (HMM)** to predict player performance 
    based on real-time NBA data.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Live Data")
        st.write("Fetches current season stats directly from NBA.com API")
    
    with col2:
        st.markdown("#### AI-Powered")
        st.write("Uses machine learning to predict player performance")
    
    with col3:
        st.markdown("#### Accurate")
        st.write("Factors in opponent defense and recent trends")
    
    st.markdown("---")
    
    st.markdown("### Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
         **Real-time Data**
        - Live defensive ratings
        - Current season game logs
        - Up-to-date player stats
        
         **Smart Predictions**
        - Hidden Markov Models
        - Opponent-aware forecasting
        - Recent performance weighting
        """)
    
    with feature_col2:
        st.markdown("""
        **Comprehensive Stats**
        - Points, Assists, Rebounds
        - Steals, Blocks, Turnovers
        - Defensive adjustments
        
        **Easy to Use**
        - Search any NBA player
        - Select upcoming opponent
        - Get instant predictions
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Live Predictions** and start forecasting!")

# ==================== LIVE PREDICTIONS PAGE ====================

elif page == "Live Predictions":
    st.title("Live Player Performance Predictions")
    st.markdown("Predict any NBA player's next game stats using real-time data!")
    
    # Fetch defensive ratings
    with st.spinner("Fetching latest defensive ratings..."):
        team_def_ratings = get_current_defensive_ratings(season)
    
    if not team_def_ratings:
        st.error("Could not fetch defensive ratings. Please try again later.")
        st.stop()
    
    #st.success(f"✅ Loaded defensive ratings for {len(team_def_ratings)} teams")
    
    # Initialize session state for player data
    if 'player_data' not in st.session_state:
        st.session_state.player_data = None
    if 'player_team' not in st.session_state:
        st.session_state.player_team = None
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    
    # Player search
    st.markdown("**Select Player**") 
    st.caption("Only showing players active in current season")
    
    # Function to clear player data when search changes
    def clear_player_data():
        st.session_state.player_data = None
        st.session_state.player_team = None
        st.session_state.selected_player = None
    
    # Create a text input that clears data when typing starts
    player_search = st.text_input(
        "Search player name:", 
        placeholder="e.g., LeBron James", 
        key="player_search",
        value=""
    )
    
    # Check if user started typing (search changed from empty to something)
    # or if they're typing a different name
    current_search = player_search.strip()
    if current_search != st.session_state.last_search:
        # Clear player data only when search actually changes
        if (st.session_state.last_search != "" and current_search != st.session_state.last_search) or \
           (st.session_state.player_data is not None and current_search == ""):
            clear_player_data()
        st.session_state.last_search = current_search
    
    if player_search:
        matching_players = search_players(player_search, season)
        if matching_players:
            selected_player = st.selectbox("Select from matches:", matching_players, key="player_select")
            
            # Fetch player data when selected and "Load Player Data" is clicked
            if st.button("Load Player Data", type="primary", key="load_player"):
                with st.spinner(f"Fetching {selected_player}'s game log..."):
                    player_df, player_team = get_player_game_log(selected_player, season)
                
                if player_df is None or len(player_df) == 0:
                    st.error(f"No games found for {selected_player} in {season} season. Player may be inactive or injured.")
                else:
                    st.session_state.player_data = player_df
                    st.session_state.player_team = player_team
                    st.session_state.selected_player = selected_player
                    st.success(f"Loaded {len(player_df)} games for {selected_player} (Team: {player_team})")
                    st.rerun()  # Refresh to show opponent selection
        else:
            st.warning("No active players found with that name. This player may be retired or not in the current season.")
            st.info("Try searching for current NBA players like: LeBron James, Stephen Curry, Giannis Antetokounmpo")
            clear_player_data()  # Clear any existing data
    else:
        # If search box is empty and we have data, clear it
        if st.session_state.player_data is not None:
            clear_player_data()
    
    # Check if we have player data loaded
    if st.session_state.player_data is not None:
        selected_player = st.session_state.selected_player
        player_team = st.session_state.player_team
        player_df = st.session_state.player_data

        # --- ADD SAVE BUTTONS HERE ---
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Favorite {selected_player}"):
                if add_to_favorites(selected_player, "fav_players"):
                    st.toast(f"Added {selected_player} to favorites!")
        with c2:
            if st.button(f"Watch {player_team}"):
                if add_to_favorites(player_team, "fav_teams"):
                    st.toast(f"Added {player_team} to watched teams!")
        # -----------------------------

        st.info(f" Currently loaded: **{selected_player}** (Team: {player_team}) - {len(player_df)} games")

        # Show recent games right after loading - ADD MINUTES
        st.markdown("### Recent Performance")
        
        # Define columns to display with minutes
        recent_cols = [
            'GAME_DATE', 'MATCHUP', 'MIN', 'Points', 'Rebounds', 'Assists', 
            'Steals', 'Blocks', 'Turnovers', 'FG', '3P', 'FT', 'TS%'
        ]
        
        # Filter to only include columns that exist
        available_cols = [col for col in recent_cols if col in player_df.columns]
        
        # Get last 5 games (most recent at top)
        recent_games = player_df.tail(5)[available_cols].iloc[::-1].copy()
        
        # Format TS% column
        if 'TS%' in recent_games.columns:
            recent_games['TS%'] = recent_games['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        
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
            total_fga = last_5_df['FGA'].sum()
            total_fta = last_5_df['FTA'].sum()
            ts_pct = round((total_points / (2 * (total_fga + 0.44 * total_fta)) * 100), 1) if (total_fga + 0.44 * total_fta) > 0 else 0
        else:
            ts_pct = "N/A"
        
        # Create averages row
        averages_row = {
            'GAME_DATE': 'AVG (Last 5)',
            'MATCHUP': '',
            'MIN': f"{avg_minutes:.1f}" if avg_minutes != "N/A" else "N/A",
            'Points': f"{avg_points:.1f}",
            'Rebounds': f"{avg_rebounds:.1f}",
            'Assists': f"{avg_assists:.1f}",
            'Steals': f"{avg_steals:.1f}",
            'Blocks': f"{avg_blocks:.1f}",
            'Turnovers': f"{avg_turnovers:.1f}",
            'FG': f"{fg_pct:.1f}%" if fg_pct != "N/A" else "N/A",
            '3P': f"{three_pct:.1f}%" if three_pct != "N/A" else "N/A",
            'FT': f"{ft_pct:.1f}%" if ft_pct != "N/A" else "N/A",
            'TS%': f"{ts_pct:.1f}%" if isinstance(ts_pct, (int, float)) else ts_pct
        }
        
        # Add the averages row to the dataframe
        averages_df_row = pd.DataFrame([averages_row])
        
        # Combine with recent games
        display_df = pd.concat([recent_games, averages_df_row], ignore_index=True)
        
        # Highlight the averages row
        def highlight_average_row(row):
            if row['GAME_DATE'] == 'AVG (Last 5)':
                return ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
            else:
                return [''] * len(row)
        
        # Display the table with styling
        st.dataframe(
            display_df.style.apply(highlight_average_row, axis=1),
            use_container_width=True,
            hide_index=True
    )

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
            selected_opponent = st.selectbox(
                "Opponent Team:",
                available_teams,
                help=f"{selected_player} currently plays for {player_team} (excluded from list)"
            )
            
            opp_rating = team_def_ratings.get(selected_opponent, 0)
            st.caption(f"Defensive Rating: **{opp_rating}** (Lower is better defense)")
            
            # Calculate player's averages against this opponent this season
            st.markdown("### Games vs " + selected_opponent + " This Season")
            
            # Filter games against this opponent
            games_vs_opponent = player_df[player_df['Opponent'] == selected_opponent]
            
            if len(games_vs_opponent) > 0:
                # Calculate averages against this opponent
                num_games = len(games_vs_opponent)
                
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
                    
                    games_vs_opponent['MIN_NUM'] = games_vs_opponent['MIN'].apply(parse_minutes_simple)
                    avg_minutes_vs = round(games_vs_opponent['MIN_NUM'].mean(), 1)
                else:
                    avg_minutes_vs = "N/A"
                
                # Calculate TS%
                if 'FGA' in games_vs_opponent.columns and 'FTA' in games_vs_opponent.columns:
                    total_points_vs = games_vs_opponent['Points'].sum()
                    total_fga_vs = games_vs_opponent['FGA'].sum()
                    total_fta_vs = games_vs_opponent['FTA'].sum()
                    ts_pct_vs = round((total_points_vs / (2 * (total_fga_vs + 0.44 * total_fta_vs)) * 100), 1) if (total_fga_vs + 0.44 * total_fta_vs) > 0 else 0
                else:
                    ts_pct_vs = "N/A"
                
                # Show individual game results against this opponent with averages row
                st.markdown(f"**Games Played: {num_games}**")
                
                # Get individual games
                vs_opponent_display = games_vs_opponent[['GAME_DATE', 'MATCHUP', 'MIN', 'Points', 'Rebounds', 'Assists', 
                                                        'Steals', 'Blocks', 'Turnovers', 'FG', '3P', 'FT', 'TS%']].iloc[::-1].copy()
                
                # Format TS% for individual games
                if 'TS%' in vs_opponent_display.columns:
                    vs_opponent_display['TS%'] = vs_opponent_display['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                
                # Create averages row
                averages_row_vs = {
                    'GAME_DATE': 'AVG vs ' + selected_opponent,
                    'MATCHUP': f'({num_games} games)',
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
                
                # Highlight the averages row
                def highlight_vs_average_row(row):
                    if 'AVG vs ' + selected_opponent in str(row['GAME_DATE']):
                        return ['background-color: #2D3748; font-weight: bold; color: #FF6B35'] * len(row)
                    else:
                        return [''] * len(row)
                
                # Display the table with styling
                st.dataframe(
                    combined_display.style.apply(highlight_vs_average_row, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.info(f"**{selected_player}** has not played against **{selected_opponent}** yet this season.")
        
        st.markdown("---")
        
        # Fixed parameters
        n_states = 4
        use_weighting = True
        weight_strength = 'medium'
            
        if st.button("Generate Prediction", type="primary", use_container_width=True):
            with st.spinner("Training prediction model..."):
                model, stat_cols, scaler, filtered_df = train_hmm_with_drtg(
                    player_df, 
                    team_def_ratings, 
                    n_states=n_states,
                    use_temporal_weighting=use_weighting,
                    weight_strength=weight_strength
                )
            
            if model is None:
                st.error("Insufficient data to train model. Need at least 5 games.")
            else:
                consistency = calculate_player_consistency(filtered_df, ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers'])
                consistency_interpretation = "High Variance" if consistency > 0.5 else "Moderate" if consistency > 0.3 else "Very Consistent"
                st.info(f"Player Consistency: **{consistency_interpretation}** (CV: {consistency:.2f})")
                
                with st.spinner("Generating prediction..."):
                    prediction = predict_with_drtg(
                        model, stat_cols, scaler, filtered_df,
                        team_def_ratings, selected_opponent
                    )
                
                if prediction:
                    st.success("Prediction Complete!")
                    st.markdown(f"### Predicted Stats: {selected_player} vs {selected_opponent}")
                    
                    # Display metrics in a nice grid
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
            st.info("Search for a player and click 'Load Player Data' to get started")
        
        # # Show popular players as examples
        # st.markdown("### 🌟 Popular Players")
        # popular_col1, popular_col2, popular_col3 = st.columns(3)
        
        # with popular_col1:
        #     st.write("• LeBron James")
        #     st.write("• Stephen Curry")
        #     st.write("• Giannis Antetokounmpo")
        
        # with popular_col2:
        #     st.write("• Luka Doncic")
        #     st.write("• Nikola Jokic")
        #     st.write("• Joel Embiid")
        
        # with popular_col3:
        #     st.write("• Kevin Durant")
        #     st.write("• Jayson Tatum")
        #     st.write("• Anthony Davis")
    
    # Clear data button
    if st.session_state.player_data is not None:
        if st.button("Clear Current Player", type="secondary"):
            st.session_state.player_data = None
            st.session_state.player_team = None
            st.session_state.filtered_df = None
            st.session_state.selected_player = None
            st.rerun()


# ==================== PLAYER STATS PAGE ====================

elif page == "Player Stats":
    st.title("Player Season Statistics")
    st.markdown("View detailed season statistics for any NBA player")
    st.caption("Only showing players active in current season")
    
    player_search = st.text_input("Search player:", placeholder="e.g., Anthony Edwards", key="stats_search")
    
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
                    st.success(f"Loaded {len(player_df)} games (Team: {player_team if player_team else 'Unknown'})")
                    
                    # Calculate season totals for accurate percentages
                    if 'FGM' in player_df.columns and 'FGA' in player_df.columns:
                        total_fgm = player_df['FGM'].sum()
                        total_fga = player_df['FGA'].sum()
                        fg_pct = (total_fgm / total_fga * 100) if total_fga > 0 else 0
                        avg_fgm = player_df['FGM'].mean()
                        avg_fga = player_df['FGA'].mean()
                    else:
                        total_fgm = total_fga = fg_pct = avg_fgm = avg_fga = 0
                    
                    if '3PM' in player_df.columns and '3PA' in player_df.columns:
                        total_3pm = player_df['3PM'].sum()
                        total_3pa = player_df['3PA'].sum()
                        three_pct = (total_3pm / total_3pa * 100) if total_3pa > 0 else 0
                        avg_3pm = player_df['3PM'].mean()
                        avg_3pa = player_df['3PA'].mean()
                    else:
                        total_3pm = total_3pa = three_pct = avg_3pm = avg_3pa = 0
                    
                    if 'FTM' in player_df.columns and 'FTA' in player_df.columns:
                        total_ftm = player_df['FTM'].sum()
                        total_fta = player_df['FTA'].sum()
                        ft_pct = (total_ftm / total_fta * 100) if total_fta > 0 else 0
                        avg_ftm = player_df['FTM'].mean()
                        avg_fta = player_df['FTA'].mean()
                    else:
                        total_ftm = total_fta = ft_pct = avg_ftm = avg_fta = 0
                    
                    # Calculate minutes played
                    if 'MIN' in player_df.columns:
                        # Convert minutes from "MM:SS" format to decimal minutes
                        def convert_minutes(min_str):
                            if pd.isna(min_str):
                                return 0
                            try:
                                if ':' in str(min_str):
                                    parts = str(min_str).split(':')
                                    minutes = int(parts[0])
                                    seconds = int(parts[1]) if len(parts) > 1 else 0
                                    return minutes + seconds/60
                                else:
                                    return float(min_str)
                            except:
                                return 0
                        
                        player_df['MIN_DECIMAL'] = player_df['MIN'].apply(convert_minutes)
                        avg_minutes = player_df['MIN_DECIMAL'].mean()
                        total_minutes = player_df['MIN_DECIMAL'].sum()
                    else:
                        avg_minutes = total_minutes = 0
                    
                    # Calculate True Shooting Percentage from totals
                    total_points = player_df['Points'].sum()
                    ts_pct = (total_points / (2 * (total_fga + 0.44 * total_fta)) * 100) if (total_fga + 0.44 * total_fta) > 0 else 0
                    
                    # Summary stats - ADD MINUTES COLUMN
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
                        st.metric("TS%", f"{ts_pct:.1f}%")
                    
                    with col6:
                        st.metric("MPG", f"{avg_minutes:.1f}")
                        st.metric("Games", len(player_df))
                    
                    # Add additional shooting metrics row
                    st.markdown("### Shooting Splits")
                    shoot_col1, shoot_col2, shoot_col3, shoot_col4, shoot_col5, shoot_col6 = st.columns(6)
                    
                    with shoot_col1:
                        # Show FGM/FGA per game
                        fg_per_game = f"{avg_fgm:.1f}/{avg_fga:.1f}"
                        st.metric("FG (M/A)", fg_per_game)
                    
                    # with shoot_col2:
                    #     # FG Percentage (season total)
                    #     st.metric("FG%", f"{fg_pct:.1f}%")
                    
                    with shoot_col2:
                        # Show 3PM/3PA per game
                        threes_per_game = f"{avg_3pm:.1f}/{avg_3pa:.1f}"
                        st.metric("3P (M/A)", threes_per_game)
                    
                    with shoot_col3:
                        # FT (M/A) per game
                        ft_per_game = f"{avg_ftm:.1f}/{avg_fta:.1f}"
                        st.metric("FT (M/A)", ft_per_game)
                    
                    # with shoot_col5:
                    #     # FT Percentage (season total)
                    #     st.metric("FT%", f"{ft_pct:.1f}%")
                    
                    # with shoot_col4:
                    #     #Minutes Per Game
                    #     st.metric("MPG", f"{avg_minutes:.1f}")
                    
                    st.markdown("---")
                    
                    # Game log (most recent first) - ADD MINUTES COLUMN
                    st.markdown("### Game Log")
                    
                    # Define all available columns including minutes
                    display_cols = [
                        'GAME_DATE', 'MATCHUP', 'MIN', 'Points', 'Rebounds', 'Assists', 
                        'Steals', 'Blocks', 'Turnovers', 'FG', 'FG%', 
                        '3P', '3P%', 'FT', 'FT%', 'TS%', 'PF'
                    ]
                    
                    # Filter to only include columns that exist in the dataframe
                    available_cols = [col for col in display_cols if col in player_df.columns]
                    
                    # Format percentage columns for display
                    display_df = player_df[available_cols].iloc[::-1].copy()
                    
                    # Format percentage columns to show as percentages
                    for pct_col in ['FG%', '3P%', 'FT%']:
                        if pct_col in display_df.columns:
                            display_df[pct_col] = display_df[pct_col].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "0.0%")
                    
                    if 'TS%' in display_df.columns:
                        display_df['TS%'] = display_df['TS%'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==================== FAVORITES PAGE ====================

elif page == "Favorites":
    st.title("My Preferred Players & Teams")
    
    prefs = load_preferences()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Favorite Players")
        if not prefs["fav_players"]:
            st.info("No favorite players saved yet.")
        else:
            for player in prefs["fav_players"]:
                p_col, b_col = st.columns([3, 1])
                p_col.write(f"**{player}**")
                if b_col.button("🗑️", key=f"remove_p_{player}"):
                    remove_from_favorites(player, "fav_players")
                    st.rerun()
                
                # Quick Action Button
                if st.button(f"Analyze {player}", key=f"analyze_{player}"):
                    st.session_state.player_search = player
                    # We trigger a search by setting the session state
                    st.info(f"Navigate to 'Live Predictions' to see {player}")

    with col2:
        st.subheader("🛡️ Watched Teams")
        if not prefs["fav_teams"]:
            st.info("No favorite teams saved yet.")
        else:
            for team in prefs["fav_teams"]:
                t_col, tb_col = st.columns([3, 1])
                t_col.write(f"**{team}**")
                if tb_col.button("🗑️", key=f"remove_t_{team}"):
                    remove_from_favorites(team, "fav_teams")
                    st.rerun()

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
    - **Comprehensive**: Predicts 6 key statistics
    
    ### Statistics Predicted
    
    - Points (PTS)
    - Assists (AST)
    - Rebounds (REB)
    - Steals (STL)
    - Blocks (BLK)
    - Turnovers (TOV)
    
    ###  Technologies
    
    - **Streamlit**: Interactive web interface
    - **nba_api**: Official NBA data source
    - **hmmlearn**: Hidden Markov Model implementation
    - **scikit-learn**: Data preprocessing and scaling
    - **pandas & numpy**: Data manipulation
    - **matplotlib**: Data visualization
    
    ### Data Sources
    
    All data is fetched directly from NBA.com's official API endpoints:
    - `leaguedashteamstats`: Team defensive ratings
    - `playergamelog`: Individual player game logs
    
    ### Disclaimer
    
    This tool is for entertainment and educational purposes. Predictions are based on statistical models and may not reflect actual game outcomes.
    """)

