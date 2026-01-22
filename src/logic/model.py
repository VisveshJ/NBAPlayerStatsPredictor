import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import streamlit as st

def ensure_minimum_transitions(model, min_prob=0.01):
    """Ensure all states have minimum transition probability."""
    n_states = model.n_components
    transmat = model.transmat_.copy()
    
    transmat[transmat < min_prob] = min_prob
    
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


def train_hmm_with_drtg(player_df, team_def_ratings, n_states=3, use_temporal_weighting=True, 
                        weight_strength='medium', min_transition_prob=0.01):
    """Train HMM with opponent defense as a feature."""
    player_df = player_df.copy()
    
    # Filter out injury-shortened games (< 60% of player's average minutes)
    if 'MIN' in player_df.columns:
        # Convert MIN to numeric if needed
        player_df['MIN_numeric'] = pd.to_numeric(player_df['MIN'], errors='coerce')
        avg_minutes = player_df['MIN_numeric'].mean()
        min_minutes_threshold = avg_minutes * 0.6
        games_before = len(player_df)
        player_df = player_df[player_df['MIN_numeric'] >= min_minutes_threshold]
        games_filtered = games_before - len(player_df)
        if games_filtered > 0:
            # Store filter info in session state for display
            try:
                if 'games_filtered' not in st.session_state:
                    st.session_state.games_filtered = {}
                st.session_state.games_filtered['count'] = games_filtered
                st.session_state.games_filtered['threshold'] = min_minutes_threshold
            except:
                pass # safely ignore if not running in streamlit context
    
    player_df['Opponent_DEF_RTG'] = player_df['Opponent'].map(team_def_ratings)
    player_df = player_df.dropna(subset=['Opponent_DEF_RTG'])
    
    stat_cols = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers']
    player_df = player_df.dropna(subset=stat_cols)
    
    if len(player_df) < 5:
        return None, None, None, None
    
    feature_cols = stat_cols + ['Opponent_DEF_RTG']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(player_df[feature_cols].values)
    
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=1000,
        random_state=100,
        verbose=False,
        init_params='ste',
        tol=0.001,
        min_covar=0.001
    )
    model.fit(X)

    if min_transition_prob > 0:
        model = ensure_minimum_transitions(model, min_prob=min_transition_prob)
    
    return model, stat_cols, scaler, player_df


def predict_with_drtg(model, stat_cols, scaler, recent_df, team_def_ratings, target_opponent, full_player_df=None):
    """Generate prediction with defensive rating and head-to-head performance."""
    target_drtg = team_def_ratings.get(target_opponent)
    if target_drtg is None:
        return None
    
    last_games = recent_df.tail(5)
    if len(last_games) < 2:
        return None
    
    feature_cols = stat_cols + ['Opponent_DEF_RTG']
    
    for col in feature_cols:
        if col not in last_games.columns:
            return None
    
    X_sequence = last_games[feature_cols].values
    X_sequence_scaled = scaler.transform(X_sequence)
    
    try:
        log_prob, state_sequence = model.decode(X_sequence_scaled, algorithm="viterbi")
        last_state = state_sequence[-1]
        
        next_state_probs = model.transmat_[last_state]
        next_state = np.argmax(next_state_probs)
        
        next_state_mean = model.means_[next_state]
        
        avg_drtg = np.mean(list(team_def_ratings.values()))
        # Amplify defensive factor: make bad defenses boost more, good defenses reduce more
        # Amplify the deviation by 1.5x for moderate effect
        raw_factor = target_drtg / avg_drtg
        deviation = raw_factor - 1.0
        defensive_factor = 1.0 + (deviation * 1.5)  # 1.5x amplification for reasonable adjustment
        
        adjusted_prediction = next_state_mean.copy()
        
        dummy_features = np.zeros((1, len(feature_cols)))
        dummy_features[0, -1] = target_drtg
        scaled_drtg = scaler.transform(dummy_features)[0, -1]
        adjusted_prediction[-1] = scaled_drtg
        
        adjusted_prediction_2d = adjusted_prediction.reshape(1, -1)
        predicted_unscaled = scaler.inverse_transform(adjusted_prediction_2d)[0]
        
        # Blend HMM prediction with recent game averages to prevent extreme predictions
        # This helps when HMM picks a rare low/high state that doesn't reflect current form
        recent_averages = last_games[stat_cols].mean().values
        # 50% HMM prediction, 50% recent average for more stable predictions
        for i, stat in enumerate(stat_cols):
            hmm_value = predicted_unscaled[i]
            recent_value = recent_averages[i]
            predicted_unscaled[i] = 0.5 * hmm_value + 0.5 * recent_value
        
        # === HEAD-TO-HEAD ADJUSTMENT ===
        # Check if we have games vs this specific opponent
        h2h_adjustment = {}
        h2h_weight = 0.0  # Weight for head-to-head data
        
        if full_player_df is not None and 'Opponent' in full_player_df.columns:
            games_vs_opp = full_player_df[full_player_df['Opponent'] == target_opponent].copy()
            
            # Filter out injury-shortened games from H2H (< 60% of player's average minutes)
            if 'MIN' in games_vs_opp.columns and len(games_vs_opp) > 0:
                games_vs_opp['MIN_numeric'] = pd.to_numeric(games_vs_opp['MIN'], errors='coerce')
                avg_minutes = full_player_df['MIN'].apply(pd.to_numeric, errors='coerce').mean()
                min_minutes_threshold = avg_minutes * 0.6
                games_vs_opp = games_vs_opp[games_vs_opp['MIN_numeric'] >= min_minutes_threshold]
            
            if len(games_vs_opp) > 0:
                # Calculate averages against this opponent
                for stat in stat_cols:
                    if stat in games_vs_opp.columns:
                        h2h_adjustment[stat] = games_vs_opp[stat].mean()
                
                # Weight based on number of games played vs this opponent
                # 1 game = 20%, 2 games = 30%, 3+ games = 40%
                if len(games_vs_opp) >= 3:
                    h2h_weight = 0.40
                elif len(games_vs_opp) >= 2:
                    h2h_weight = 0.30
                else:
                    h2h_weight = 0.20
        
        result = {}
        defensive_sensitive_stats = ['Points', 'Assists', 'Turnovers']
        
        for i, stat in enumerate(stat_cols):
            value = predicted_unscaled[i]
            if stat in defensive_sensitive_stats:
                value *= defensive_factor
            
            # Blend with head-to-head data if available
            if stat in h2h_adjustment and h2h_weight > 0:
                h2h_value = h2h_adjustment[stat]
                # Blend: (1 - h2h_weight) * HMM prediction + h2h_weight * H2H average
                value = (1 - h2h_weight) * value + h2h_weight * h2h_value
            
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
        
        # Store the H2H weight used for display purposes
        result['_h2h_weight'] = h2h_weight
        result['_h2h_games'] = len(games_vs_opp) if full_player_df is not None and 'Opponent' in full_player_df.columns else 0
        
        return result
        
    except Exception as e:
        return None
