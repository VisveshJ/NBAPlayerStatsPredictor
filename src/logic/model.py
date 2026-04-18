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


def compute_playoff_adjustments(player_df, playoff_games_df=None):
    """
    Compute playoff-specific multipliers for each stat.

    Playoff basketball differs from the regular season in key ways:
    - Tighter rotations: fewer players get minutes (~8 vs ~10 in reg season).
      Stars often play more minutes per game.
    - Slower pace: defenses study opponents game-by-game; fewer possessions.
      Upper-bound scoring opportunities decrease slightly.
    - Higher defensive intensity: points and assists see a small pullback unless
      the player has already shown strong playoff performance.
    - Physical/fatigue factor: if a series is already deep (games 5-7), fatigue
      compounds; if early (games 1-2), players are fresh.

    Returns a dict: {stat: multiplier}  where multiplier is applied to the
    base HMM+H2H prediction before bounding.

    Parameters
    ----------
    player_df : DataFrame – full regular-season game log (already filtered)
    playoff_games_df : DataFrame or None – playoff games played by this player
        this post-season (subset of game log with GAME_DATE >= playoff start).
        Must have the same columns as player_df.
    """
    adjustments = {
        'Points': 1.0,
        'Assists': 1.0,
        'Rebounds': 1.0,
        'Steals': 1.0,
        'Blocks': 1.0,
        'Turnovers': 1.0,
    }

    # --- Minutes adjustment ---
    # In the playoffs, stars typically play ~3–5 more minutes per game.
    # We won't change minutes directly but use them to scale stats.
    reg_min = 0.0
    if 'MIN' in player_df.columns and len(player_df) > 0:
        reg_min = player_df['MIN'].apply(pd.to_numeric, errors='coerce').mean()

    # --- If we have actual playoff game data, use it for a direct calibration ---
    if playoff_games_df is not None and len(playoff_games_df) >= 2:
        # Compute per-stat ratio: playoff avg / regular-season avg
        # Weight actual playoff data heavily once we have ≥3 games
        playoff_weight = min(0.60, 0.25 + 0.10 * len(playoff_games_df))  # cap at 60%

        for stat in ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers']:
            if stat not in player_df.columns or stat not in playoff_games_df.columns:
                continue
            reg_avg = player_df[stat].mean()
            if reg_avg <= 0:
                continue
            po_avg = playoff_games_df[stat].mean()
            # How much better/worse in playoffs vs reg season
            ratio = po_avg / reg_avg  # e.g. 1.10 = 10% better in playoffs
            # Blend: shift the base prediction toward actual playoff performance
            # adjustment = 1.0 + playoff_weight * (ratio - 1.0)
            adjustments[stat] = 1.0 + playoff_weight * (ratio - 1.0)

        # Also factor in minutes trend: if player is logging more playoff minutes, scale up
        if 'MIN' in playoff_games_df.columns and reg_min > 0:
            po_min = playoff_games_df['MIN'].apply(pd.to_numeric, errors='coerce').mean()
            min_ratio = po_min / reg_min
            # Apply a moderate scale to counting stats
            for stat in ['Points', 'Assists', 'Rebounds']:
                # Multiply by sqrt of minutes ratio to avoid over-adjusting
                adjustments[stat] *= (min_ratio ** 0.5)

    else:
        # No actual playoff data yet – apply conservative baseline adjustments
        # reflecting the statistical reality of playoff basketball:
        # • Pace slows ~3–5% → fewer possessions → slight reduction in counting stats
        # • But stars play more → partially offsets this
        # Net effect: a small negative on scoring/assists, small positive on defense

        pace_reduction = 0.97   # ~3% fewer possessions
        defense_boost = 1.03    # harder to score/create → slightly fewer assists

        adjustments['Points'] = pace_reduction * 0.99   # -4% pts baseline
        adjustments['Assists'] = pace_reduction * 0.98  # -5% assists baseline (ball movement tighter)
        adjustments['Rebounds'] = 1.01                  # slightly more in slower game
        adjustments['Steals'] = 1.02                    # more pressure defense
        adjustments['Blocks'] = 1.02
        adjustments['Turnovers'] = 0.98                 # better ball control in playoffs usually

    return adjustments


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


def predict_with_drtg(model, stat_cols, scaler, recent_df, team_def_ratings, target_opponent,
                      full_player_df=None, playoff_games_df=None, is_playoff_game=False):
    """
    Generate prediction with defensive rating, head-to-head, and optional playoff context.

    Parameters
    ----------
    model, stat_cols, scaler, recent_df : HMM model outputs from train_hmm_with_drtg
    team_def_ratings : dict of team abbrev → defensive rating
    target_opponent : str – opponent team abbreviation
    full_player_df : DataFrame – full season game log (for H2H lookup)
    playoff_games_df : DataFrame or None – subset of game log that are playoff games
        (games with GAME_DATE on or after the first playoff game date).
        If provided, the prediction blends these in for calibration.
    is_playoff_game : bool – if True, apply playoff-specific adjustments even if
        playoff_games_df is None (uses baseline adjustments).
    """
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
        #
        # In the playoffs, we prioritise the most recent (playoff) games more.
        # If playoff data is available, weight recent (playoff) games at 60%; otherwise 50%.
        _has_playoff_data = playoff_games_df is not None and len(playoff_games_df) >= 1
        if _has_playoff_data:
            recent_weight = 0.60  # lean more on recent playoff form
            # Use playoff games as the "recent" context if available
            po_recent = playoff_games_df.tail(5)
            if len(po_recent) >= 2 and all(c in po_recent.columns for c in stat_cols):
                recent_averages = po_recent[stat_cols].mean().values
            else:
                recent_averages = last_games[stat_cols].mean().values
        else:
            recent_weight = 0.50
            recent_averages = last_games[stat_cols].mean().values


        for i, stat in enumerate(stat_cols):
            hmm_value = predicted_unscaled[i]
            recent_value = recent_averages[i]
            predicted_unscaled[i] = (1 - recent_weight) * hmm_value + recent_weight * recent_value
        
        # === HEAD-TO-HEAD ADJUSTMENT ===
        # In the playoffs this is strengthened because teams game-plan specifically
        # against each other and the sample of prior playoff matchups is more predictive.
        h2h_adjustment = {}
        h2h_weight = 0.0
        games_vs_opp = pd.DataFrame()
        
        if full_player_df is not None and 'Opponent' in full_player_df.columns:
            games_vs_opp = full_player_df[full_player_df['Opponent'] == target_opponent].copy()
            
            # Filter out injury-shortened games from H2H (< 60% of player's average minutes)
            if 'MIN' in games_vs_opp.columns and len(games_vs_opp) > 0:
                games_vs_opp['MIN_numeric'] = pd.to_numeric(games_vs_opp['MIN'], errors='coerce')
                avg_minutes = full_player_df['MIN'].apply(pd.to_numeric, errors='coerce').mean()
                min_minutes_threshold = avg_minutes * 0.6
                games_vs_opp = games_vs_opp[games_vs_opp['MIN_numeric'] >= min_minutes_threshold]
            
            if len(games_vs_opp) > 0:
                for stat in stat_cols:
                    if stat in games_vs_opp.columns:
                        h2h_adjustment[stat] = games_vs_opp[stat].mean()
                
                # Playoff versus playoff: if opponent is the same postseason opponent,
                # weight those series games more heavily (each game in a series is very
                # predictive of the next because rotations, schemes don't change).
                playoff_h2h = pd.DataFrame()
                if playoff_games_df is not None:
                    playoff_h2h = games_vs_opp[games_vs_opp.index.isin(playoff_games_df.index)]

                if len(playoff_h2h) >= 2:
                    # Have 2+ playoff games vs this exact opponent → very high confidence
                    h2h_weight = min(0.55, 0.30 + 0.08 * len(playoff_h2h))
                    for stat in stat_cols:
                        if stat in playoff_h2h.columns:
                            h2h_adjustment[stat] = playoff_h2h[stat].mean()
                elif len(games_vs_opp) >= 3:
                    h2h_weight = 0.40
                elif len(games_vs_opp) >= 2:
                    h2h_weight = 0.30
                else:
                    h2h_weight = 0.20

        # === PLAYOFF ADJUSTMENTS ===
        # Apply after H2H blend so the playoff calibration is applied to the most
        # informed prediction we can generate.
        if is_playoff_game or (playoff_games_df is not None and len(playoff_games_df) >= 1):
            po_adjustments = compute_playoff_adjustments(
                full_player_df if full_player_df is not None else recent_df,
                playoff_games_df
            )
        else:
            po_adjustments = {stat: 1.0 for stat in stat_cols}

        result = {}
        defensive_sensitive_stats = ['Points', 'Assists', 'Turnovers']
        
        for i, stat in enumerate(stat_cols):
            value = predicted_unscaled[i]
            if stat in defensive_sensitive_stats:
                value *= defensive_factor
            
            # Blend with head-to-head data if available
            if stat in h2h_adjustment and h2h_weight > 0:
                h2h_value = h2h_adjustment[stat]
                value = (1 - h2h_weight) * value + h2h_weight * h2h_value
            
            # Apply playoff adjustment multiplier
            value *= po_adjustments.get(stat, 1.0)
            
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
        
        # Store metadata for display purposes
        result['_h2h_weight'] = h2h_weight
        result['_h2h_games'] = len(games_vs_opp)
        result['_is_playoff'] = is_playoff_game or (playoff_games_df is not None and len(playoff_games_df) >= 1)
        result['_playoff_games_used'] = len(playoff_games_df) if playoff_games_df is not None else 0
        result['_po_adjustments'] = po_adjustments
        
        return result
        
    except Exception as e:
        return None
