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

    Applies a recency-decay within playoff_games_df so that games from
    earlier rounds receive lower weight than games from the current round.
    Relies on 'MATCHUP' containing 'vs.' or '@' to distinguish home/road
    if the column is present, but the home/road factor is handled separately
    in predict_with_drtg.
    """
    adjustments = {stat: 1.0 for stat in ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks', 'Turnovers']}

    # --- Minutes Trend Factor ---
    if 'MIN' in player_df.columns and len(player_df) > 0:
        min_series = pd.to_numeric(player_df['MIN'], errors='coerce').fillna(0)
        reg_min = min_series.mean()
        recent_min = min_series.tail(5).mean()

        if reg_min > 5:
            min_trend_ratio = recent_min / reg_min
            trend_factor = min(1.2, max(0.8, min_trend_ratio ** 0.7))
            for stat in ['Points', 'Assists', 'Rebounds']:
                adjustments[stat] = trend_factor

    # --- Actual Playoff Calibration with Recency Decay ---
    # Recent playoff games are weighted more heavily than earlier-round games.
    if playoff_games_df is not None and len(playoff_games_df) >= 1:
        n = len(playoff_games_df)
        # Base weight increases with more games played (caps at 0.50)
        playoff_weight = min(0.50, 0.20 + 0.10 * n)

        # Build per-game decay weights: most recent game = 1.0, each older game
        # discounted by 0.80 (so game n-1 = 0.80, n-2 = 0.64, ...).  This means
        # the current series matters far more than Round 1 games.
        decay = 0.80
        raw_weights = np.array([decay ** i for i in range(n - 1, -1, -1)], dtype=float)
        game_weights = raw_weights / raw_weights.sum()  # normalise

        for stat in adjustments.keys():
            if stat not in player_df.columns or stat not in playoff_games_df.columns:
                continue
            reg_avg = player_df[stat].mean()
            if reg_avg <= 0:
                continue

            po_vals = playoff_games_df[stat].values
            if len(po_vals) != len(game_weights):
                # Fallback to simple mean if shapes mismatch
                po_avg = np.mean(po_vals)
            else:
                po_avg = np.dot(game_weights, po_vals)

            ratio = po_avg / reg_avg
            adjustments[stat] = (adjustments[stat] * (1 - playoff_weight)) + (ratio * playoff_weight)

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
                      full_player_df=None, playoff_games_df=None, is_playoff_game=False,
                      is_home_game=None, opp_injury_score=0.0, own_injury_score=0.0):
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
    is_home_game : bool or None – whether the predicted game is at home. If None,
        the factor is derived automatically from the player's historical home/road
        splits in full_player_df.
    opp_injury_score : float – opponent team injury impact (0–10). Higher = key
        players missing, so the player's offense gets a modest boost.
    own_injury_score : float – player's own team injury impact (0–10). Higher =
        fewer quality teammates, slight drag on stats.
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

        # === HOME / ROAD FACTOR ===
        # Compute the player's home-vs-road split from their full game log if
        # is_home_game is not explicitly supplied.
        home_road_mult = {stat: 1.0 for stat in stat_cols}
        _is_home = is_home_game  # may be None → auto-detect

        src_df = full_player_df if full_player_df is not None else recent_df
        if 'MATCHUP' in src_df.columns and len(src_df) >= 10:
            home_mask = src_df['MATCHUP'].str.contains(r'\bvs\.', na=False)
            road_mask = src_df['MATCHUP'].str.contains(r'@', na=False)
            home_df = src_df[home_mask]
            road_df = src_df[road_mask]

            if len(home_df) >= 3 and len(road_df) >= 3:
                for stat in ['Points', 'Assists', 'Rebounds']:
                    if stat not in home_df.columns:
                        continue
                    home_avg = home_df[stat].mean()
                    road_avg = road_df[stat].mean()
                    overall_avg = src_df[stat].mean()
                    if overall_avg <= 0:
                        continue

                    if _is_home is True:
                        # Amplify upward for home advantage
                        ratio = home_avg / overall_avg
                    elif _is_home is False:
                        # Amplify downward for road games
                        ratio = road_avg / overall_avg
                    else:
                        ratio = 1.0  # unknown — no adjustment

                    # Cap the multiplier at ±15% to avoid noise dominating
                    home_road_mult[stat] = min(1.15, max(0.85, ratio))

        # === INJURY FACTOR ===
        # opponent injuries → offensive boost (capped at +12 %)
        # own-team injuries → slight stat drag (capped at −12 %)
        opp_inj_mult  = 1.0 + min(0.12, opp_injury_score * 0.012)
        own_inj_mult  = 1.0 - min(0.12, own_injury_score * 0.012)
        inj_sensitive = ['Points', 'Assists', 'Rebounds']  # stats most affected

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
            
            # Apply playoff adjustment multiplier (with recency decay)
            value *= po_adjustments.get(stat, 1.0)

            # Apply home/road multiplier
            value *= home_road_mult.get(stat, 1.0)

            # Apply injury multipliers to sensitive stats
            if stat in inj_sensitive:
                value *= opp_inj_mult
                value *= own_inj_mult
            
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
        result['_home_road_mult'] = home_road_mult
        result['_opp_inj_mult'] = round(opp_inj_mult, 3)
        result['_own_inj_mult'] = round(own_inj_mult, 3)
        
        return result
        
    except Exception as e:
        return None
