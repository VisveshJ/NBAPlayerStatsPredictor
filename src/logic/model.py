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
        
        # ── SEASON AVERAGE ANCHOR (used by both paths) ──────────────────────
        _src = full_player_df if full_player_df is not None else recent_df
        season_averages = np.array([
            _src[s].mean() if s in _src.columns else 0.0
            for s in stat_cols
        ])

        _has_playoff_data = playoff_games_df is not None and len(playoff_games_df) >= 1
        _skip_standard_h2h = False  # set True when playoff blend already covers it

        if is_playoff_game and _has_playoff_data:
            # ═══════════════════════════════════════════════════════════════
            # PLAYOFF MULTI-SOURCE BLEND
            # Sources (in priority order):
            #   1. Current series games vs this exact opponent  (most predictive)
            #   2. Reg-season H2H vs same opponent (injury-filtered)
            #   3. Playoff games vs similar defenses (opponent DRTG ± 5)
            #   4. Overall playoff average (all opponents this postseason)
            #   5. Season average anchor
            #   6. HMM state (floor of 5% so model still contributes)
            # ═══════════════════════════════════════════════════════════════
            _skip_standard_h2h = True  # incorporated below

            po_enriched = playoff_games_df.copy()
            # Enrich with DRTG for similar-defense lookup
            if 'Opponent' in po_enriched.columns and 'Opponent_DEF_RTG' not in po_enriched.columns:
                po_enriched['Opponent_DEF_RTG'] = po_enriched['Opponent'].map(team_def_ratings)

            avg_drtg_val = np.mean(list(team_def_ratings.values()))

            # 1. Series games vs current opponent
            series_games = (
                po_enriched[po_enriched['Opponent'] == target_opponent].copy()
                if 'Opponent' in po_enriched.columns else pd.DataFrame()
            )

            # 2. Reg-season H2H (injury-filtered: ≥60 % avg minutes)
            reg_h2h = pd.DataFrame()
            if full_player_df is not None and 'Opponent' in full_player_df.columns:
                reg_h2h = full_player_df[full_player_df['Opponent'] == target_opponent].copy()
                if 'MIN' in reg_h2h.columns and len(reg_h2h) > 0:
                    reg_h2h['_MIN_n'] = pd.to_numeric(reg_h2h['MIN'], errors='coerce')
                    _avg_m = pd.to_numeric(full_player_df['MIN'], errors='coerce').mean()
                    reg_h2h = reg_h2h[reg_h2h['_MIN_n'] >= _avg_m * 0.6]

            # 3. Playoff games vs similar defenses (DRTG within ±5, excluding current opp)
            similar_po_df = pd.DataFrame()
            if 'Opponent_DEF_RTG' in po_enriched.columns and 'Opponent' in po_enriched.columns:
                _sim_mask = (
                    (po_enriched['Opponent_DEF_RTG'] - target_drtg).abs() <= 5.0
                ) & (po_enriched['Opponent'] != target_opponent)
                similar_po_df = po_enriched[_sim_mask].copy()

            # 4. Overall playoff average (all opponents)
            full_po_avg = np.array([
                po_enriched[s].mean() if s in po_enriched.columns else season_averages[i]
                for i, s in enumerate(stat_cols)
            ])

            # ── Dynamic weight allocation ────────────────────────────────
            n_series  = len(series_games)
            n_h2h_reg = len(reg_h2h)
            n_similar = len(similar_po_df)

            # Series weight: 15% per game up to 45% — grows as the series unfolds
            series_w  = min(0.45, n_series * 0.15)
            # Reg H2H: meaningful if ≥2 full-health games
            h2h_reg_w = 0.15 if n_h2h_reg >= 2 else (0.07 if n_h2h_reg == 1 else 0.0)
            # Similar PO defense: bonus signal if available
            similar_w = 0.12 if n_similar >= 3 else (0.06 if n_similar >= 1 else 0.0)
            # Overall PO average: modest constant anchor
            po_full_w = 0.08
            # Season anchor: keeps predictions tethered to proven baseline
            season_w  = 0.20
            # HMM: gets at least 5% — the remainder after everything else
            hmm_w     = max(0.05, 1.0 - series_w - h2h_reg_w - similar_w - po_full_w - season_w)

            # Normalise to guarantee sum = 1.0
            _total_w = series_w + h2h_reg_w + similar_w + po_full_w + season_w + hmm_w
            series_w  /= _total_w
            h2h_reg_w /= _total_w
            similar_w /= _total_w
            po_full_w /= _total_w
            season_w  /= _total_w
            hmm_w     /= _total_w

            # ── Per-stat averages for each source ────────────────────────
            def _src_avg(df, i, s):
                """Mean for stat s in df, or fall back to season avg."""
                return df[s].mean() if (len(df) > 0 and s in df.columns) else season_averages[i]

            series_avg  = np.array([_src_avg(series_games,  i, s) for i, s in enumerate(stat_cols)])
            h2h_reg_avg = np.array([_src_avg(reg_h2h,       i, s) for i, s in enumerate(stat_cols)])
            similar_avg = np.array([_src_avg(similar_po_df, i, s) for i, s in enumerate(stat_cols)])

            # ── Recency decay within series games ────────────────────────
            # More recent series games get higher weight (0.80 decay per older game)
            if n_series > 0:
                _decay  = 0.80
                _rw     = np.array([_decay ** k for k in range(n_series - 1, -1, -1)], dtype=float)
                _rw    /= _rw.sum()
                series_avg = np.array([
                    np.dot(_rw, series_games[s].values) if s in series_games.columns else season_averages[i]
                    for i, s in enumerate(stat_cols)
                ])

            # ── Final playoff blend ──────────────────────────────────────
            for i in range(len(stat_cols)):
                predicted_unscaled[i] = (
                    hmm_w     * predicted_unscaled[i] +
                    series_w  * series_avg[i]  +
                    h2h_reg_w * h2h_reg_avg[i] +
                    similar_w * similar_avg[i] +
                    po_full_w * full_po_avg[i] +
                    season_w  * season_averages[i]
                )

            # Store playoff blend metadata
            _po_blend_meta = {
                'series_games': n_series,
                'h2h_reg_games': n_h2h_reg,
                'similar_po_games': n_similar,
                'weights': {
                    'series':     round(series_w, 3),
                    'h2h_reg':    round(h2h_reg_w, 3),
                    'similar_po': round(similar_w, 3),
                    'po_full':    round(po_full_w, 3),
                    'season':     round(season_w, 3),
                    'hmm':        round(hmm_w, 3),
                }
            }

        else:
            # ── STANDARD 3-WAY BLEND (regular season / no PO data) ──────
            # 40% HMM  +  35% recent-10-game avg  +  25% season avg
            hmm_weight    = 0.40
            recent_weight = 0.35
            season_weight = 0.25
            recent_averages = recent_df.tail(10)[stat_cols].mean().values

            for i, stat in enumerate(stat_cols):
                predicted_unscaled[i] = (
                    hmm_weight    * predicted_unscaled[i] +
                    recent_weight * recent_averages[i]    +
                    season_weight * season_averages[i]
                )
            _po_blend_meta = None

        # === HEAD-TO-HEAD ADJUSTMENT (standard path only) ===============
        # In playoff path this is already baked into the blend above.
        h2h_adjustment = {}
        h2h_weight = 0.0
        games_vs_opp = pd.DataFrame()

        if not _skip_standard_h2h and full_player_df is not None and 'Opponent' in full_player_df.columns:
            games_vs_opp = full_player_df[full_player_df['Opponent'] == target_opponent].copy()

            if 'MIN' in games_vs_opp.columns and len(games_vs_opp) > 0:
                games_vs_opp['MIN_numeric'] = pd.to_numeric(games_vs_opp['MIN'], errors='coerce')
                avg_minutes = full_player_df['MIN'].apply(pd.to_numeric, errors='coerce').mean()
                min_minutes_threshold = avg_minutes * 0.6
                games_vs_opp = games_vs_opp[games_vs_opp['MIN_numeric'] >= min_minutes_threshold]

            if len(games_vs_opp) > 0:
                for stat in stat_cols:
                    if stat in games_vs_opp.columns:
                        h2h_adjustment[stat] = games_vs_opp[stat].mean()

                playoff_h2h = pd.DataFrame()
                if playoff_games_df is not None:
                    playoff_h2h = games_vs_opp[games_vs_opp.index.isin(playoff_games_df.index)]

                if len(playoff_h2h) >= 2:
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
        result['_po_blend_meta'] = _po_blend_meta  # None for non-playoff path
        
        return result
        
    except Exception as e:
        return None
