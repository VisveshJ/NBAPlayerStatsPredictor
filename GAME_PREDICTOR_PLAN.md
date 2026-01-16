# Game Score Predictor - Implementation Plan

Predict final game scores by aggregating HMM player predictions from both rosters.

---

## 🎯 Concept

```
Team A Predicted Score = Σ (Each player's predicted points × minutes share)
                        + Bench contribution estimate
                        + Pace adjustment
```

---

## 📊 How It Works

### Step 1: Get Active Rosters
For each team, fetch the roster and filter out:
- **Injured players** (via NBA injury report API)
- **DNP/Rest** (recent patterns)
- **Suspended/Inactive**

### Step 2: Predict Each Player
Run HMM prediction for each active player:
```python
{
    "player": "LeBron James",
    "predicted_pts": 27.3,
    "predicted_min": 34.2,
    "confidence": 0.72
}
```

### Step 3: Aggregate Team Score
```python
team_score = sum(player_pts * (player_min / total_team_min))
           + bench_adjustment
           + pace_factor * (team_pace / league_avg_pace)
```

### Step 4: Display Prediction
```
┌─────────────────────────────────────────────────┐
│  🏀 Game Prediction: LAL @ BOS                  │
├─────────────────────────────────────────────────┤
│                                                 │
│   LAL Lakers        vs        BOS Celtics       │
│      112.4                      118.7           │
│                                                 │
│   ⚠️ LAL Missing: Anthony Davis (back)         │
│   ✅ BOS Full Healthy Roster                   │
│                                                 │
├─────────────────────────────────────────────────┤
│  Top Contributors (Predicted)                   │
│  ─────────────────────────────                  │
│  LAL: LeBron 27.3 | Austin 18.2 | Rui 12.4     │
│  BOS: Tatum 28.1 | Brown 24.2 | White 14.3     │
│                                                 │
│  Spread: BOS -6.3 | O/U: 231.1                 │
└─────────────────────────────────────────────────┘
```

---

## 🏥 Injury Integration

### Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| NBA Official Injury Report | GTD, Out, Doubtful | Daily 5pm ET |
| ESPN Injury API | Status, return date | Real-time |
| RotoBombs/Rotowire | Injury news | Real-time |

### Status Handling

| Status | Action |
|--------|--------|
| **Out** | Remove from prediction completely |
| **Doubtful** | 20% weight (likely out) |
| **Questionable** | 50% weight |
| **Probable** | 90% weight |
| **GTD (Game-Time Decision)** | 50% weight, flag for user |
| **Available** | 100% weight |

### Replacement Logic
When a star is out:
1. Identify their typical minutes (e.g., AD plays 34 mpg)
2. Distribute to backup (e.g., Jaxson Hayes +10 min)
3. Distribute remaining to rotation players
4. Adjust starting lineup predictions

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ get_team_roster │────▶│ filter_injuries  │
└─────────────────┘     └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ For each player: │
                        │ predict_with_drtg│
                        └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ aggregate_score  │
                        │ (weighted sum)   │
                        └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ Display with     │
                        │ confidence band  │
                        └──────────────────┘
```

---

## 📁 New Files

| File | Purpose |
|------|---------|
| `src/logic/team_predictor.py` | Team score aggregation logic |
| `src/logic/injury_tracker.py` | Fetch and parse injury reports |

---

## 📝 Code Outline

### `team_predictor.py`

```python
def predict_game_score(home_team: str, away_team: str, injuries: dict):
    """
    Predict final score for a game.
    
    Returns:
        {
            'home_score': 118.7,
            'away_score': 112.4,
            'home_players': [{'name': 'Tatum', 'pts': 28.1, 'min': 36}, ...],
            'away_players': [{'name': 'LeBron', 'pts': 27.3, 'min': 35}, ...],
            'home_injuries': ['None'],
            'away_injuries': ['Anthony Davis (back) - OUT'],
            'confidence': 0.68,
            'spread': -6.3,
            'total': 231.1
        }
    """
```

### `injury_tracker.py`

```python
def get_team_injuries(team_abbrev: str):
    """
    Fetch current injury report for a team.
    
    Returns:
        [
            {'player': 'Anthony Davis', 'status': 'Out', 'injury': 'back'},
            {'player': 'Gabe Vincent', 'status': 'Questionable', 'injury': 'knee'}
        ]
    """
```

---

## 🖥️ UI Addition

Add to **Today's Games** section:
- "Predict Score" button on each game card
- Expander with full breakdown

Or new tab: **"Game Predictions"**
- List all today's games with predicted scores
- Click to see player-by-player breakdown

---

## 📅 Implementation Phases

### Phase 1: Injury Tracking (2 hours)
- [ ] Research NBA injury API endpoints
- [ ] Build `injury_tracker.py`
- [ ] Cache injury data (refresh every 30 min)

### Phase 2: Roster Predictions (3 hours)
- [ ] Get team rosters via `commonteamroster`
- [ ] Filter by injury status
- [ ] Run HMM for each active player
- [ ] Handle players without enough data

### Phase 3: Score Aggregation (2 hours)
- [ ] Weighted sum by expected minutes
- [ ] Bench contribution estimation
- [ ] Pace adjustment factor
- [ ] Calculate spread and O/U

### Phase 4: UI Integration (2 hours)
- [ ] Add prediction button to game cards
- [ ] Show score prediction with breakdown
- [ ] Display injuries and confidence

---

## ⚠️ Challenges

1. **Player Data Gaps**: Rookies/new players may lack HMM training data
   - Solution: Use season averages as fallback

2. **Minutes Distribution**: Hard to predict exactly who plays how much
   - Solution: Use recent avg minutes, adjust for injury absences

3. **Injury Accuracy**: Reports update close to game time
   - Solution: Show "Last Updated" timestamp, allow refresh

4. **Pace Variance**: Games vs different teams have different paces
   - Solution: Factor in opponent pace, not just team pace

---

## ✅ Success Criteria

1. Predict scores within ±8 points of actual (65% of games)
2. Handle all injury statuses correctly
3. Sub-5 second prediction time
4. Clear UI showing player contributions
