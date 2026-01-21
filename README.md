# 🏀 NBA Player Stats Predictor

An advanced, full-stack analytics platform built to provide AI-driven performance forecasting using **Hidden Markov Models (HMM)**. This application bridges the gap between raw sports data and actionable insights through a modern, responsive dashboard.

![Home Page](screenshots/home.png)

## Core Features

### Personalized Dashboard
- **Google OAuth 2.0 Integration**: Secure login (PKCE flow) to save your personalized experience.
- **Favorites Management**: Track your favorite teams and players with a dedicated dashboard.
- **Dynamic Seeding & Records**: Real-time Western and Eastern Conference standings with color-coded streaks and detailed splits.

![Favorites](screenshots/favorites.png)

### Around the NBA
- **Real-time News Wire**: Continuous scrolling news ticker and headline grid directly from NBA.com.
- **Featured Stories**: Stay updated with the latest major headlines across the league.

![Around the NBA](screenshots/aroundnba.png)

### Today's Games
- **Scoreboard**: Real-time scores, channel information, and one-click **Box Score** links for every game on the slate.

![Today's Games](screenshots/today_games.png)

### Player Stats
- **Detailed History**: Comprehensive player game logs and biographical information.
- **Visual Analytics**: Interactive charts showing season trends and historical performance.

![Player Stats](screenshots/playerstats.png)

### AI-Powered Predictions
- **Hidden Markov Models**: Utilizes Gaussian HMM to identify player performance states (Cold, Average, Hot) and predict future stat lines.
- **Situational Context**: Integrates Opponent Defensive Ratings (DRTG) to scale predictions based on matchup difficulty.

![Predictions](screenshots/prediction.png)

### Compare Players
- **Head-to-Head (H2H)**: Compare players specifically when they play against each other, including historical matchup logs and IND REC.
- **Visual Overlays**: Side-by-side statistical comparisons for season averages and situational performance.

![Compare Players](screenshots/compareplayers.png)

### NBA Standings
- **Conference & Division Standings**: Detailed view of both conferences and all six divisions.
- **Color-Coded Analysis**: Instantly identify playoff seeds, play-in spots, and performance streaks.

![Standings](screenshots/standings.png)

### Play-In Tournament
- **Live Matchups**: Real-time visualization of the 7-10 seed play-in scenarios for both conferences.

![Play-In Tournament](screenshots/playin.png)

### Playoff Picture
- **Dynamic Visualization**: Live look at the First Round and Play-In Tournament matchups if the season ended today.

![Playoff Picture](screenshots/playoffs.png)

### NBA Awards & Odds
- **Real-time Odds Scraper**: Live betting odds for DPOY, 6MOY and more, scraped via Playwright.
- **MVP/ROY Ladder**: Comprehensive view of the current MVP race with live statistical resumes.

![Awards](screenshots/awards.png)

---

## Architecture & Design

The application follows a modular, state-driven architecture designed for high data throughput and real-time responsiveness.

### System Components
- **Data Acquisition Layer**: Orchestrates requests to the `nba-api` with robust caching (`st.cache_data`) and `Playwright` for browser automation.
- **AI Processing Engine**: A custom implementation of Gaussian HMM that maps observable statistics to hidden performance states.
- **Presentation Layer**: A premium dark-themed UI (Hex `#161B22`) utilizing custom CSS tokens and circle-free logo styling.
- **Persistence**: **SQLite** backend stores user preferences, favorite lists, and cached leaderboard data.

## Predictive Modeling (The HMM Approach)
Unlike simple regressions, this app treats a player's season as a sequence of transitions between internal performance states.
- **State Selection**: Automatically identifies hidden states from recent game sequences.
- **Situational Adjustments**: 
    - **Defensive Weighting**: Predictions are scaled based on the target team's Defensive Rating.
    - **H2H Integration**: Matches against specific opponents are blended with the model output (weighted up to 40%).
    - **Injury Filtration**: Excludes "noise" games (low minutes) to maintain model integrity.

## Tech Stack
- **Languages**: Python 3.11+
- **ML Frameworks**: `hmmlearn`, `scikit-learn`, `numpy`, `pandas`
- **Frontend**: `Streamlit`, `Vanilla CSS (Modern Dark Theme)`
- **Automation**: `Playwright` (Chromium Headless Shell)
- **Database**: `SQLite`
- **Auth**: `Google Cloud Console (OAuth 2.0 + PKCE)`

---

## Getting Started

### Prerequisites
For Linux/Cloud environments (like Streamlit Cloud), ensure the system libraries in `packages.txt` are installed to support Playwright.

### Installation
```bash
# 1. Clone & Enter
git clone https://github.com/VisveshJ/NBAPlayerStatsPredictor.git
cd NBAPlayerStatsPredictor

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Initialize Playwright (Required for Odds Scraper)
playwright install chromium

# 4. Launch Application
streamlit run app.py
```

### Configuration
Update `.streamlit/secrets.toml` with your Google Cloud credentials for OAuth functionality.

## License
This project is for educational and personal use. Special thanks to the `nba-api` contributors and the Streamlit community.
