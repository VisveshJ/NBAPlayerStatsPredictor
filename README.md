# 🏀 NBA Player Stats Predictor

An advanced, full-stack analytics platform built to provide AI-driven performance forecasting using **Hidden Markov Models (HMM)**. This application bridges the gap between raw sports data and actionable insights through a modern, responsive dashboard.

![Home Page](screenshots/home.png)

## Core Features

### Personalized Dashboard
- **Google OAuth 2.0 Integration**: Secure login (PKCE flow) to save your personalized experience.
- **Favorites Management**: Track your favorite teams and players with a dedicated dashboard.
- **Dynamic Seeding & Records**: Real-time Western and Eastern Conference standings with color-coded streaks and detailed splits.

![Standings](screenshots/standings.png)

### AI-Powered Predictions
- **Hidden Markov Models**: Gaussian HMM identifies hidden performance states (Cold, Average, Hot) to forecast future stat lines.
- **Situational Context**: Integrates Opponent Defensive Ratings (DRTG) to scale predictions.
- **Consistency Analysis**: Calculates player volatility and performance reliability over the season.

![Predictions](screenshots/prediction.png)

### Comparative Analysis
- **Autocomplete Search**: Effortlessly find any active NBA player or team.
- **Head-to-Head (H2H) Filter**: Compare players specifically when they play against each other.
- **Visual Comparisons**: Aesthetic stat overlays and IND REC records.

### NBA Awards & Odds
- **Real-time Odds Scraper**: Live betting odds for MVP, DPOY, and ROTY scraped from DraftKings via Playwright.
- **MVP Ladder**: Comprehensive view of the current MVP race with live statistical resumes.
- **Auto-Refresh**: Background workers ensure odds stay fresh within a 12-hour cycle.

### Live Scoreboard & News
- **Today's Games**: Real-time scores, channel information, and one-click **Box Score** links for every game.

![Today's Games](screenshots/todays_games.png)

- **Playoff Picture**: Dynamic visualization of First Round and Play-In Tournament matchups.

![Playoff Picture](screenshots/playoffs.png)

- **News Wire**: Continuous scrolling news ticker and headline grid directly from NBA.com.

## Architecture & Design

The application follows a modular, state-driven architecture designed for high data throughput and real-time responsiveness.

### System Components
- **Data Acquisition Layer**: Orchestrates requests to the `nba-api` with robust caching and `Playwright` for automation.
- **AI Processing Engine**: A custom implementation of Gaussian HMM that maps observable statistics to hidden performance states.
- **Presentation Layer**: A premium dark-themed UI (Hex `#161B22`) utilizing custom CSS tokens and circle-free logo styling.
- **Persistence**: **SQLite** backend stores user preferences, favorite lists, and cached leadboard data.

## Tech Stack
- **Languages**: Python 3.11+
- **ML Frameworks**: `hmmlearn`, `scikit-learn`, `numpy`, `pandas`
- **Frontend**: `Streamlit`, `Vanilla CSS (Modern Dark Theme)`
- **Automation**: `Playwright` (Chromium Headless Shell)
- **Database**: `SQLite`
- **Auth**: `Google Cloud Console (OAuth 2.0 + PKCE)`

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
