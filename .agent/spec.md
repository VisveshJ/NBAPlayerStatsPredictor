# NBA Player Stats Predictor - Agent Specification

This document serves as the core specification and cognitive memory for the Antigravity AI Agent. It details the architecture, logic, and operational guidelines for maintaining and evolving the NBA Player Stats Predictor.

## 🚀 Overview
The NBA Player Stats Predictor is a full-stack Streamlit application that provides AI-driven performance forecasting, real-time NBA data visualization, and personalized user features.

## 🏗️ Technical Architecture
- **Frontend**: Streamlit with custom CSS (Dark Theme, Premium Aesthetics).
- **Core Logic**: `Gaussian Hidden Markov Models (HMM)` for stat prediction.
- **Data Integration**: 
  - `nba-api` for historical and live game data.
  - `Playwright` for real-time betting odds scraping (DraftKings).
- **Authentication**: Google OAuth 2.0 (PKCE) for personalized user states.
- **Persistence**: SQLite database (`user_data.db`) for tracking favorite players and teams.

## 🧠 Core Prediction Logic (HMM)
1. **State Training**: Uses `hmmlearn` to identify hidden performance states (e.g., Cold, Average, Hot).
2. **Features**: Trained on Points, Assists, Rebounds, Steals, and Blocks.
3. **Adjustments**:
   - **Defensive Rating**: Scaled against the league average to adjust raw predictions.
   - **H2H Factor**: Historical matchups against a specific team weighted into the output.
   - **Consistency Filtering**: Player variance (CV) calculation to interpret prediction reliability.

## 📝 Lessons Learned & Edge Cases
- **Scraping Reliability**: DraftKings often updates its DOM. Use specific selectors like `.cb-market__button-title` and capture `stderr/stdout` for failed subprocess runs to debug browser issues.
- **Environment Management**: Playwright requires browser installation (`uv run playwright install chromium`) in the host environment (Streamlit Cloud or local).
- **Session State Persistence**: Essential for interactive components. When users search for a player, session state must be cleared/reset properly to avoid data bleeding between selections.
- **Timezone Handling**: Always localized to the user's selected timezone (default: US/Pacific) using `pytz` for all "Last Updated" timestamps.
- **Data Rate Limits**: Use `@st.cache_data` aggressively to avoid `nba-api` timeout issues.

## 🎨 UI/UX Guidelines
- **Premium Aesthetics**: Use dark mode, `#FF6B35` (orange) for highlights, and `#2D3748` for container backgrounds.
- **Standardized Layouts**: Player photos (150px) and team logos (120px) should be centered and balanced.
- **Dynamic Feedback**: Use `st.spinner` and `st.toast` for asynchronous tasks like model training or odds refreshing.

## 🛠️ Future Roadmap
- [ ] Integration of injury reports and starting lineups.
- [ ] Multi-market odds comparison beyond DraftKings.

## 📜 Operational Rules for the Agent
1. **Absolute Paths**: Always use absolute paths for file operations.
2. **Safe Command Execution**: Always check for environment prerequisites (like browser installs) before running scraper scripts.
3. **Atomic Commits**: Group logic changes with documentation updates (e.g., updating `data/awards_odds.json` alongside its processor).
4. **Style Consistency**: Maintain the "Premium" visual standard. Do not use default Streamlit styling; use the established CSS tokens in `apply_dark_theme()`.
