# Project Instructions: NBA Player Stats Predictor

Please refer to [.agent/spec.md](.agent/spec.md) for the full architectural specification and project memory.

## Development Priorities
1. **Maintain AI Accuracy**: Always verify HMM transition matrices when modifying the prediction engine.
2. **Visual Excellence**: Adhere to the Premium Dark Theme. Use `st.markdown` with `unsafe_allow_html=True` for complex branding.
3. **Data Integrity**: Ensure `nba-api` calls are wrapped in robust exception handling and caching.
4. **Scraper Maintenance**: The `scripts/update_awards_odds.py` is sensitive to DraftKings UI changes. Verify and update CSS selectors immediately if a "No Candidates Found" error occurs.

## Key Files
- `app.py`: Entry point and UI routing.
- `src/logic/model.py`: HMM implementation.
- `src/auth/google_oauth.py`: Identity management.
- `scripts/update_awards_odds.py`: Browser automation.
