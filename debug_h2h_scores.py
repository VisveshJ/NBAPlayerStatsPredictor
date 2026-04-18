
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import time

def get_team_game_log(team_abbrev, season="2025-26"):
    all_teams = teams.get_teams()
    team = [t for t in all_teams if t['abbreviation'] == team_abbrev]
    if not team:
        return None
    team_id = team[0]['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=10
    )
    return gamefinder.get_data_frames()[0]

def test():
    season = "2025-26"
    for team in ["MIN", "CLE"]:
        print(f"\nChecking games for {team} in {season}...")
        df = get_team_game_log(team, season)
        if df is not None:
            print(f"Found {len(df)} games.")
            # Filter for Jan 8 and Jan 10
            target_dates = ['2026-01-08', '2026-01-10']
            subset = df[df['GAME_DATE'].isin(target_dates)]
            if not subset.empty:
                print(subset[['GAME_DATE', 'GAME_ID', 'MATCHUP', 'PTS', 'PLUS_MINUS']])
            else:
                print("No games found for Jan 8 or Jan 10.")
                # Show first 5 games to see date format
                print("First 5 games:")
                print(df[['GAME_DATE', 'GAME_ID', 'MATCHUP']].head())
        else:
            print(f"Failed to fetch data for {team}")
        time.sleep(1)

if __name__ == "__main__":
    test()
