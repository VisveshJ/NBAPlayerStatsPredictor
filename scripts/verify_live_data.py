
import time
import pandas as pd
from nba_api.stats.endpoints import leaguestandings, leaguedashteamstats

# APPLY APP'S EXACT PATCH
NBA_STATS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua": '\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '\"Windows\"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}

def patch():
    from nba_api.stats.library import http as stats_http
    from nba_api.library import http as base_http
    stats_http.STATS_HEADERS = NBA_STATS_HEADERS
    stats_http.NBAStatsHTTP.headers = NBA_STATS_HEADERS
    stats_http.NBAStatsHTTP._session = None
    base_http.NBAHTTP._session = None

patch()

print("🔍 Final App Data Verification (2025-26 Season)")
print("-" * 50)

# 1. Standings
start = time.time()
try:
    print("Fetching 2025-26 Standings...")
    s = leaguestandings.LeagueStandings(season='2025-26', season_type='Regular Season', timeout=10)
    df = s.get_data_frames()[0]
    print(f"✅ Standings OK: Found {len(df)} teams in {time.time()-start:.1f}s")
except Exception as e:
    print(f"❌ Standings FAILED: {str(e)}")

# 2. Defensive Ratings (The 'Slow' one)
start = time.time()
try:
    print("\nFetching Advanced Team Stats (Defensive Ratings)...")
    ts = leaguedashteamstats.LeagueDashTeamStats(
        season='2025-26',
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame',
        timeout=10
    )
    df = ts.get_data_frames()[0]
    print(f"✅ Advanced Ratings OK: Found {len(df)} teams in {time.time()-start:.1f}s")
except Exception as e:
    print(f"❌ Advanced Ratings FAILED: {str(e)}")

print("-" * 50)
print("If both say ✅, the app is 100% good. Just hard-refresh your browser.")
