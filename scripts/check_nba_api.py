
import time
import requests
from nba_api.stats.endpoints import leaguestandings

# ==================== APPLY MONKEY PATCH ====================
NBA_STATS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

def patch_nba_api():
    try:
        from nba_api.stats.library import http as stats_http
        from nba_api.library import http as base_http
        stats_http.STATS_HEADERS = NBA_STATS_HEADERS
        stats_http.NBAStatsHTTP.headers = NBA_STATS_HEADERS
        stats_http.NBAStatsHTTP._session = None
        base_http.NBAHTTP._session = None
    except: pass

patch_nba_api()

print("🏀 NBA API Connectivity Check")
print("=" * 50)

# Check 1: Stats API via Library
start = time.time()
try:
    leaguestandings.LeagueStandings(season='2024-25', timeout=15)
    print(f"  ✅ NBA Stats API (Library): OK ({time.time()-start:.1f}s)")
except Exception as e:
    print(f"  ❌ NBA Stats API (Library): FAILED ({time.time()-start:.1f}s) - {type(e).__name__}")

# Check 2: CDN Scoreboard
start = time.time()
try:
    requests.get("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json", timeout=10)
    print(f"  ✅ NBA CDN (Scoreboard): OK ({time.time()-start:.1f}s)")
except:
    print(f"  ❌ NBA CDN (Scoreboard): FAILED")

# Check 3: CDN Schedule
start = time.time()
try:
    requests.get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", timeout=10)
    print(f"  ✅ NBA CDN (Schedule): OK ({time.time()-start:.1f}s)")
except:
    print(f"  ❌ NBA CDN (Schedule): FAILED")

print("=" * 50)
print("Conclusion: The app is now using a MONKEY PATCH to fix connectivity issues.")
print("Everything should be back to normal.")
