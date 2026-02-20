#!/usr/bin/env python3
"""Quick health check for NBA API endpoints used by the app."""
import time
import requests

ENDPOINTS = [
    ("NBA Stats API (standings)", "https://stats.nba.com/stats/leaguestandingsv3?Season=2025-26&SeasonType=Regular+Season"),
    ("NBA CDN (scoreboard)", "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"),
    ("NBA CDN (schedule)", "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}

print("🏀 NBA API Health Check")
print("=" * 50)

all_ok = True
for name, url in ENDPOINTS:
    start = time.time()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        elapsed = time.time() - start
        if resp.status_code == 200:
            print(f"  ✅ {name}: OK ({elapsed:.1f}s)")
        else:
            print(f"  ⚠️  {name}: HTTP {resp.status_code} ({elapsed:.1f}s)")
            all_ok = False
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  ❌ {name}: TIMEOUT ({elapsed:.1f}s)")
        all_ok = False
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ {name}: {type(e).__name__} ({elapsed:.1f}s)")
        all_ok = False

print("=" * 50)
if all_ok:
    print("✅ All endpoints are UP — your app should work fine!")
    print("   If your app still shows stale data, click 'Refresh Data' in the sidebar.")
else:
    print("❌ Some endpoints are DOWN — the app will use fallback/cached data.")
    print("   The NBA Stats API has occasional outages. Usually resolves within 1-2 hours.")
    print("   Your app will automatically recover once the API is back up.")
