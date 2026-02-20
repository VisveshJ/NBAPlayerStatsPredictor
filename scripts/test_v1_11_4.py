
import time
from nba_api.stats.endpoints import leaguestandings

print("🚀 Testing nba_api v1.11.4 (Official Fix)")
print("-" * 50)

# NO MONKEY PATCH HERE
start = time.time()
try:
    print("Attempting to fetch 2024-25 Standings (v1.11.4 default headers)...")
    s = leaguestandings.LeagueStandings(season='2024-25', timeout=15)
    df = s.get_data_frames()[0]
    print(f"✅ SUCCESS! Found {len(df)} teams in {time.time()-start:.1f}s")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")

print("-" * 50)
if 'df' in locals():
    print("Conclusion: v1.11.4 official headers work! We can remove the monkey patch.")
else:
    print("Conclusion: Even v1.11.4 is failing. We might still need our custom patch or the NBA is having a real outage.")
