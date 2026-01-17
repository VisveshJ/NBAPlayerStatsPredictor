import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from nba_api.stats.static import teams

def normalize_coach_name(name):
    """Normalize coach names (e.g., 'J.B. Bickerstaff' -> 'JB Bickerstaff')."""
    # Specifically requested: remove dots
    name = name.replace('.', '').strip()
    return name

def get_team_abbr_map():
    """Get a mapping of full team names to abbreviations using nba_api."""
    nba_teams = teams.get_teams()
    abbr_map = {}
    for team in nba_teams:
        abbr_map[team['full_name']] = team['abbreviation']
    
    # Custom mappings for names that might vary or special cases
    abbr_map['LA Clippers'] = 'LAC'
    abbr_map['Philadelphia 76ers'] = 'PHI'
    
    return abbr_map

def scrape_espn_coaches():
    url = "https://www.espn.com/nba/coaches"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch ESPN page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    coaches_data = []
    abbr_map = get_team_abbr_map()
    
    # Looking for table.tablehead and rows with oddrow/evenrow classes
    rows = soup.find_all('tr', class_=['oddrow', 'evenrow', 'Table__TR'])
    
    if not rows:
        # Fallback search
        rows = soup.find_all('tr')

    for row in rows:
        cols = row.find_all(['td', 'font'])
        if len(cols) >= 4:
            # Based on inspection: 
            # cols[0] = Name
            # cols[1] = Exp
            # cols[2] = 2026 Record
            # cols[3] = Team
            
            coach_name = cols[0].get_text(strip=True)
            exp = cols[1].get_text(strip=True)
            record = cols[2].get_text(strip=True)
            team_name = cols[3].get_text(strip=True)
            
            # Skip header rows
            if coach_name.lower() == 'name' or team_name.lower() == 'team':
                continue
            
            # Additional check for data rows - Name shouldn't be empty or header-like
            if not coach_name or coach_name == 'NAME':
                continue
                
            norm_name = normalize_coach_name(coach_name)
            
            # Derive abbreviation
            abbr = abbr_map.get(team_name, "N/A")
            
            # If not found, try partial match
            if abbr == "N/A":
                if "Clippers" in team_name:
                    abbr = "LAC"
                elif "Lakers" in team_name:
                    abbr = "LAL"
                elif "Thunder" in team_name:
                    abbr = "OKC"
                else:
                    # Search keys
                    for full_name, a in abbr_map.items():
                        if team_name in full_name or full_name in team_name:
                            abbr = a
                            break
            
            coaches_data.append({
                'NAME': norm_name,
                '2026 RECORD': record,
                'TEAM': team_name,
                'ABBREV': abbr
            })
            
    return coaches_data

def update_csv():
    print("Scraping ESPN coaches with refined logic...")
    new_coaches = scrape_espn_coaches()
    
    if not new_coaches:
        print("No coaches found. Aborting update.")
        return

    csv_path = 'nbacoaches.csv'
    df = pd.DataFrame(new_coaches)
    
    # Basic validation: we expect around 30 coaches
    if len(df) < 25:
        print(f"Warning: Only found {len(df)} coaches. ESPN layout might have changed.")

    df.to_csv(csv_path, index=False)
    print(f"Successfully updated {csv_path} with {len(df)} coaches.")

if __name__ == "__main__":
    update_csv()
