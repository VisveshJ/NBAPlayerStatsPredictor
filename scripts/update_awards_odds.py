
import json
import os
import time
import argparse
from playwright.sync_api import sync_playwright

# Configuration
# Path relative to project root (where the script is expected to be run)
OUTPUT_FILE = "data/awards_odds.json"
DRAFTKINGS_URL = "https://sportsbook.draftkings.com/leagues/basketball/nba?category=awards&subcategory=mvp"

def scrape_dk_odds():
    print("🏀 Starting NBA Awards Odds Scraper...")
    
    awards_data = []
    
    # Map of tab names to Award Names
    awards_to_scrape = [
        {"tab": "Rookie of the Year", "name": "Rookie of the Year"},
        {"tab": "Defensive Player of the Year", "name": "Defensive Player of the Year"},
        {"tab": "Most Improved Player", "name": "Most Improved Player"},
        {"tab": "Sixth Man of the Year", "name": "Sixth Man of the Year"},
        {"tab": "Coach of the Year", "name": "Coach of the Year"}
    ]
    
    with sync_playwright() as p:
        # Launch browser (headless=True for background run)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print(f"Loading {DRAFTKINGS_URL}...")
        try:
            page.goto(DRAFTKINGS_URL, timeout=30000)
            
            # Wait for potential subcategory nav
            page.wait_for_selector('div.subcategory-navigation', timeout=10000)
        except Exception as e:
            print(f"Error loading page or navigation: {e}")
            browser.close()
            return

        for award in awards_to_scrape:
            print(f"Fetching: {award['name']}...")
            
            try:
                # Click the tab
                tab = page.get_by_text(award["tab"], exact=False).first
                if tab and tab.is_visible():
                    tab.click()
                    time.sleep(3) # Wait for content load
                    
                    candidates = []
                    
                    # Locate outcome cells
                    outcomes = page.query_selector_all('.sportsbook-outcome-cell__body')
                    
                    for outcome in outcomes[:10]: # Top 10
                         try:
                             label_el = outcome.query_selector('.sportsbook-outcome-cell__label')
                             odds_el = outcome.query_selector('.sportsbook-outcome-cell__elements')
                             
                             if label_el and odds_el:
                                 player_name = label_el.inner_text().strip()
                                 odds = odds_el.inner_text().strip()
                                 candidates.append({"player": player_name, "odds": odds})
                         except:
                             continue
                    
                    if candidates:
                        awards_data.append({
                            "award_name": award["name"],
                            "candidates": candidates
                        })
                        print(f"  Found {len(candidates)} candidates.")
                    else:
                        print("  No candidates found (selectors might be wrong).")
                else:
                    print(f"  Tab '{award['tab']}' not found.")
                    
            except Exception as e:
                print(f"  Error scraping {award['name']}: {e}")
        
        browser.close()
    
    # Save to JSON
    if awards_data:
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(awards_data, f, indent=2)
        print(f"✅ Successfully saved odds for {len(awards_data)} awards to {OUTPUT_FILE}")
    else:
        print("❌ No data collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NBA Awards Odds from DraftKings")
    parser.add_argument("--force", action="store_true", help="Force scrape regardless of last update time")
    args = parser.parse_args()
    
    # Check if data is fresh (3-day cycle)
    if not args.force and os.path.exists(OUTPUT_FILE):
        mtime = os.path.getmtime(OUTPUT_FILE)
        age_days = (time.time() - mtime) / (24 * 3600)
        # Update if older than 12 hours (0.5 days)
        if age_days < 0.5:
            print(f"ℹ️ Odds data is only {age_days:.1f} days old. Skipping scrape (12-hour cycle).")
            print("Use --force to override.")
            exit(0)
            
    scrape_dk_odds()

