#!/usr/bin/env python3
"""
Test the actual ROY ladder scraping to debug the issue.
"""

import requests
from datetime import datetime, timedelta

def test_scraping():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Test the exact URL the user says should work
    test_url = "https://www.nba.com/news/kia-rookie-ladder-feb-4-2026"
    
    print(f"Testing URL: {test_url}\n")
    
    try:
        resp = requests.get(test_url, headers=headers, timeout=10)
        print(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            print(f"Page loaded successfully!")
            
            # Check if it contains expected content
            if "Rookie Ladder" in resp.text:
                print("✓ Contains 'Rookie Ladder' text")
            else:
                print("✗ Does NOT contain 'Rookie Ladder' text")
                
            if "Rookie of the Year" in resp.text:
                print("✓ Contains 'Rookie of the Year' text")
            else:
                print("✗ Does NOT contain 'Rookie of the Year' text")
                
            # Check for player names
            if "Knueppel" in resp.text:
                print("✓ Contains 'Knueppel'")
            if "Flagg" in resp.text:
                print("✓ Contains 'Flagg'")
                
            print(f"\nFirst 500 characters of response:")
            print(resp.text[:500])
        else:
            print(f"Failed to load page")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Now testing the date search logic:")
    print("="*60 + "\n")
    
    # Test the backward search logic
    today = datetime(2026, 2, 5)  # Current date
    
    for days_back in range(14):
        check_date = today - timedelta(days=days_back)
        month_abbrev = check_date.strftime('%b').lower()
        day = check_date.day
        year = check_date.year
        url = f"https://www.nba.com/news/kia-rookie-ladder-{month_abbrev}-{day}-{year}"
        
        try:
            resp = requests.head(url, headers=headers, timeout=5)
            status = resp.status_code
            marker = " ✓ EXISTS" if status == 200 else ""
            print(f"{check_date.strftime('%b %d')}: HTTP {status}{marker}")
            
            # If we found a 200, check the content
            if status == 200 and days_back < 5:
                resp_get = requests.get(url, headers=headers, timeout=5)
                if "Rookie Ladder" in resp_get.text:
                    print(f"         → Contains 'Rookie Ladder' content! This should be used.")
                    break
        except Exception as e:
            print(f"{check_date.strftime('%b %d')}: Error - {str(e)[:50]}")

if __name__ == "__main__":
    test_scraping()
