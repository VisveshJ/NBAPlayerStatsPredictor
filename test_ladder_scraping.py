#!/usr/bin/env python3
"""
Test script to verify the date calculation for the ROY ladder scraping.
This demonstrates what URLs would be checked starting from today.
"""

from datetime import datetime, timedelta

def check_dates():
    # Use a fixed date for demonstration
    today = datetime(2026, 2, 5)  # Feb 5, 2026 (current date based on user info)
    
    print(f"=== ROY Ladder URL Search Test ===")
    print(f"Today's date: {today.strftime('%B %d, %Y')}\n")
    print("URLs that will be checked (in order):")
    
    for days_back in range(14):
        check_date = today - timedelta(days=days_back)
        month_abbrev = check_date.strftime('%b').lower()
        day = check_date.day
        year = check_date.year
        url = f"https://www.nba.com/news/kia-rookie-ladder-{month_abbrev}-{day}-{year}"
        
        # Highlight Feb 3 (expected article date)
        marker = " ← EXPECTED ARTICLE" if check_date.day == 3 and check_date.month == 2 else ""
        print(f"  {days_back+1:2d}. {check_date.strftime('%b %d, %Y')} -> {url}{marker}")

if __name__ == "__main__":
    check_dates()
