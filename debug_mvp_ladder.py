#!/usr/bin/env python3
"""
Debug script to check what the MVP ladder scraper is finding
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

def test_mvp_ladder_scrape():
    """Test the MVP ladder scraping for Feb 6, 2026"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    # Try to find the latest MVP ladder article
    today = datetime(2026, 2, 6)
    
    article_url = None
    article_date = None
    
    print("=== Searching for MVP Ladder Article ===\n")
    
    for days_back in range(14):
        check_date = today - timedelta(days=days_back)
        month_abbrev = check_date.strftime('%b').lower()
        day = check_date.day
        year = check_date.year
        url = f"https://www.nba.com/news/kia-mvp-ladder-{month_abbrev}-{day}-{year}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            has_ladder = "Kia MVP Ladder" in resp.text
            print(f"{check_date.strftime('%b %d')}: {resp.status_code} - {'✓ Found' if has_ladder else '✗ Not found'}")
            
            if resp.status_code == 200 and has_ladder:
                article_url = url
                article_date = check_date
                break
        except Exception as e:
            print(f"{check_date.strftime('%b %d')}: Error - {str(e)[:50]}")
            continue
    
    if not article_url:
        print("\n❌ No MVP ladder article found!")
        return
    
    print(f"\n✅ Found article: {article_url}")
    print(f"   Date: {article_date.strftime('%B %d, %Y')}\n")
    
    # Scrape the article
    print("=== Scraping Article ===\n")
    art_resp = requests.get(article_url, headers=headers, timeout=10)
    art_resp.raise_for_status()
    art_soup = BeautifulSoup(art_resp.text, 'html.parser')
    
    players = []
    
    # Parse TOP 5 from h2/h3/h4 headings
    print("Looking for Top 5 rankings in h2/h3/h4 tags:")
    rankings = art_soup.find_all(['h2', 'h3', 'h4'])
    found_ranks = set()
    
    for r in rankings:
        text = r.get_text(strip=True)
        # Use \s* to handle both "4. Name" and "4.Name" (no space after period)
        match = re.match(r'^(\d+)\.\s*(.+)', text)
        if match:
            rank_num = int(match.group(1))
            if rank_num <= 5 and rank_num not in found_ranks:
                rank = match.group(1)
                rest = match.group(2)
                
                # Validate
                if any(kw in rest.upper() for kw in ["THINGS TO KNOW", "DOUBLEHEADER", "WATCH", "SCENARIOS", "LATEST"]):
                    continue
                if len(rest) > 100:
                    continue
                
                # Split name and team
                if ',' in rest:
                    parts = rest.split(',', 1)
                    name = parts[0].strip()
                    team = parts[1].strip() if len(parts) > 1 else "N/A"
                else:
                    name = rest
                    team = "N/A"
                
                print(f"  #{rank}: {name} - {team}")
                players.append({'rank': rank, 'name': name, 'team': team})
                found_ranks.add(rank_num)
    
    # Parse "The Next 5" section
    print("\nLooking for 'The Next 5' section:")
    full_text = art_soup.get_text()
    
    next5_match = re.search(r'(?:the\s+)?next\s+5[:\s]*\n?(.*?)(?:and\s+five\s+more|$)', full_text, re.IGNORECASE | re.DOTALL)
    if next5_match:
        print("  Found 'The Next 5' section")
        next5_text = next5_match.group(1)
        
        # Show a preview of the text
        preview = next5_text[:500].replace('\n', ' ')
        print(f"  Preview: {preview}...")
        
        for line in next5_text.split('\n'):
            line = line.strip()
            # Match pattern: "6. Player Name, Team Name emoji" - use \s* for flexibility
            rank_match = re.match(r'^(\d+)\.?\s*([^,]+),\s*([^↔️⬆️⬇️↗️↘️\n]+)', line)
            if rank_match:
                rank = rank_match.group(1)
                name = rank_match.group(2).strip()
                team = rank_match.group(3).strip()
                
                print(f"  #{rank}: {name} - {team}")
                players.append({'rank': rank, 'name': name, 'team': team})
    else:
        print("  ❌ 'The Next 5' section not found")
    
    # Sort and display final results
    players.sort(key=lambda x: int(x['rank']))
    
    print("\n=== Final Parsed Rankings ===")
    for p in players:
        print(f"#{p['rank']}: {p['name']} ({p['team']})")
    
    # Check for missing ranks
    found_ranks = set(int(p['rank']) for p in players)
    missing_ranks = set(range(1, 11)) - found_ranks
    if missing_ranks:
        print(f"\n⚠️  Missing ranks: {sorted(missing_ranks)}")

if __name__ == "__main__":
    test_mvp_ladder_scrape()
