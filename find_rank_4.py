#!/usr/bin/env python3
"""
Deep dive into the MVP ladder article HTML to find rank #4
"""
import requests
from bs4 import BeautifulSoup
import re

def find_rank_4():
    """Search for rank #4 in the MVP ladder article"""
    
    url = "https://www.nba.com/news/kia-mvp-ladder-feb-6-2026"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    }
    
    print(f"Fetching: {url}\n")
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Get full text
    full_text = soup.get_text()
    
    # Search for "4." in the text
    print("=== Searching for '4.' in full text ===\n")
    lines = full_text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('4.'):
            print(f"Line {i}: {line.strip()}")
            # Show context (3 lines before and after)
            print("\nContext:")
            for j in range(max(0, i-3), min(len(lines), i+4)):
                marker = " >>> " if j == i else "     "
                print(f"{marker}{lines[j]}")
            print()
    
    # Search in all HTML elements containing "4."
    print("\n=== Searching for elements containing '4.' ===\n")
    
    # Check all h tags
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        elements = soup.find_all(tag)
        for elem in elements:
            text = elem.get_text(strip=True)
            if re.match(r'^4\.', text):
                print(f"Found in <{tag}>: {text}")
    
    # Check all p tags
    p_tags = soup.find_all('p')
    for p in p_tags:
        text = p.get_text(strip=True)
        if text.startswith('4.'):
            print(f"Found in <p>: {text}")
    
    # Check all divs
    div_tags = soup.find_all('div')
    for div in div_tags:
        text = div.get_text(strip=True)
        if text.startswith('4.') and len(text) < 150:  # Avoid huge divs
            print(f"Found in <div>: {text[:200]}")
    
    # Search for patterns like "4. Name, Team"
    print("\n=== Searching for pattern '4. <name>, <team>' ===\n")
    pattern = re.compile(r'4\.\s+([A-Za-zÀ-ÿ\s\'\-\.]+),\s+([A-Za-z\s]+(?:Thunder|Lakers|Nuggets|Spurs|Celtics|Cavaliers|Timberwolves|Knicks|76ers|Mavericks|Warriors|Heat|Nets))')
    matches = pattern.findall(full_text)
    for match in matches:
        print(f"  Player: {match[0].strip()}")
        print(f"  Team: {match[1].strip()}")

if __name__ == "__main__":
    find_rank_4()
