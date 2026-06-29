import sqlite3
import os
import sys
import time
import shutil
import requests
import urllib3
from fanduel_odds import american_to_decimal

# Disable SSL verification warnings since OddsShark uses a *.chalk247.com certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_PATH = "wnba.db"
FRONTEND_DB_PATH = "frontend/wnba.db"

def parse_american_odds(odds_str):
    if odds_str is None:
        return None
    odds_str = str(odds_str).strip()
    if odds_str == "" or odds_str.lower() in ("null", "none", "-", "pk"):
        return None
    if odds_str.lower() == "even":
        return 100
    try:
        return int(odds_str)
    except ValueError:
        try:
            return int(float(odds_str))
        except ValueError:
            return None

def parse_spread(spread_str):
    if spread_str is None:
        return None
    spread_str = str(spread_str).strip()
    if spread_str.lower() in ("pk", "pick", "even"):
        return 0.0
    if spread_str == "" or spread_str.lower() in ("null", "none", "-"):
        return None
    try:
        return float(spread_str)
    except ValueError:
        return None

def parse_total(total_str):
    if total_str is None:
        return None
    total_str = str(total_str).strip()
    if total_str == "" or total_str.lower() in ("null", "none", "-"):
        return None
    try:
        return float(total_str)
    except ValueError:
        return None

def fetch_date_odds(date):
    url = f"https://io.oddsshark.com/scores/wnba/{date}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.oddsshark.com/'
    }
    
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=10)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    print(f"[{date}] Failed to decode JSON response.")
                    return None
            elif r.status_code == 429:
                print(f"[{date}] Rate limited (429). Sleeping before retry...")
                time.sleep(2.0 * (attempt + 1))
            elif r.status_code == 404:
                # Some dates might not have WNBA games at all or return 404
                return []
            else:
                print(f"[{date}] Received status code {r.status_code}.")
                time.sleep(1.0)
        except Exception as e:
            print(f"[{date}] Request failed: {e}")
            time.sleep(1.0)
    return None

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database {DB_PATH} not found.")
        sys.exit(1)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if IsFanduelOdds column exists
    cursor.execute("PRAGMA table_info(raw_matches);")
    columns = [c[1] for c in cursor.fetchall()]
    if "IsFanduelOdds" not in columns:
        print("Adding IsFanduelOdds column to raw_matches table...")
        cursor.execute("ALTER TABLE raw_matches ADD COLUMN IsFanduelOdds INTEGER DEFAULT 0;")
        conn.commit()
        
    # Query distinct dates where we haven't scraped/saved real odds yet (IsFanduelOdds = 0)
    cursor.execute("SELECT DISTINCT Date FROM raw_matches WHERE IsFanduelOdds = 0 ORDER BY Date;")
    dates = [r[0] for r in cursor.fetchall()]
    
    total_dates = len(dates)
    print(f"Found {total_dates} unique dates to check/scrape.")
    if total_dates == 0:
        print("All matches already have scraped/real odds. Nothing to do!")
        conn.close()
        return

    updated_games_count = 0
    scraped_dates_count = 0
    
    try:
        for idx, date in enumerate(dates):
            print(f"[{idx+1}/{total_dates}] Fetching odds for {date}...", end="", flush=True)
            games = fetch_date_odds(date)
            
            if games is None:
                print(" failed.")
                continue
                
            scraped_dates_count += 1
            if not games:
                print(" no games found.")
                continue
                
            print(f" parsed {len(games)} games.", end="")
            
            daily_updates = 0
            for g in games:
                home_name = g.get('home_name')
                away_name = g.get('away_name')
                if not home_name or not away_name:
                    continue
                
                # Parse lines
                home_ml = parse_american_odds(g.get('home_money_line'))
                away_ml = parse_american_odds(g.get('away_money_line'))
                home_spread = parse_spread(g.get('home_spread'))
                total = parse_total(g.get('total'))
                
                # Convert money line to decimal odds
                home_odds = american_to_decimal(home_ml) if home_ml is not None else None
                away_odds = american_to_decimal(away_ml) if away_ml is not None else None
                
                # Only update if we parsed some valid betting data
                if home_odds is None and away_odds is None and home_spread is None and total is None:
                    continue
                
                # Update DB row
                cursor.execute("""
                    UPDATE raw_matches
                    SET BookieHomeOdds = COALESCE(?, BookieHomeOdds),
                        BookieAwayOdds = COALESCE(?, BookieAwayOdds),
                        OpeningSpread = COALESCE(?, OpeningSpread),
                        ClosingSpread = COALESCE(?, ClosingSpread),
                        OverUnder = COALESCE(?, OverUnder),
                        IsFanduelOdds = 2
                    WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?
                """, (home_odds, away_odds, home_spread, home_spread, total, date, home_name, away_name))
                
                if cursor.rowcount > 0:
                    daily_updates += 1
                    updated_games_count += 1
                    
            print(f" Updated {daily_updates} matches in DB.")
            conn.commit()
            
            # Rate limit politeness
            time.sleep(0.25)
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Saving progress...")
    finally:
        conn.commit()
        conn.close()
        
    print(f"\nDone! Scraped {scraped_dates_count} dates, updated {updated_games_count} match records with real OddsShark closing odds.")
    
    # Sync database to the frontend directory
    if os.path.exists(DB_PATH):
        print(f"Copying updated {DB_PATH} to {FRONTEND_DB_PATH}...")
        shutil.copy(DB_PATH, FRONTEND_DB_PATH)
        print("Database sync complete.")

if __name__ == "__main__":
    main()
