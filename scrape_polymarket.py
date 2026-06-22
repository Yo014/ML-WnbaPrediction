import sys
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
# Constants
DB_NAME = "wnba.db"
TEAM_MAP = {
    'Indiana Fever': 'IND',
    'Chicago Sky': 'CHI',
    'Las Vegas Aces': 'LVA',
    'New York Liberty': 'NYL',
    'Seattle Storm': 'SEA',
    'Minnesota Lynx': 'MIN',
    'Phoenix Mercury': 'PHO',
    'Dallas Wings': 'DAL',
    'Atlanta Dream': 'ATL',
    'Connecticut Sun': 'CON',
    'Los Angeles Sparks': 'LAS',
    'Washington Mystics': 'WAS',
    'Golden State Valkyries': 'GSV',
    'Toronto Tempo': 'TOR',
    'Portland Fire': 'POR'
}
def safe_json_loads(val):
    """
    Safely loads a JSON string or returns the value if already a list/dict.
    """
    if not val:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []
def parse_match_date(end_date_str):
    """
    Parses a UTC ISO date string and returns the date in local game time (America/New_York) YYYY-MM-DD format.
    """
    if not end_date_str:
        return None
    try:
        # Convert Z to +00:00 for python fromisoformat
        dt_utc = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        dt_est = dt_utc.astimezone(ZoneInfo('America/New_York'))
        return dt_est.strftime('%Y-%m-%d')
    except Exception as e:
        # Fallback to simple date parsing if ISO parsing fails
        try:
            return end_date_str.split('T')[0]
        except Exception:
            return None
def parse_market(market):
    """
    Parses a single market object from Polymarket's Gamma API.
    Returns a dict with processed fields if it matches a valid WNBA matchup, or None.
    """
    question = market.get('question', '')
    if not question:
        return None
        
    # Standard WNBA game match-ups look like: "Away Team vs. Home Team"
    # Or "Away Team vs Home Team", or sometimes "Away Team @ Home Team"
    q_normalized = " ".join(question.split())
    
    away_team_full = None
    home_team_full = None
    
    def clean_str(s):
        return "".join(c for c in s.lower() if c.isalnum())
        
    for sep in [" vs. ", " vs ", " @ "]:
        if sep in q_normalized:
            parts = q_normalized.split(sep)
            if len(parts) == 2:
                raw_team_a = parts[0].strip()
                raw_team_b = parts[1].strip()
                
                # Check which team from TEAM_MAP matches raw_team_a and raw_team_b
                for fullname in TEAM_MAP:
                    clean_full = clean_str(fullname)
                    clean_a = clean_str(raw_team_a)
                    clean_b = clean_str(raw_team_b)
                    
                    if clean_full in clean_a or (len(clean_a) >= 4 and clean_a in clean_full):
                        away_team_full = fullname
                    if clean_full in clean_b or (len(clean_b) >= 4 and clean_b in clean_full):
                        home_team_full = fullname
                break
                
    if not away_team_full or not home_team_full or away_team_full == home_team_full:
        return None
        
    home_team_abbr = TEAM_MAP[home_team_full]
    away_team_abbr = TEAM_MAP[away_team_full]
    
    # Parse outcomes and prices
    outcomes = safe_json_loads(market.get('outcomes'))
    prices = safe_json_loads(market.get('outcomePrices'))
    
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
        
    home_yes_price = None
    away_yes_price = None
    
    for idx, outcome in enumerate(outcomes):
        clean_outcome = clean_str(outcome)
        if clean_str(home_team_full) in clean_outcome or (len(clean_outcome) >= 4 and clean_outcome in clean_str(home_team_full)):
            try:
                home_yes_price = float(prices[idx])
            except (ValueError, TypeError, IndexError):
                pass
        elif clean_str(away_team_full) in clean_outcome or (len(clean_outcome) >= 4 and clean_outcome in clean_str(away_team_full)):
            try:
                away_yes_price = float(prices[idx])
            except (ValueError, TypeError, IndexError):
                pass
                
    if home_yes_price is None or away_yes_price is None:
        return None
        
    # Parse date
    end_date_str = market.get('endDate')
    match_date = parse_match_date(end_date_str)
    if not match_date:
        return None
        
    # Parse volume
    try:
        volume = float(market.get('volumeNum') or market.get('volume') or 0.0)
    except (ValueError, TypeError):
        volume = 0.0
        
    return {
        'match_date': match_date,
        'home_team': home_team_abbr,
        'away_team': away_team_abbr,
        'home_yes_price': home_yes_price,
        'away_yes_price': away_yes_price,
        'polymarket_volume': volume
    }
def fetch_markets():
    """
    Fetches active WNBA markets from both the user's primary endpoint
    and a paginated WNBA-tagged endpoint as a robust fallback.
    """
    all_markets = []
    
    # 1. Fetch from the primary URL requested by the user
    primary_url = "https://gamma-api.polymarket.com/markets?active=true&limit=100&query=WNBA"
    print(f"Fetching from primary URL: {primary_url}")
    try:
        r = requests.get(primary_url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                all_markets.extend(data)
                print(f"  Successfully fetched {len(data)} markets from primary URL.")
            else:
                print(f"  Warning: Primary URL returned non-list JSON: {type(data)}")
        else:
            print(f"  Warning: Primary URL returned status code {r.status_code}")
    except Exception as e:
        print(f"  Error fetching from primary URL: {e}")
        
    # 2. Fetch from WNBA tag with pagination to ensure we get live WNBA matchups
    # (since the real Polymarket API ignores query=WNBA on /markets)
    tag_id = "100254"
    print(f"Fetching paginated markets for WNBA tag (tag_id={tag_id})...")
    for offset in range(0, 500, 100):
        url = f"https://gamma-api.polymarket.com/markets?active=true&limit=100&offset={offset}&tag_id={tag_id}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  Tag endpoint failed at offset {offset} with status: {r.status_code}")
                break
            data = r.json()
            if not data or not isinstance(data, list):
                break
            all_markets.extend(data)
            print(f"  Fetched {len(data)} markets at offset {offset}.")
            if len(data) < 100:
                break
        except Exception as e:
            print(f"  Error fetching tag endpoint at offset {offset}: {e}")
            break
            
    return all_markets
def init_db(db_path=DB_NAME):
    """
    Ensures that the polymarket_odds table exists before inserting.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS polymarket_odds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_yes_price REAL NOT NULL,
        away_yes_price REAL NOT NULL,
        polymarket_volume REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_poly_match 
    ON polymarket_odds(match_date, home_team, away_team);
    """)
    conn.commit()
    conn.close()
def main():
    # Ensure DB is initialized
    try:
        init_db()
    except Exception as e:
        print(f"Critical error initializing database: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Fetch markets
    raw_markets = fetch_markets()
    if not raw_markets:
        print("No markets fetched from Polymarket API.")
        return
        
    print(f"Parsing {len(raw_markets)} fetched markets...")
    parsed_records = []
    seen_keys = set()
    
    for m in raw_markets:
        try:
            parsed = parse_market(m)
            if parsed:
                key = (parsed['match_date'], parsed['home_team'], parsed['away_team'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    parsed_records.append(parsed)
        except Exception as e:
            print(f"  Failed parsing market: {e}")
            
    print(f"Successfully parsed {len(parsed_records)} active WNBA matchup markets.")
    
    if not parsed_records:
        print("No valid WNBA matchup markets were found to insert.")
        return
        
    # Insert or update in database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    inserted_count = 0
    try:
        cursor.executemany("""
        INSERT OR REPLACE INTO polymarket_odds (
            match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume
        ) VALUES (
            :match_date, :home_team, :away_team, :home_yes_price, :away_yes_price, :polymarket_volume
        );
        """, parsed_records)
        inserted_count = cursor.rowcount
        conn.commit()
        print(f"Database update complete. Affected/inserted rows: {inserted_count}")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting parsed records into database: {e}", file=sys.stderr)
    finally:
        conn.close()
if __name__ == "__main__":
    main()
