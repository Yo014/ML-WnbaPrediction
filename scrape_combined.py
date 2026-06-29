import time
import sqlite3
import random
import sys
import argparse
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3
import nba_api.stats.library.http

# Override global User-Agent to bypass rate limit checks
nba_api.stats.library.http.STATS_HEADERS['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

DB_NAME = "wnba.db"

# Team abbreviation to canonical name mapping
TEAM_NAME_MAP = {
    'ATL': 'Atlanta Dream', 'CHI': 'Chicago Sky', 'CON': 'Connecticut Sun',
    'DAL': 'Dallas Wings', 'IND': 'Indiana Fever', 'LAS': 'Los Angeles Sparks',
    'LVA': 'Las Vegas Aces', 'MIN': 'Minnesota Lynx', 'NYL': 'New York Liberty',
    'PHO': 'Phoenix Mercury', 'PHX': 'Phoenix Mercury', 'SEA': 'Seattle Storm',
    'WAS': 'Washington Mystics', 'GSV': 'Golden State Valkyries', 'TOR': 'Toronto Tempo',
    'PDX': 'Portland Fire', 'POR': 'Portland Fire'
}

def init_db(conn):
    cursor = conn.cursor()
    # 1. Historical inactives table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_inactives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT NOT NULL,
        Team TEXT NOT NULL,
        Player TEXT NOT NULL
    );
    """)
    # 2. Progress/Caching tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scraped_games (
        GameID TEXT PRIMARY KEY
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS referee_scraping_cache (
        game_id TEXT PRIMARY KEY,
        Date TEXT,
        HomeTeam TEXT,
        AwayTeam TEXT,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

def fetch_game_logs(seasons):
    all_games = {}
    for season in seasons:
        print(f"Fetching WNBA regular season game logs for {season}...")
        retries = 5
        while retries > 0:
            try:
                time.sleep(1.0)
                g = leaguegamelog.LeagueGameLog(
                    league_id='10',
                    season=season,
                    season_type_all_star='Regular Season'
                )
                df = g.get_data_frames()[0]
                print(f"  Successfully fetched {len(df)} rows for {season}")
                
                # Pair the rows by game ID
                for _, row in df.iterrows():
                    gid = row['GAME_ID']
                    date = row['GAME_DATE']
                    matchup = row['MATCHUP']
                    team = row['TEAM_ABBREVIATION']
                    
                    if gid not in all_games:
                        all_games[gid] = {'Date': date, 'Season': int(season)}
                        
                    if 'vs.' in matchup:
                        all_games[gid]['HomeTeam'] = TEAM_NAME_MAP.get(team, team)
                        parts = matchup.split(' vs. ')
                        if len(parts) > 1:
                            opp = parts[1].strip()
                            all_games[gid]['AwayTeam'] = TEAM_NAME_MAP.get(opp, opp)
                    elif '@' in matchup:
                        all_games[gid]['AwayTeam'] = TEAM_NAME_MAP.get(team, team)
                        parts = matchup.split(' @ ')
                        if len(parts) > 1:
                            opp = parts[1].strip()
                            all_games[gid]['HomeTeam'] = TEAM_NAME_MAP.get(opp, opp)
                break
            except Exception as e:
                retries -= 1
                print(f"  Error fetching season {season}: {e}. Retrying {retries} more times...")
                time.sleep(5.0)
        else:
            print(f"  Failed to fetch games for season {season} after retries.")
            
    return all_games

def main():
    parser = argparse.ArgumentParser(description="Scrape WNBA Referees and Inactives")
    parser.add_argument('--seasons', nargs='+', required=True, help="List of seasons to process")
    parser.add_argument('--delay', type=float, default=0.6, help="Delay between API calls in seconds")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    init_db(conn)
    
    # Load already scraped game IDs from caches
    cursor = conn.cursor()
    cursor.execute("SELECT GameID FROM scraped_games")
    scraped_inactives = set(row[0] for row in cursor.fetchall())
    
    cursor.execute("SELECT game_id FROM referee_scraping_cache")
    scraped_referees = set(row[0] for row in cursor.fetchall())
    
    # We consider a game fully scraped if it's in both caches
    fully_scraped = scraped_inactives.intersection(scraped_referees)
    print(f"Loaded cache: {len(scraped_inactives)} inactives, {len(scraped_referees)} referees. Fully scraped: {len(fully_scraped)}")
    
    # Fetch game logs for requested seasons
    api_games = fetch_game_logs(args.seasons)
    print(f"Fetched {len(api_games)} unique games from WNBA API for seasons {args.seasons}.")
    
    # Load all matches from DB
    cursor.execute("SELECT Date, HomeTeam, AwayTeam FROM raw_matches")
    db_matches = cursor.fetchall()
    
    # Map (Date, HomeTeam, AwayTeam) -> GAME_ID
    match_map = {}
    for gid, g_info in api_games.items():
        key = (g_info['Date'], g_info.get('HomeTeam'), g_info.get('AwayTeam'))
        match_map[key] = gid

    # Filter to-scrape list
    to_scrape = []
    for row in db_matches:
        date, home, away = row
        season = int(date[:4])
        if str(season) not in args.seasons:
            continue
        key = (date, home, away)
        gid = match_map.get(key)
        if not gid:
            continue
        if gid in fully_scraped:
            continue
        to_scrape.append((gid, date, home, away))
        
    print(f"Seasons {args.seasons}: Total matches in DB: {len(to_scrape) + len(fully_scraped.intersection(set(match_map.values())))}")
    print(f"Remaining to scrape: {len(to_scrape)}")
    
    if not to_scrape:
        print("All games in these seasons are already scraped.")
        conn.close()
        return

    success_count = 0
    error_count = 0
    
    for idx, (gid, date, home, away) in enumerate(to_scrape):
        print(f"[{idx+1}/{len(to_scrape)}] Processing game {gid} ({date}: {away} @ {home})...")
        retries = 3
        data = None
        
        while retries > 0:
            try:
                # Add random jitter to delay (0.5 to 1.5 seconds extra) to avoid detection
                jitter = random.uniform(0.5, 1.5)
                time.sleep(args.delay + jitter)
                box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=gid)
                data = box.get_dict()
                break
            except Exception as e:
                retries -= 1
                print(f"  Error querying game {gid}: {e}. Retrying {retries} more times...")
                time.sleep(5.0)
                
        if not data or 'boxScoreSummary' not in data:
            print(f"  Failed to retrieve boxscore for game {gid}")
            error_count += 1
            continue
            
        try:
            box = data['boxScoreSummary']
            
            # 1. Parse Officials
            officials_data = box.get('officials', [])
            officials = [o.get('name') for o in officials_data]
            crew_chief = officials[0] if len(officials) > 0 else None
            home_ref = officials[1] if len(officials) > 1 else None
            away_ref = officials[2] if len(officials) > 2 else None
            
            # 2. Parse Inactives
            home_team = box.get('homeTeam', {})
            home_city = home_team.get('teamCity', '')
            home_name = home_team.get('teamName', '')
            home_full = f"{home_city} {home_name}".strip()
            home_inactives = home_team.get('inactives', [])
            
            away_team = box.get('awayTeam', {})
            away_city = away_team.get('teamCity', '')
            away_name = away_team.get('teamName', '')
            away_full = f"{away_city} {away_name}".strip()
            away_inactives = away_team.get('inactives', [])
            
            records_inactives = []
            for p in home_inactives:
                first = p.get('firstName', '')
                last = p.get('familyName', '')
                name = f"{first} {last}".strip()
                if name:
                    records_inactives.append((date, home_full, name))
                    
            for p in away_inactives:
                first = p.get('firstName', '')
                last = p.get('familyName', '')
                name = f"{first} {last}".strip()
                if name:
                    records_inactives.append((date, away_full, name))
                    
            # 3. Database Updates in a Single Transaction
            cursor.execute("BEGIN TRANSACTION;")
            
            # Update referees in raw_matches
            cursor.execute("""
            UPDATE raw_matches
            SET CrewChief = ?, HomeRef = ?, AwayRef = ?
            WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?
            """, (crew_chief, home_ref, away_ref, date, home, away))
            
            # Insert inactive players
            for record in records_inactives:
                cursor.execute(
                    "INSERT INTO historical_inactives (Date, Team, Player) VALUES (?, ?, ?);",
                    record
                )
                
            # Mark caches
            cursor.execute("INSERT OR REPLACE INTO scraped_games (GameID) VALUES (?);", (gid,))
            cursor.execute("INSERT OR REPLACE INTO referee_scraping_cache (game_id, Date, HomeTeam, AwayTeam) VALUES (?, ?, ?, ?);", (gid, date, home, away))
            
            conn.commit()
            success_count += 1
            print(f"  Success: Referees updated, {len(records_inactives)} inactive(s) inserted.")
        except Exception as db_ex:
            conn.rollback()
            print(f"  Database transaction failed for game {gid}: db_ex={db_ex}")
            error_count += 1
            
    conn.close()
    
    # Also populate frontend/wnba.db
    import shutil
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_db_path = os.path.join(base_dir, 'frontend', 'wnba.db')
    try:
        shutil.copy2(DB_NAME, frontend_db_path)
        print(f"Copied database to frontend: {frontend_db_path}")
    except Exception as e:
        print(f"Warning: Failed to copy database to frontend: {e}")

    print(f"\n--- Run completed ---")
    print(f"Processed: {success_count}, Failed: {error_count}")

if __name__ == '__main__':
    main()
