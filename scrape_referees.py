import time
import sqlite3
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3

# Database configuration
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

def init_cache(conn):
    cursor = conn.cursor()
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

def fetch_all_game_logs():
    """
    Fetches all game logs for WNBA regular seasons 2018-2026.
    """
    seasons = [str(year) for year in range(2018, 2027)]
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
                        all_games[gid] = {'Date': date}
                        
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
    conn = sqlite3.connect(DB_NAME)
    init_cache(conn)
    
    # Fetch existing cache
    cursor = conn.cursor()
    cursor.execute("SELECT game_id FROM referee_scraping_cache")
    cached_gids = set(row[0] for row in cursor.fetchall())
    print(f"Loaded {len(cached_gids)} already scraped games from cache.")
    
    # Fetch all WNBA games from API
    api_games = fetch_all_game_logs()
    print(f"Fetched {len(api_games)} unique games from WNBA API.")
    
    # Match API games to DB raw_matches keys
    cursor.execute("SELECT Date, HomeTeam, AwayTeam, CrewChief, HomeRef, AwayRef FROM raw_matches")
    db_matches = cursor.fetchall()
    
    # Map (Date, HomeTeam, AwayTeam) -> GAME_ID
    match_map = {}
    for gid, g_info in api_games.items():
        key = (g_info['Date'], g_info.get('HomeTeam'), g_info.get('AwayTeam'))
        match_map[key] = gid
        
    # Filter the list of DB matches to only those that have a corresponding GAME_ID
    # and whose GAME_ID is NOT in cached_gids
    to_scrape = []
    skipped_cached = 0
    not_found_in_api = 0
    
    for row in db_matches:
        date, home, away, cc, hr, ar = row
        key = (date, home, away)
        gid = match_map.get(key)
        if not gid:
            not_found_in_api += 1
            continue
        if gid in cached_gids:
            skipped_cached += 1
            continue
        to_scrape.append((gid, date, home, away))
        
    print(f"Matches status: Total in DB: {len(db_matches)}")
    print(f"  Already cached/skipped: {skipped_cached}")
    print(f"  Not found in API: {not_found_in_api}")
    print(f"  To scrape now: {len(to_scrape)}")
    
    success_count = 0
    error_count = 0
    
    for i, (gid, date, home, away) in enumerate(to_scrape):
        # Optional print progress every 10 games
        print(f"[{i+1}/{len(to_scrape)}] Querying officials for Game ID: {gid} ({date}: {away} @ {home})...")
        retries = 3
        officials = None
        
        while retries > 0:
            try:
                time.sleep(1.0) # Always sleep 1.0s to avoid rate limiting
                box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=gid)
                box_dict = box.get_dict()
                
                # Check for officials data
                officials_data = box.officials.get_dict().get('data', [])
                officials = [row[2] for row in officials_data]
                break
            except Exception as e:
                retries -= 1
                print(f"  Error querying Game ID {gid}: {e}. Retrying {retries} more times...")
                time.sleep(5.0)
                
        if officials is not None:
            crew_chief = officials[0] if len(officials) > 0 else None
            home_ref = officials[1] if len(officials) > 1 else None
            away_ref = officials[2] if len(officials) > 2 else None
            
            print(f"  Found officials: CrewChief={crew_chief}, HomeRef={home_ref}, AwayRef={away_ref}")
            
            # Update raw_matches and referee_scraping_cache in a transaction
            try:
                cursor.execute("""
                UPDATE raw_matches
                SET CrewChief = ?, HomeRef = ?, AwayRef = ?
                WHERE Date = ? AND HomeTeam = ? AND AwayTeam = ?
                """, (crew_chief, home_ref, away_ref, date, home, away))
                
                cursor.execute("""
                INSERT OR REPLACE INTO referee_scraping_cache (game_id, Date, HomeTeam, AwayTeam)
                VALUES (?, ?, ?, ?)
                """, (gid, date, home, away))
                
                conn.commit()
                success_count += 1
            except Exception as e:
                conn.rollback()
                print(f"  Database transaction failed for Game ID {gid}: {e}")
                error_count += 1
        else:
            print(f"  Failed to query officials for Game ID {gid} after all retries.")
            error_count += 1
            
    # Get total successfully updated in cache
    cursor.execute("SELECT COUNT(*) FROM referee_scraping_cache")
    total_in_cache = cursor.fetchone()[0]
    
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

    print("\n--- Scraping Report ---")
    print(f"Successfully updated matches in this run: {success_count}")
    print(f"Failed matches in this run: {error_count}")
    print(f"Total games successfully updated with actual referees in database: {total_in_cache}")

if __name__ == '__main__':
    main()
