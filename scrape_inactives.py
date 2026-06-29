import sqlite3
import time
import random
import sys
from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3

DB_PATH = "wnba.db"

def init_db():
    """Initializes the required database tables."""
    print("Initializing database tables...")
    conn = sqlite3.connect(DB_PATH)
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
    
    # 2. Progress/Caching table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scraped_games (
        GameID TEXT PRIMARY KEY
    );
    """)
    
    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")

def get_all_games():
    """Queries WNBA regular season game logs for seasons 2018-2026 to extract game IDs and dates."""
    seasons = [str(year) for year in range(2018, 2027)]
    game_info = {} # game_id -> date
    
    for season in seasons:
        print(f"Fetching WNBA regular season game logs for {season}...")
        retries = 3
        for attempt in range(retries):
            try:
                g = leaguegamelog.LeagueGameLog(
                    league_id='10',
                    season=season,
                    season_type_all_star='Regular Season'
                )
                df = g.get_data_frames()[0]
                unique_games = 0
                for _, row in df.iterrows():
                    game_id = row['GAME_ID']
                    game_date = row['GAME_DATE']
                    if game_id not in game_info:
                        game_info[game_id] = game_date
                        unique_games += 1
                print(f"  Season {season}: Found {unique_games} unique games.")
                time.sleep(1.0)
                break
            except Exception as e:
                print(f"  Error fetching season {season} game logs (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    sleep_time = 2.0 * (attempt + 1)
                    time.sleep(sleep_time)
                else:
                    print(f"  Failed to fetch game logs for season {season}.")
                    
    return game_info

def get_scraped_game_ids():
    """Retrieves the set of game IDs that have already been successfully scraped."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT GameID FROM scraped_games;")
    rows = cursor.fetchall()
    conn.close()
    return set(r[0] for r in rows)

def process_game(game_id, date):
    """Fetches boxscore for a single game, extracts inactives, and inserts them into the DB."""
    retries = 3
    data = None
    for attempt in range(retries):
        try:
            summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
            data = summary.get_dict()
            break
        except Exception as e:
            print(f"Error fetching boxscore for game {game_id} (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                # Exponential backoff with jitter
                sleep_time = 2.0 * (attempt + 1) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
            else:
                raise e

    if not data or 'boxScoreSummary' not in data:
        raise ValueError(f"Invalid or missing boxScoreSummary structure for game {game_id}")

    box = data['boxScoreSummary']
    
    # Process Home Team
    home_team = box.get('homeTeam', {})
    home_city = home_team.get('teamCity', '')
    home_name = home_team.get('teamName', '')
    home_full_name = f"{home_city} {home_name}".strip()
    home_inactives = home_team.get('inactives', [])
    if not home_inactives:
        home_inactives = []

    # Process Away Team
    away_team = box.get('awayTeam', {})
    away_city = away_team.get('teamCity', '')
    away_name = away_team.get('teamName', '')
    away_full_name = f"{away_city} {away_name}".strip()
    away_inactives = away_team.get('inactives', [])
    if not away_inactives:
        away_inactives = []

    # Prepare inactive records
    records_to_insert = []
    
    for player in home_inactives:
        first = player.get('firstName', '')
        last = player.get('familyName', '')
        name = f"{first} {last}".strip()
        if name:
            records_to_insert.append((date, home_full_name, name))
            
    for player in away_inactives:
        first = player.get('firstName', '')
        last = player.get('familyName', '')
        name = f"{first} {last}".strip()
        if name:
            records_to_insert.append((date, away_full_name, name))

    # Save to database in a single atomic transaction
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN TRANSACTION;")
        for record in records_to_insert:
            cursor.execute(
                "INSERT INTO historical_inactives (Date, Team, Player) VALUES (?, ?, ?);",
                record
            )
        # Mark game as scraped
        cursor.execute("INSERT INTO scraped_games (GameID) VALUES (?);", (game_id,))
        conn.commit()
    except Exception as ex:
        conn.rollback()
        raise ex
    finally:
        conn.close()

    return len(records_to_insert)

def main():
    init_db()
    
    # Get game IDs from league logs
    game_info = get_all_games()
    if not game_info:
        print("No WNBA game logs found.")
        sys.exit(1)
        
    all_game_ids = list(game_info.keys())
    print(f"Total games retrieved from game logs: {len(all_game_ids)}")
    
    # Get already scraped game IDs
    scraped_game_ids = get_scraped_game_ids()
    print(f"Already scraped games: {len(scraped_game_ids)}")
    
    # Determine games that need to be scraped
    games_to_scrape = [gid for gid in all_game_ids if gid not in scraped_game_ids]
    print(f"Games remaining to scrape: {len(games_to_scrape)}")
    
    if not games_to_scrape:
        print("All games have already been processed.")
        # Print current totals
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM historical_inactives;")
        total_records = c.fetchone()[0]
        conn.close()
        print(f"Total inactive player records in table: {total_records}")
        
        # Also populate frontend/wnba.db
        import shutil
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_db_path = os.path.join(base_dir, 'frontend', 'wnba.db')
        try:
            shutil.copy2(DB_PATH, frontend_db_path)
            print(f"Copied database to frontend: {frontend_db_path}")
        except Exception as e:
            print(f"Warning: Failed to copy database to frontend: {e}")
            
        return

    # Process games sequentially with a 1.0 second delay
    new_inserts = 0
    scraped_count = 0
    failed_count = 0
    
    print("Starting scraping of inactive players...")
    try:
        for idx, game_id in enumerate(games_to_scrape):
            date = game_info[game_id]
            print(f"[{idx+1}/{len(games_to_scrape)}] Scraping game {game_id} ({date})...")
            
            try:
                inserted = process_game(game_id, date)
                new_inserts += inserted
                scraped_count += 1
                print(f"  Success: found {inserted} inactive player(s).")
            except Exception as e:
                failed_count += 1
                print(f"  Failed to scrape game {game_id}: {e}")
            
            # Enforce rate-limiting delay
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Saving progress and exiting...")
        
    print("\n--- Scraping Report ---")
    print(f"Games successfully processed in this run: {scraped_count}")
    print(f"Games failed in this run: {failed_count}")
    print(f"New inactive records inserted in this run: {new_inserts}")
    
    # Query total records
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM historical_inactives;")
    total_records = c.fetchone()[0]
    conn.close()
    print(f"Total inactive player records in 'historical_inactives' table: {total_records}")

    # Also populate frontend/wnba.db
    import shutil
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_db_path = os.path.join(base_dir, 'frontend', 'wnba.db')
    try:
        shutil.copy2(DB_PATH, frontend_db_path)
        print(f"Copied database to frontend: {frontend_db_path}")
    except Exception as e:
        print(f"Warning: Failed to copy database to frontend: {e}")

if __name__ == "__main__":
    main()
