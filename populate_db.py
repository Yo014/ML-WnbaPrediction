import time
import random
import hashlib
import sqlite3
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashplayerstats
from db_manager import initialize_db, get_connection, DB_NAME
from fanduel_odds import fetch_fanduel_odds


# Referee Pool (WNBA officials)
WNBA_REFS = [
    "Roy Gulbeyan", "Cheryl Flores", "Maj Forsberg", "Amy Bonner", "Eric Brewton",
    "Isaac Barnett", "Billy Smith", "Tiara Cruse", "Angel Kent", "Randy Richardson",
    "Kurt Walker", "Byron Jarrett", "Jeff Smith", "Kelly Dawson", "Robert Hussey",
    "Sarah Williams", "Jenna Reneau", "Brandon Enterline", "Kevin Fahy", "Tiffany Bird",
    "Michael Price", "Dallas Gomez", "Ashley Gloss", "Mitchell Ervin", "Fatou Cissoko-Stephens",
    "Tim Greene", "Alexis Mercado", "Jeffrey Smith"
]

# Injury Seeds
INJURY_SEEDS = [
    ("NYL", "Nyara Sabally", "Out", "Day-to-Day"),
    ("LVA", "Chelsea Gray", "Questionable", "2026-06-20"),
    ("IND", "Temi Fagbenle", "Out", "2026-07-01"),
    ("PHO", "Brittney Griner", "Out", "2026-06-25"),
    ("MIN", "Diamond Miller", "Out", "2026-07-15"),
    ("DAL", "Satou Sabally", "Out", "2026-08-01"),
    ("CHI", "Elizabeth Williams", "Out", "Season")
]

class EloModel:
    """
    Tracks team ELO ratings over time with Home Field Advantage and mean reversion.
    """
    def __init__(self, k_factor=20, hfa=50):
        self.ratings = {}
        self.k_factor = k_factor
        self.hfa = hfa
        
    def get_rating(self, team):
        if team not in self.ratings:
            self.ratings[team] = 1500.0
        return self.ratings[team]
        
    def update_ratings(self, home_team, away_team, home_score, away_score):
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        
        # Calculate expected outcome
        expected_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - self.hfa) / 400.0))
        
        # Actual outcome
        actual_home = 1.0 if home_score > away_score else 0.0
        if home_score == away_score:
            actual_home = 0.5
            
        # Update ratings
        self.ratings[home_team] = r_home + self.k_factor * (actual_home - expected_home)
        self.ratings[away_team] = r_away + self.k_factor * ((1.0 - actual_home) - (1.0 - expected_home))

    def revert_to_mean(self):
        """
        Applies season-to-season regression towards 1500.
        """
        for team in self.ratings:
            self.ratings[team] = 0.75 * self.ratings[team] + 0.25 * 1500.0

def assign_refs(date, home_team, away_team):
    """
    Deterministically assigns three referees (Crew Chief, HomeRef, AwayRef)
    from the WNBA_REFS pool based on the game's characteristics.
    """
    seed_str = f"{date}_{home_team}_{away_team}"
    hash_val = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
    rng = random.Random(hash_val)
    selected = rng.sample(WNBA_REFS, 3)
    return selected[0], selected[1], selected[2]

def generate_betting_data(home_team, away_team, date, r_home, r_away):
    """
    Deterministically generates realistic historical closing spreads, opening spreads,
    bookie odds, and over/unders using the ELO ratings.
    """
    seed_str = f"odds_{date}_{home_team}_{away_team}"
    hash_val = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
    rng = random.Random(hash_val)
    
    # ELO difference including HFA
    hfa = 50.0
    elo_diff = r_home + hfa - r_away
    
    # Base spread (approx 28 ELO points = 1 point of spread)
    base_spread = - (elo_diff / 28.0)
    
    # Add Gaussian noise with std=3 for closing spread
    noise_close = rng.gauss(0, 3)
    closing_spread = base_spread + noise_close
    
    # Add Gaussian noise with std=1.5 for opening spread from closing spread
    noise_open = rng.gauss(0, 1.5)
    opening_spread = closing_spread + noise_open
    
    # Round spreads to nearest 0.5
    closing_spread = round(closing_spread * 2) / 2.0
    opening_spread = round(opening_spread * 2) / 2.0
    
    # Approximate win probability based on closing spread (approx 3% prob change per point of spread)
    p_home = 0.5 - (closing_spread * 0.03)
    p_home = max(0.05, min(0.95, p_home))
    
    # Bookie odds with a ~5% margin (vigorish)
    margin = 1.05
    p_home_odds = p_home * margin
    p_away_odds = (1.0 - p_home) * margin
    
    bookie_home_odds = round(1.0 / p_home_odds, 2)
    bookie_away_odds = round(1.0 / p_away_odds, 2)
    
    # Over/Under (average of 162 points in WNBA)
    base_ou = 162.0
    noise_ou = rng.gauss(0, 5)
    over_under = round((base_ou + noise_ou) * 2) / 2.0
    
    return opening_spread, closing_spread, bookie_home_odds, bookie_away_odds, over_under

def fetch_games(seasons):
    """
    Fetches team game logs for WNBA regular seasons.
    """
    all_game_rows = []
    for season in seasons:
        print(f"Fetching game logs for WNBA season {season}...")
        try:
            time.sleep(1.2)
            g = leaguegamelog.LeagueGameLog(
                league_id='10',
                season=season,
                season_type_all_star='Regular Season'
            )
            df = g.get_data_frames()[0]
            print(f"  Successfully fetched {len(df)} rows for season {season}")
            all_game_rows.append(df)
        except Exception as e:
            print(f"  Error fetching games for season {season}: {e}. Retrying with double wait...")
            time.sleep(3.0)
            g = leaguegamelog.LeagueGameLog(
                league_id='10',
                season=season,
                season_type_all_star='Regular Season'
            )
            df = g.get_data_frames()[0]
            print(f"  Successfully fetched {len(df)} rows for season {season} on retry")
            all_game_rows.append(df)
            
    if not all_game_rows:
        return pd.DataFrame()
        
    return pd.concat(all_game_rows, ignore_index=True)

def parse_games_into_matches(df):
    """
    Pairs individual team game logs into unified match records.
    """
    games = {}
    for _, row in df.iterrows():
        game_id = row['GAME_ID']
        team = row['TEAM_ABBREVIATION']
        pts = row['PTS']
        matchup = row['MATCHUP']
        date = row['GAME_DATE']
        season_id = row['SEASON_ID']
        
        fga = row.get('FGA', 0)
        fta = row.get('FTA', 0)
        oreb = row.get('OREB', 0)
        tov = row.get('TOV', 0)
        min_played = row.get('MIN', 0)
        fgm = row.get('FGM', 0)
        fg3m = row.get('FG3M', 0)
        ftm = row.get('FTM', 0)
        dreb = row.get('DREB', 0)
        pf = row.get('PF', 0)
        
        if game_id not in games:
            games[game_id] = {
                'Date': date,
                'SEASON_ID': season_id
            }
            
        if 'vs.' in matchup:
            games[game_id]['HomeTeam'] = team
            games[game_id]['HomeScore'] = pts
            games[game_id]['HomeFGA'] = fga
            games[game_id]['HomeFTA'] = fta
            games[game_id]['HomeOREB'] = oreb
            games[game_id]['HomeTOV'] = tov
            games[game_id]['HomeMIN'] = min_played
            games[game_id]['HomeFGM'] = fgm
            games[game_id]['HomeFG3M'] = fg3m
            games[game_id]['HomeFTM'] = ftm
            games[game_id]['HomeDREB'] = dreb
            games[game_id]['HomePF'] = pf
            parts = matchup.split(' vs. ')
            if len(parts) > 1:
                games[game_id]['AwayTeam'] = parts[1].strip()
        elif '@' in matchup:
            games[game_id]['AwayTeam'] = team
            games[game_id]['AwayScore'] = pts
            games[game_id]['AwayFGA'] = fga
            games[game_id]['AwayFTA'] = fta
            games[game_id]['AwayOREB'] = oreb
            games[game_id]['AwayTOV'] = tov
            games[game_id]['AwayMIN'] = min_played
            games[game_id]['AwayFGM'] = fgm
            games[game_id]['AwayFG3M'] = fg3m
            games[game_id]['AwayFTM'] = ftm
            games[game_id]['AwayDREB'] = dreb
            games[game_id]['AwayPF'] = pf
            parts = matchup.split(' @ ')
            if len(parts) > 1:
                games[game_id]['HomeTeam'] = parts[1].strip()
                
    # Keep only completed games where we successfully identified both teams and scores
    match_list = []
    for gid, g in games.items():
        if all(k in g for k in ['HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore']):
            for prefix in ['Home', 'Away']:
                for stat in ['FGA', 'FTA', 'OREB', 'TOV', 'MIN', 'FGM', 'FG3M', 'FTM', 'DREB', 'PF']:
                    k = f"{prefix}{stat}"
                    if k not in g:
                        g[k] = 0.0
            for col in ['HomePossessions', 'HomePace', 'AwayPossessions', 'AwayPace', 
                        'HomePtsScored', 'HomePtsConceded', 'AwayPtsScored', 'AwayPtsConceded']:
                g[col] = None
            match_list.append(g)
            
    # Sort matches chronologically
    match_list.sort(key=lambda x: x['Date'])
    return match_list

def fetch_player_stats(seasons):
    """
    Fetches base and advanced player stats for WNBA regular seasons.
    """
    all_player_stats = []
    for season in seasons:
        print(f"Fetching player stats for WNBA season {season}...")
        try:
            time.sleep(1.2)
            base_df = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable='10',
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Base'
            ).get_data_frames()[0]
            
            time.sleep(1.2)
            adv_df = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable='10',
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            
            print(f"  Successfully fetched {len(base_df)} base and {len(adv_df)} advanced rows for {season}")
            
            # Merge base and advanced datasets
            merged = pd.merge(
                base_df,
                adv_df,
                on=['PLAYER_ID', 'TEAM_ID'],
                suffixes=('', '_adv')
            )
            merged['Season'] = int(season)
            all_player_stats.append(merged)
        except Exception as e:
            print(f"  Error fetching player stats for season {season}: {e}. Retrying...")
            time.sleep(3.0)
            base_df = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable='10',
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Base'
            ).get_data_frames()[0]
            time.sleep(1.5)
            adv_df = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable='10',
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            merged = pd.merge(
                base_df,
                adv_df,
                on=['PLAYER_ID', 'TEAM_ID'],
                suffixes=('', '_adv')
            )
            merged['Season'] = int(season)
            all_player_stats.append(merged)
            
    if not all_player_stats:
        return pd.DataFrame()
        
    return pd.concat(all_player_stats, ignore_index=True)

def process_player_stats(df):
    """
    Computes/approximates BPM and WS from raw player statistics.
    """
    processed_records = []
    for _, row in df.iterrows():
        gp = row.get('GP', 0)
        if gp <= 0:
            continue
            
        pts = row.get('PTS', 0)
        ast = row.get('AST', 0)
        tov = row.get('TOV', 0)
        fgm = row.get('FGM', 0)
        reb = row.get('REB', 0)
        oreb = row.get('OREB', 0)
        dreb = max(0, reb - oreb)
        stl = row.get('STL', 0)
        blk = row.get('BLK', 0)
        pf = row.get('PF', 0)
        w_pct = row.get('W_PCT', 0.5)
        
        # Per game averages
        min_pg = row.get('MIN', 0) / gp
        if min_pg <= 0:
            continue
            
        pts_pg = pts / gp
        ast_pg = ast / gp
        reb_pg = reb / gp
        stl_pg = stl / gp
        blk_pg = blk / gp
        tov_pg = tov / gp
        pf_pg = pf / gp
        
        # Per 36-minute rates
        factor = 36.0 / min_pg
        pts_36 = pts_pg * factor
        ast_36 = ast_pg * factor
        reb_36 = reb_pg * factor
        stl_36 = stl_pg * factor
        blk_36 = blk_pg * factor
        tov_36 = tov_pg * factor
        pf_36 = pf_pg * factor
        
        net_rating = row.get('NET_RATING', 0.0)
        usg_pct = row.get('USG_PCT', 0.0) * 100.0
        
        # Box score contribution (BSC)
        bsc_36 = pts_36 + 1.2 * ast_36 + 0.8 * reb_36 + 2.0 * stl_36 + 2.0 * blk_36 - 1.5 * tov_36 - 0.2 * pf_36
        
        # Approximate BPM: rate-based player performance metric
        bpm = 0.22 * bsc_36 + 0.15 * net_rating + 0.12 * usg_pct - 9.0
        bpm = round(max(-15.0, min(15.0, bpm)), 2)
        
        # Approximate Win Shares: cumulative metric
        off_cont = pts + 0.5 * ast - 0.9 * tov + 0.1 * fgm
        def_cont = 0.5 * dreb + 2.0 * stl + 1.5 * blk - 0.5 * pf
        ws_base = 0.006 * off_cont + 0.006 * def_cont
        # Boost/debuff based on team success
        ws = ws_base * (w_pct + 0.5)
        ws = round(max(-3.0, min(15.0, ws)), 2)
        
        processed_records.append({
            'Season': int(row['Season']),
            'Player': row['PLAYER_NAME'],
            'Team': row['TEAM_ABBREVIATION'],
            'GP': int(gp),
            'MIN': round(min_pg, 1),
            'PTS': round(pts_pg, 1),
            'AST': round(ast_pg, 1),
            'TRB': round(reb_pg, 1),
            'USG_PCT': round(row.get('USG_PCT', 0.0), 3),
            'BPM': bpm,
            'WS': ws
        })
        
    return processed_records

def main():
    # Drop existing tables to apply updated schema
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS raw_matches;")
    cursor.execute("DROP TABLE IF EXISTS player_stats;")
    cursor.execute("DROP TABLE IF EXISTS injuries;")
    conn.commit()
    conn.close()

    # 1. Initialize the database
    initialize_db()
    
    seasons = [str(year) for year in range(2018, 2027)]
    
    # 2. Fetch and parse matches
    games_df = fetch_games(seasons)
    if games_df.empty:
        print("Error: No games fetched.")
        return
        
    matches = parse_games_into_matches(games_df)
    print(f"Parsed {len(matches)} total match records.")
    
    # 3. Simulate ELO and generate deterministic spreads, odds, and referee assignments
    elo = EloModel()
    current_season = None
    
    # Query existing FanDuel odds from database before they get cleared
    existing_fanduel_odds = {}
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(raw_matches);")
        columns = [c[1] for c in cursor.fetchall()]
        if 'IsFanduelOdds' in columns:
            cursor.execute("""
                SELECT Date, HomeTeam, AwayTeam, BookieHomeOdds, BookieAwayOdds, OpeningSpread, ClosingSpread, OverUnder 
                FROM raw_matches 
                WHERE IsFanduelOdds = 1
            """)
            for row in cursor.fetchall():
                key = (row[0], row[1], row[2])
                existing_fanduel_odds[key] = {
                    'BookieHomeOdds': row[3],
                    'BookieAwayOdds': row[4],
                    'OpeningSpread': row[5],
                    'ClosingSpread': row[6],
                    'OverUnder': row[7]
                }
            print(f"Loaded {len(existing_fanduel_odds)} existing FanDuel odds records from database.")
        conn.close()
    except Exception as e:
        print("Failed to query existing FanDuel odds:", e)
    
    # Fetch live FanDuel odds
    try:
        fd_odds_list = fetch_fanduel_odds()
        print(f"Fetched {len(fd_odds_list)} live FanDuel odds.")
    except Exception as e:
        print("Failed to fetch live FanDuel odds in populate_db.py:", e)
        fd_odds_list = []
        
    for match in matches:
        date = match['Date']
        season = date[:4]
        
        # Mean reversion at season transitions
        if current_season is not None and season != current_season:
            elo.revert_to_mean()
        current_season = season
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        home_score = match['HomeScore']
        away_score = match['AwayScore']
        
        # Get ratings before match
        r_home = elo.get_rating(home_team)
        r_away = elo.get_rating(away_team)
        
        # Generate lines & referee assignments
        opening_spread, closing_spread, bookie_home_odds, bookie_away_odds, over_under = generate_betting_data(
            home_team, away_team, date, r_home, r_away
        )
        
        is_fanduel_odds = 0
        db_key = (date, home_team, away_team)
        
        # Check if match has preserved FanDuel odds in database
        if db_key in existing_fanduel_odds:
            stored = existing_fanduel_odds[db_key]
            bookie_home_odds = stored['BookieHomeOdds']
            bookie_away_odds = stored['BookieAwayOdds']
            opening_spread = stored['OpeningSpread']
            closing_spread = stored['ClosingSpread']
            over_under = stored['OverUnder']
            is_fanduel_odds = 1
        else:
            # Check if match exists in live FanDuel odds (matching date and teams)
            fd_match = None
            for fd_g in fd_odds_list:
                if fd_g.get('home_team_full') == home_team and fd_g.get('away_team_full') == away_team:
                    fd_date = fd_g.get('date')
                    if fd_date == date:
                        fd_match = fd_g
                        break
                    else:
                        try:
                            d1 = pd.to_datetime(fd_date).date()
                            d2 = pd.to_datetime(date).date()
                            if abs((d1 - d2).days) <= 1:
                                fd_match = fd_g
                                break
                        except Exception:
                            pass
            
            if fd_match:
                bookie_home_odds = fd_match['home_odds']
                bookie_away_odds = fd_match['away_odds']
                closing_spread = fd_match['closing_spread']
                over_under = fd_match['over_under']
                is_fanduel_odds = 1
            
        crew_chief, home_ref, away_ref = assign_refs(date, home_team, away_team)
        
        # Store back in the match dictionary
        match['OpeningSpread'] = opening_spread
        match['ClosingSpread'] = closing_spread
        match['BookieHomeOdds'] = bookie_home_odds
        match['BookieAwayOdds'] = bookie_away_odds
        match['OverUnder'] = over_under
        match['IsFanduelOdds'] = is_fanduel_odds
        match['CrewChief'] = crew_chief
        match['HomeRef'] = home_ref
        match['AwayRef'] = away_ref
        
        # Update ELO ratings

        elo.update_ratings(home_team, away_team, home_score, away_score)
        
    # 4. Fetch and process player statistics
    player_df = fetch_player_stats(seasons)
    if player_df.empty:
        print("Error: No player statistics fetched.")
        return
        
    processed_players = process_player_stats(player_df)
    print(f"Processed {len(processed_players)} player stats records.")
    
    # 5. Insert everything into the database
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clear out any existing records to ensure a fresh, clean seed
    cursor.execute("DELETE FROM raw_matches;")
    cursor.execute("DELETE FROM player_stats;")
    cursor.execute("DELETE FROM injuries;")
    conn.commit()
    
    # Insert matches
    cursor.executemany("""
    INSERT INTO raw_matches (
        Date, HomeTeam, AwayTeam, HomeScore, AwayScore,
        HomeRef, AwayRef, CrewChief,
        BookieHomeOdds, BookieAwayOdds, OpeningSpread, ClosingSpread, OverUnder,
        HomeFGA, HomeFTA, HomeOREB, HomeTOV, HomeMIN,
        HomeFGM, HomeFG3M, HomeFTM, HomeDREB, HomePF,
        AwayFGA, AwayFTA, AwayOREB, AwayTOV, AwayMIN,
        AwayFGM, AwayFG3M, AwayFTM, AwayDREB, AwayPF,
        HomePossessions, HomePace, AwayPossessions, AwayPace,
        HomePtsScored, HomePtsConceded, AwayPtsScored, AwayPtsConceded,
        IsFanduelOdds
    ) VALUES (
        :Date, :HomeTeam, :AwayTeam, :HomeScore, :AwayScore,
        :HomeRef, :AwayRef, :CrewChief,
        :BookieHomeOdds, :BookieAwayOdds, :OpeningSpread, :ClosingSpread, :OverUnder,
        :HomeFGA, :HomeFTA, :HomeOREB, :HomeTOV, :HomeMIN,
        :HomeFGM, :HomeFG3M, :HomeFTM, :HomeDREB, :HomePF,
        :AwayFGA, :AwayFTA, :AwayOREB, :AwayTOV, :AwayMIN,
        :AwayFGM, :AwayFG3M, :AwayFTM, :AwayDREB, :AwayPF,
        :HomePossessions, :HomePace, :AwayPossessions, :AwayPace,
        :HomePtsScored, :HomePtsConceded, :AwayPtsScored, :AwayPtsConceded,
        :IsFanduelOdds
    );
    """, matches)
    
    # Insert player stats
    cursor.executemany("""
    INSERT INTO player_stats (
        Season, Player, Team, GP, MIN, PTS, AST, TRB, USG_PCT, BPM, WS
    ) VALUES (
        :Season, :Player, :Team, :GP, :MIN, :PTS, :AST, :TRB, :USG_PCT, :BPM, :WS
    );
    """, processed_players)
    
    # Insert injuries
    cursor.executemany("""
    INSERT INTO injuries (
        Team, Player, InjuryStatus, ExpectedReturnDate
    ) VALUES (?, ?, ?, ?);
    """, INJURY_SEEDS)
    
    conn.commit()
    conn.close()
    
    # Run the downstream processing pipeline to update CSV files and database fields
    try:
        print("\nRunning downstream pipeline to sync files...")
        from build_squad_health import build_squad_health
        from data_processing import standardize_and_calculate_metrics
        import feature_engineering
        import train_model
        
        print("Calculating squad health...")
        build_squad_health()
        
        print("Standardizing names and calculating metrics...")
        standardize_and_calculate_metrics()
        
        print("Running feature engineering...")
        feature_engineering.main()
        print("Running model training...")
        train_model.main()
        
        
        print("Pipeline sync completed successfully!")
    except Exception as e:
        print(f"Warning: Failed to run pipeline sync: {e}")
        
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
    
    print("\nSeeding complete!")
    print(f"Seeded {len(matches)} match records.")
    print(f"Seeded {len(processed_players)} player stats records.")
    print(f"Seeded {len(INJURY_SEEDS)} injury records.")

if __name__ == "__main__":
    main()
