import sqlite3
import pandas as pd
from db_manager import DB_NAME, get_connection

# Comprehensive mapping of team name variations and abbreviations to their full canonical names
TEAM_NAME_MAP = {
    # Atlanta
    'ATL': 'Atlanta Dream',
    'Atlanta': 'Atlanta Dream',
    'Atlanta Dream': 'Atlanta Dream',
    'Dream': 'Atlanta Dream',
    # Chicago
    'CHI': 'Chicago Sky',
    'Chicago': 'Chicago Sky',
    'Chicago Sky': 'Chicago Sky',
    'Sky': 'Chicago Sky',
    # Connecticut
    'CON': 'Connecticut Sun',
    'Connecticut': 'Connecticut Sun',
    'Connecticut Sun': 'Connecticut Sun',
    'Sun': 'Connecticut Sun',
    'Conn. Sun': 'Connecticut Sun',
    # Dallas
    'DAL': 'Dallas Wings',
    'Dallas': 'Dallas Wings',
    'Dallas Wings': 'Dallas Wings',
    'Wings': 'Dallas Wings',
    # Indiana
    'IND': 'Indiana Fever',
    'Indiana': 'Indiana Fever',
    'Indiana Fever': 'Indiana Fever',
    'Fever': 'Indiana Fever',
    # Los Angeles
    'LAS': 'Los Angeles Sparks',
    'Los Angeles': 'Los Angeles Sparks',
    'Los Angeles Sparks': 'Los Angeles Sparks',
    'Sparks': 'Los Angeles Sparks',
    'L.A. Sparks': 'Los Angeles Sparks',
    # Las Vegas
    'LVA': 'Las Vegas Aces',
    'Las Vegas': 'Las Vegas Aces',
    'Las Vegas Aces': 'Las Vegas Aces',
    'Aces': 'Las Vegas Aces',
    # Minnesota
    'MIN': 'Minnesota Lynx',
    'Minnesota': 'Minnesota Lynx',
    'Minnesota Lynx': 'Minnesota Lynx',
    'Lynx': 'Minnesota Lynx',
    'Minn. Lynx': 'Minnesota Lynx',
    # New York
    'NYL': 'New York Liberty',
    'New York': 'New York Liberty',
    'New York Liberty': 'New York Liberty',
    'Liberty': 'New York Liberty',
    'NY Liberty': 'New York Liberty',
    'N.Y. Liberty': 'New York Liberty',
    # Phoenix
    'PHO': 'Phoenix Mercury',
    'PHX': 'Phoenix Mercury',
    'Phoenix': 'Phoenix Mercury',
    'Phoenix Mercury': 'Phoenix Mercury',
    'Mercury': 'Phoenix Mercury',
    'Phx Mercury': 'Phoenix Mercury',
    # Seattle
    'SEA': 'Seattle Storm',
    'Seattle': 'Seattle Storm',
    'Seattle Storm': 'Seattle Storm',
    'Storm': 'Seattle Storm',
    # Washington
    'WAS': 'Washington Mystics',
    'Washington': 'Washington Mystics',
    'Washington Mystics': 'Washington Mystics',
    'Mystics': 'Washington Mystics',
    'Wash. Mystics': 'Washington Mystics',
    # Golden State
    'GSV': 'Golden State Valkyries',
    'Golden State': 'Golden State Valkyries',
    'Golden State Valkyries': 'Golden State Valkyries',
    'Valkyries': 'Golden State Valkyries',
    # Toronto
    'TOR': 'Toronto Tempo',
    'Toronto': 'Toronto Tempo',
    'Toronto Tempo': 'Toronto Tempo',
    'Tempo': 'Toronto Tempo',
    # Portland
    'PDX': 'Portland Fire',
    'Portland': 'Portland Fire',
    'Portland Fire': 'Portland Fire',
    'Fire': 'Portland Fire',
    'POR': 'Portland Fire',
}

def standardize_and_calculate_metrics():
    """
    Reads matches and stats from the SQLite database wnba.db.
    Standardizes team names across raw_matches, player_stats, and injuries tables.
    Calculates possessions, pace, points scored, and conceded for both teams.
    Saves the metrics back to raw_matches.
    """
    print("Connecting to database...")
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Standardize player_stats team names
    print("Standardizing team names in player_stats table...")
    cursor.execute("SELECT id, Team FROM player_stats")
    player_rows = cursor.fetchall()
    player_updates = []
    for row_id, team in player_rows:
        canonical = TEAM_NAME_MAP.get(team, team)
        if canonical != team:
            player_updates.append((canonical, row_id))
    
    if player_updates:
        cursor.executemany("UPDATE player_stats SET Team = ? WHERE id = ?", player_updates)
        print(f"  Updated {len(player_updates)} records in player_stats.")

    # 2. Standardize injuries team names
    print("Standardizing team names in injuries table...")
    cursor.execute("SELECT id, Team FROM injuries")
    injury_rows = cursor.fetchall()
    injury_updates = []
    for row_id, team in injury_rows:
        canonical = TEAM_NAME_MAP.get(team, team)
        if canonical != team:
            injury_updates.append((canonical, row_id))
            
    if injury_updates:
        cursor.executemany("UPDATE injuries SET Team = ? WHERE id = ?", injury_updates)
        print(f"  Updated {len(injury_updates)} records in injuries.")

    # 3. Read raw_matches and calculate possessions, pace, points scored/conceded
    print("Processing raw_matches table...")
    cursor.execute("""
        SELECT 
            id, HomeTeam, AwayTeam, HomeScore, AwayScore, 
            HomeFGA, HomeFTA, HomeOREB, HomeTOV, HomeMIN,
            AwayFGA, AwayFTA, AwayOREB, AwayTOV, AwayMIN
        FROM raw_matches
    """)
    match_rows = cursor.fetchall()
    match_updates = []
    
    for row in match_rows:
        row_id = row[0]
        home_team = row[1]
        away_team = row[2]
        home_score = row[3]
        away_score = row[4]
        
        home_fga = row[5] if row[5] is not None else 0.0
        home_fta = row[6] if row[6] is not None else 0.0
        home_oreb = row[7] if row[7] is not None else 0.0
        home_tov = row[8] if row[8] is not None else 0.0
        home_min = row[9] if row[9] is not None else 200.0  # default to regulation minutes
        
        away_fga = row[10] if row[10] is not None else 0.0
        away_fta = row[11] if row[11] is not None else 0.0
        away_oreb = row[12] if row[12] is not None else 0.0
        away_tov = row[13] if row[13] is not None else 0.0
        away_min = row[14] if row[14] is not None else 200.0
        
        # Standardize team names
        canonical_home = TEAM_NAME_MAP.get(home_team, home_team)
        canonical_away = TEAM_NAME_MAP.get(away_team, away_team)
        
        # Calculate possessions
        # Possessions = 0.5 * (FGA + 0.44 * FTA - OREB + TOV) + 0.5 * (Opp_FGA + 0.44 * Opp_FTA - Opp_OREB + Opp_TOV)
        home_poss = 0.5 * (home_fga + 0.44 * home_fta - home_oreb + home_tov) + 0.5 * (away_fga + 0.44 * away_fta - away_oreb + away_tov)
        away_poss = home_poss
        
        # Calculate pace
        # Pace = 40 * Possessions / (Minutes_Played / 5)
        # Note: WNBA regulation length is 40 minutes, and Minutes_Played is total player minutes in the game (e.g. 200)
        home_pace = 40.0 * home_poss / (home_min / 5.0) if home_min > 0 else 0.0
        away_pace = 40.0 * away_poss / (away_min / 5.0) if away_min > 0 else 0.0
        
        # Points scored/conceded
        home_pts_scored = home_score
        home_pts_conceded = away_score
        away_pts_scored = away_score
        away_pts_conceded = home_score
        
        match_updates.append((
            canonical_home, canonical_away,
            home_poss, home_pace,
            away_poss, away_pace,
            home_pts_scored, home_pts_conceded,
            away_pts_scored, away_pts_conceded,
            row_id
        ))

    if match_updates:
        cursor.executemany("""
            UPDATE raw_matches SET
                HomeTeam = ?, AwayTeam = ?,
                HomePossessions = ?, HomePace = ?,
                AwayPossessions = ?, AwayPace = ?,
                HomePtsScored = ?, HomePtsConceded = ?,
                AwayPtsScored = ?, AwayPtsConceded = ?
            WHERE id = ?
        """, match_updates)
        print(f"  Successfully updated {len(match_updates)} match records.")

    conn.commit()
    conn.close()
    print("Database updates completed successfully.")

if __name__ == "__main__":
    standardize_and_calculate_metrics()
