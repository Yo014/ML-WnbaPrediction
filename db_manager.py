import sqlite3
import os

DB_NAME = "wnba.db"

def get_connection(db_path=DB_NAME):
    """
    Establishes and returns a connection to the SQLite database.
    """
    return sqlite3.connect(db_path)

def create_tables(db_path=DB_NAME):
    """
    Creates the required raw_matches, player_stats, and injuries tables.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create raw_matches table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT NOT NULL,
        HomeTeam TEXT NOT NULL,
        AwayTeam TEXT NOT NULL,
        HomeScore INTEGER NOT NULL,
        AwayScore INTEGER NOT NULL,
        HomeRef TEXT,
        AwayRef TEXT,
        CrewChief TEXT,
        BookieHomeOdds REAL,
        BookieAwayOdds REAL,
        OpeningSpread REAL,
        ClosingSpread REAL,
        OverUnder REAL,
        HomeFGA REAL,
        HomeFTA REAL,
        HomeOREB REAL,
        HomeTOV REAL,
        HomeMIN REAL,
        HomeFGM REAL,
        HomeFG3M REAL,
        HomeFTM REAL,
        HomeDREB REAL,
        HomePF REAL,
        AwayFGA REAL,
        AwayFTA REAL,
        AwayOREB REAL,
        AwayTOV REAL,
        AwayMIN REAL,
        AwayFGM REAL,
        AwayFG3M REAL,
        AwayFTM REAL,
        AwayDREB REAL,
        AwayPF REAL,
        HomePossessions REAL,
        HomePace REAL,
        AwayPossessions REAL,
        AwayPace REAL,
        HomePtsScored INTEGER,
        HomePtsConceded INTEGER,
        AwayPtsScored INTEGER,
        AwayPtsConceded INTEGER,
        IsFanduelOdds INTEGER DEFAULT 0,
        OverOdds REAL,
        UnderOdds REAL
    );
    """)
    
    # Create player_stats table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS player_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Season INTEGER NOT NULL,
        Player TEXT NOT NULL,
        Team TEXT NOT NULL,
        GP INTEGER,
        MIN REAL,
        PTS REAL,
        AST REAL,
        TRB REAL,
        USG_PCT REAL,
        BPM REAL,
        WS REAL
    );
    """)
    
    # Create injuries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS injuries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Team TEXT NOT NULL,
        Player TEXT NOT NULL,
        InjuryStatus TEXT,
        ExpectedReturnDate TEXT
    );
    """)

    # Create polymarket_odds table
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

    # Create confirmed_bets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS confirmed_bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        recommended_side TEXT NOT NULL,
        wager_type TEXT NOT NULL,
        wager_amount REAL NOT NULL,
        odds REAL NOT NULL,
        outcome TEXT,
        bankroll_change REAL,
        confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (match_date, home_team, away_team, recommended_side)
    );
    """)

    conn.commit()
    conn.close()

def create_indices(db_path=DB_NAME):
    """
    Sets up SQLite indices on Date, HomeTeam, and AwayTeam inside raw_matches.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON raw_matches(Date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_hometeam ON raw_matches(HomeTeam);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_awayteam ON raw_matches(AwayTeam);")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_poly_match ON polymarket_odds(match_date, home_team, away_team);")
    
    conn.commit()
    conn.close()

def initialize_db(db_path=DB_NAME):
    """
    Runs the full database setup.
    """
    create_tables(db_path)
    create_indices(db_path)

if __name__ == "__main__":
    print(f"Initializing SQLite database: {DB_NAME}")
    initialize_db()
    print("Database initialization complete.")
