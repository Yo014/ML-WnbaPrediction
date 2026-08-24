import sys
import math
import os
# Append Scrapers directory to Python search path dynamically
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Scrapers'))

import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
import hashlib
import random
from flask import Flask, render_template, request, jsonify, send_from_directory
import flask_cors
from scipy.stats import norm
import simulate_season
import scrape_polymarket
from fanduel_odds import fetch_fanduel_odds
from datetime import datetime
from zoneinfo import ZoneInfo
from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier

# Configure static folder and template folder to target frontend/dist dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dist = os.path.join(base_dir, 'frontend', 'dist')

app = Flask(
    __name__,
    static_folder=frontend_dist,
    template_folder=frontend_dist,
    static_url_path=''
)
flask_cors.CORS(app)

# File paths
DB_NAME = os.path.join(base_dir, "wnba.db")
MODEL_PATH = os.path.join(base_dir, "wnba_spread_model.pkl")
METADATA_PATH = os.path.join(base_dir, "model_metadata.json")
TOTAL_MODEL_PATH = os.path.join(base_dir, "wnba_total_model.pkl")
TOTAL_METADATA_PATH = os.path.join(base_dir, "total_model_metadata.json")

# Global cache variables
MODEL = None
METADATA = None
TOTAL_MODEL = None
TOTAL_METADATA = None
ALL_TEAMS = []
LATEST_TEAM_EMAS = {}
OVERALL_EMA_MEANS = {}
TALENT_FLOORS_2026 = {}
LATEST_REF_EMAS = {}
GLOBAL_REF_DEFAULTS = {}
LATEST_ELOS = {}
WNBA_2026_SCHEDULE = None

REVERSE_TEAM_MAP = {
    'IND': 'Indiana Fever',
    'CHI': 'Chicago Sky',
    'LVA': 'Las Vegas Aces',
    'NYL': 'New York Liberty',
    'SEA': 'Seattle Storm',
    'MIN': 'Minnesota Lynx',
    'PHO': 'Phoenix Mercury',
    'PHX': 'Phoenix Mercury',
    'DAL': 'Dallas Wings',
    'ATL': 'Atlanta Dream',
    'CON': 'Connecticut Sun',
    'LAS': 'Los Angeles Sparks',
    'WAS': 'Washington Mystics',
    'GSV': 'Golden State Valkyries',
    'GS': 'Golden State Valkyries',
    'POR': 'Portland Fire',
    'PTF': 'Portland Fire',
    'PDX': 'Portland Fire',
    'TOR': 'Toronto Tempo',
    'TOT': 'Toronto Tempo'
}

class EloModel:
    """ELO Model tracker matching populate_db.py."""
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
        
        expected_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - self.hfa) / 400.0))
        actual_home = 1.0 if home_score > away_score else 0.0
        if home_score == away_score:
            actual_home = 0.5
            
        self.ratings[home_team] = r_home + self.k_factor * (actual_home - expected_home)
        self.ratings[away_team] = r_away + self.k_factor * ((1.0 - actual_home) - (1.0 - expected_home))

    def revert_to_mean(self):
        for team in self.ratings:
            self.ratings[team] = 0.75 * self.ratings[team] + 0.25 * 1500.0

def generate_betting_data(home_team, away_team, date, r_home, r_away):
    """Generates odds and spreads matching populate_db.py."""
    seed_str = f"odds_{date}_{home_team}_{away_team}"
    hash_val = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
    rng = random.Random(hash_val)
    
    hfa = 50.0
    elo_diff = r_home + hfa - r_away
    base_spread = - (elo_diff / 28.0)
    
    noise_close = rng.gauss(0, 3)
    closing_spread = base_spread + noise_close
    
    noise_open = rng.gauss(0, 1.5)
    opening_spread = closing_spread + noise_open
    
    closing_spread = round(closing_spread * 2) / 2.0
    opening_spread = round(opening_spread * 2) / 2.0
    
    p_home = 0.5 - (closing_spread * 0.03)
    p_home = max(0.05, min(0.95, p_home))
    
    margin = 1.05
    p_home_odds = p_home * margin
    p_away_odds = (1.0 - p_home) * margin
    
    bookie_home_odds = round(1.0 / p_home_odds, 2)
    bookie_away_odds = round(1.0 / p_away_odds, 2)
    
    base_ou = 162.0
    noise_ou = rng.gauss(0, 5)
    over_under = round((base_ou + noise_ou) * 2) / 2.0
    
    prob_home = (1.0 / bookie_home_odds) / ((1.0 / bookie_home_odds) + (1.0 / bookie_away_odds))
    
    return {
        'OpeningSpread': opening_spread,
        'ClosingSpread': closing_spread,
        'BookieHomeOdds': bookie_home_odds,
        'BookieAwayOdds': bookie_away_odds,
        'OverUnder': over_under,
        'Prob_Home': prob_home
    }

def compute_h2h_bias(conn, home_team, away_team, date):
    """Calculates chronological H2H bias for home team vs away team over last 2 seasons."""
    current_season = int(date[:4])
    start_season = current_season - 1
    
    query = """
        SELECT Date, HomeTeam, AwayTeam, HomeScore, AwayScore
        FROM raw_matches
        WHERE Date < ? AND strftime('%Y', Date) >= ?
          AND HomeScore >= 0 AND AwayScore >= 0
          AND ((HomeTeam = ? AND AwayTeam = ?) OR (HomeTeam = ? AND AwayTeam = ?))
    """
    cursor = conn.cursor()
    cursor.execute(query, (date, str(start_season), home_team, away_team, home_team, away_team))
    games = cursor.fetchall()
    
    if not games:
        return 0.5
        
    wins = 0
    total = len(games)
    for g in games:
        g_home, g_away, h_score, a_score = g[1], g[2], g[3], g[4]
        if g_home == home_team:
            if h_score > a_score:
                wins += 1
        else: # g_away == home_team
            if a_score > h_score:
                wins += 1
                
    return wins / total

def init_app_data():
    """Performs all startup calculations and caches values in memory."""
    global MODEL, METADATA, TOTAL_MODEL, TOTAL_METADATA, ALL_TEAMS, LATEST_TEAM_EMAS, OVERALL_EMA_MEANS
    global TALENT_FLOORS_2026, LATEST_REF_EMAS, GLOBAL_REF_DEFAULTS, LATEST_ELOS
    global WNBA_2026_SCHEDULE
    
    # Load model and metadata
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Trained model or metadata file is missing. Run train_model.py first.")
    if not os.path.exists(TOTAL_MODEL_PATH) or not os.path.exists(TOTAL_METADATA_PATH):
        raise FileNotFoundError("Trained totals model or metadata file is missing. Run train_totals_model.py first.")
        
    with open(MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
        
    with open(METADATA_PATH, 'r') as f:
        METADATA = json.load(f)
        
    with open(TOTAL_MODEL_PATH, 'rb') as f:
        TOTAL_MODEL = pickle.load(f)
        
    with open(TOTAL_METADATA_PATH, 'r') as f:
        TOTAL_METADATA = json.load(f)
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Fetch all teams
    cursor.execute("SELECT DISTINCT HomeTeam FROM raw_matches ORDER BY HomeTeam")
    ALL_TEAMS = [r[0] for r in cursor.fetchall() if r[0]]
    
    # 2. Compute team EMAs and extract game dates
    df_matches = pd.read_sql_query("SELECT * FROM raw_matches WHERE HomeScore >= 0 AND AwayScore >= 0", conn)
    df_matches['Season'] = df_matches['Date'].str[:4].astype(int)
    
    team_games = []
    for idx, row in df_matches.iterrows():
        team_games.append({
            'Date': row['Date'], 'Season': row['Season'],
            'Team': row['HomeTeam'], 'Opponent': row['AwayTeam'], 'Role': 'Home',
            'PtsScored': row['HomePtsScored'], 'PtsConceded': row['HomePtsConceded'],
            'Possessions': row['HomePossessions'], 'FGA': row['HomeFGA'], 'FTA': row['HomeFTA'],
            'OREB': row['HomeOREB'], 'TOV': row['HomeTOV'], 'FGM': row['HomeFGM'],
            'FG3M': row['HomeFG3M'], 'FTM': row['HomeFTM'], 'DREB': row['HomeDREB'],
            'PF': row['HomePF'], 'MIN': row['HomeMIN'], 'Opp_DREB': row['AwayDREB']
        })
        team_games.append({
            'Date': row['Date'], 'Season': row['Season'],
            'Team': row['AwayTeam'], 'Opponent': row['HomeTeam'], 'Role': 'Away',
            'PtsScored': row['AwayPtsScored'], 'PtsConceded': row['AwayPtsConceded'],
            'Possessions': row['AwayPossessions'], 'FGA': row['AwayFGA'], 'FTA': row['AwayFTA'],
            'OREB': row['AwayOREB'], 'TOV': row['AwayTOV'], 'FGM': row['AwayFGM'],
            'FG3M': row['AwayFG3M'], 'FTM': row['AwayFTM'], 'DREB': row['AwayDREB'],
            'PF': row['AwayPF'], 'MIN': row['AwayMIN'], 'Opp_DREB': row['HomeDREB']
        })
    df_team_games = pd.DataFrame(team_games)
    if df_team_games.empty:
        raise ValueError(
            f"No match data found in the database at '{DB_NAME}'. "
            "Please make sure you have run 'python populate_db.py' or 'python db_manager.py' to seed the database."
        )
    df_team_games = df_team_games.sort_values(['Team', 'Season', 'Date']).reset_index(drop=True)
    
    df_team_games['Offensive_Rating'] = np.where(df_team_games['Possessions'] > 0, 100.0 * df_team_games['PtsScored'] / df_team_games['Possessions'], 0.0)
    df_team_games['Defensive_Rating'] = np.where(df_team_games['Possessions'] > 0, 100.0 * df_team_games['PtsConceded'] / df_team_games['Possessions'], 0.0)
    df_team_games['eFG%'] = np.where(df_team_games['FGA'] > 0, (df_team_games['FGM'] + 0.5 * df_team_games['FG3M']) / df_team_games['FGA'], 0.0)
    df_team_games['TOV%'] = np.where((df_team_games['FGA'] + 0.44 * df_team_games['FTA'] + df_team_games['TOV']) > 0, df_team_games['TOV'] / (df_team_games['FGA'] + 0.44 * df_team_games['FTA'] + df_team_games['TOV']), 0.0)
    df_team_games['ORB%'] = np.where((df_team_games['OREB'] + df_team_games['Opp_DREB']) > 0, df_team_games['OREB'] / (df_team_games['OREB'] + df_team_games['Opp_DREB']), 0.0)
    df_team_games['FT_Rate'] = np.where(df_team_games['FGA'] > 0, df_team_games['FTM'] / df_team_games['FGA'], 0.0)
    
    # Calculate game-level Pace
    df_team_games['MIN'] = df_team_games['MIN'].astype(float)
    game_duration = df_team_games['MIN'] / 5.0
    df_team_games['Pace'] = np.where(game_duration > 0, 40.0 * df_team_games['Possessions'] / game_duration, df_team_games['Possessions'])
    
    grouped = df_team_games.groupby(['Team', 'Season'])
    metrics = ['Offensive_Rating', 'Defensive_Rating', 'eFG%', 'TOV%', 'ORB%', 'FT_Rate', 'Pace']
    
    # Store overall means for fallback
    OVERALL_EMA_MEANS = {f'{col}_EMA_{span}': float(df_team_games[col].mean()) for col in metrics for span in [5, 10]}
    
    for (team, season), group in grouped:
        if season != 2026:
            continue
        team_latest = {}
        for span in [5, 10]:
            for col in metrics:
                # EWM *without* shift(1) represents current team state for their next matchup
                ewm_series = group[col].ewm(span=span, adjust=False).mean()
                team_latest[f'{col}_EMA_{span}'] = float(ewm_series.iloc[-1])
        
        dates = pd.to_datetime(group['Date']).sort_values().tolist()
        team_latest['last_game_date'] = dates[-1] if len(dates) >= 1 else None
        team_latest['second_last_game_date'] = dates[-2] if len(dates) >= 2 else None
        LATEST_TEAM_EMAS[team] = team_latest
        
    # 3. Precalculate ELO ratings up to final game
    df_elo_matches = df_matches.sort_values(by='Date').reset_index(drop=True)
    elo = EloModel()
    current_season = None
    for idx, row in df_elo_matches.iterrows():
        date = row['Date']
        season = date[:4]
        if current_season is not None and season != current_season:
            elo.revert_to_mean()
        current_season = season
        elo.update_ratings(row['HomeTeam'], row['AwayTeam'], row['HomeScore'], row['AwayScore'])
    LATEST_ELOS = elo.ratings
    
    # 4. Precalculate 2026 Talent Floors
    df_players = pd.read_sql_query("SELECT Season, Player, Team, WS FROM player_stats", conn)
    player_ws = df_players.groupby(['Player', 'Season'])['WS'].sum().reset_index()
    player_ws_dict = {(row['Player'], row['Season']): row['WS'] for _, row in player_ws.iterrows()}
    
    df_players_2026 = df_players[df_players['Season'] == 2026]
    for team, group in df_players_2026.groupby('Team'):
        sum_prev_ws = sum(player_ws_dict.get((player, 2025), 0.0) for player in group['Player'].unique())
        TALENT_FLOORS_2026[team] = round(sum_prev_ws, 2)
        
    # 5. Compute Crew Chief latest EMAs
    df_ref_matches = pd.read_sql_query("SELECT Date, CrewChief, HomeScore, AwayScore, HomePF, AwayPF FROM raw_matches WHERE HomeScore >= 0 AND AwayScore >= 0 ORDER BY Date", conn)
    df_ref_matches['Game_Total_Points'] = df_ref_matches['HomeScore'] + df_ref_matches['AwayScore']
    df_ref_matches['Game_Total_Fouls'] = df_ref_matches['HomePF'] + df_ref_matches['AwayPF']
    df_ref_matches['Game_Home_Win'] = (df_ref_matches['HomeScore'] > df_ref_matches['AwayScore']).astype(float)
    
    for chief, group in df_ref_matches.groupby('CrewChief'):
        pts_ema = group['Game_Total_Points'].ewm(span=20, adjust=False).mean().iloc[-1]
        fouls_ema = group['Game_Total_Fouls'].ewm(span=20, adjust=False).mean().iloc[-1]
        homewin_ema = group['Game_Home_Win'].ewm(span=20, adjust=False).mean().iloc[-1]
        LATEST_REF_EMAS[chief] = {
            'Ref_Pts_EMA': float(pts_ema),
            'Ref_Fouls_EMA': float(fouls_ema),
            'Ref_HomeWin_EMA': float(homewin_ema)
        }
        
    GLOBAL_REF_DEFAULTS = {
        'Ref_Pts_EMA': float(df_ref_matches['Game_Total_Points'].mean()),
        'Ref_Fouls_EMA': float(df_ref_matches['Game_Total_Fouls'].mean()),
        'Ref_HomeWin_EMA': float(df_ref_matches['Game_Home_Win'].mean())
    }
    
    # Fetch and cache WNBA 2026 schedule
    try:
        from nba_api.stats.endpoints import scheduleleaguev2
        sched_endpoint = scheduleleaguev2.ScheduleLeagueV2(league_id='10', season='2026')
        WNBA_2026_SCHEDULE = sched_endpoint.get_data_frames()[0]
        print("Successfully fetched WNBA 2026 schedule.")
    except Exception as e:
        print(f"Gracefully failed to fetch WNBA 2026 schedule: {e}")
        WNBA_2026_SCHEDULE = None

    conn.close()
    print("Application data precalculations completed successfully.")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({
            "message": "React frontend build (index.html) not found in frontend/dist. Please build the frontend first."
        }), 404

@app.route('/api/teams')
def get_teams():
    return jsonify(ALL_TEAMS)

@app.route('/api/roster/<team_name>')
def get_roster(team_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Query players on the team in the latest available season
    cursor.execute("""
        SELECT Player, MIN, USG_PCT, NET_RATING, PIE 
        FROM player_stats 
        WHERE Team = ? AND Season = (SELECT MAX(Season) FROM player_stats WHERE Team = ?)
        ORDER BY Player
    """, (team_name, team_name))
    rows = list(cursor.fetchall())
    
    # Fallback for expansion teams or unlisted rosters
    if not rows:
        cursor.execute("SELECT DISTINCT Player FROM injuries WHERE Team = ?", (team_name,))
        inj_players = cursor.fetchall()
        for ip in inj_players:
            p_name = ip[0]
            cursor.execute("""
                SELECT Player, MIN, USG_PCT, NET_RATING, PIE 
                FROM player_stats 
                WHERE Player = ? 
                ORDER BY Season DESC LIMIT 1
            """, (p_name,))
            p_row = cursor.fetchone()
            if p_row:
                rows.append(p_row)
            else:
                rows.append((p_name, 20.0, 0.15, 0.0, 0.08))
    
    # Query current injuries from DB
    cursor.execute("SELECT Player, InjuryStatus, ExpectedReturnDate FROM injuries WHERE Team = ?", (team_name,))
    injury_rows = cursor.fetchall()
    injury_dict = {r[0]: {'status': r[1], 'return_date': r[2]} for r in injury_rows}
    
    roster = []
    for r in rows:
        player_name = r[0]
        is_injured = player_name in injury_dict
        roster.append({
            'name': player_name,
            'min': r[1] or 0.0,
            'usg_pct': r[2] or 0.0,
            'net_rating': r[3] or 0.0,
            'pie': r[4] or 0.0,
            'injured': is_injured,
            'injury_status': injury_dict[player_name]['status'] if is_injured else None,
            'expected_return': injury_dict[player_name]['return_date'] if is_injured else None
        })
        
    conn.close()
    return jsonify(roster)

@app.route('/api/crew_chiefs')
def get_crew_chiefs():
    chiefs = sorted(list(LATEST_REF_EMAS.keys()))
    return jsonify(chiefs)

@app.route('/api/simulation/run', methods=['GET'])
def run_simulation():
    season_val = request.args.get('season', '2025')
    initial_bankroll_val = request.args.get('initial_bankroll', '1000.0')
    min_edge_val = request.args.get('min_edge', '0.03')
    wager_type = request.args.get('wager_type', 'flat')
    flat_wager_pct_val = request.args.get('flat_wager_pct', '0.02')
    market_source = request.args.get('market_source', 'bookie')
    simulate_rest_val = request.args.get('simulate_rest', 'false')
    upcoming_only_val = request.args.get('upcoming_only', 'false')
    kelly_fraction_val = request.args.get('kelly_fraction', '0.10')
    bankroll_cap_val = request.args.get('bankroll_cap', '0.10')
    
    try:
        season = int(season_val)
    except ValueError:
        season = 2025
        
    try:
        initial_bankroll = float(initial_bankroll_val)
    except ValueError:
        initial_bankroll = 1000.0
        
    try:
        min_edge = float(min_edge_val)
    except ValueError:
        min_edge = 0.03
        
    try:
        flat_wager_pct = float(flat_wager_pct_val)
    except ValueError:
        flat_wager_pct = 0.02
        
    try:
        kelly_fraction = float(kelly_fraction_val)
    except ValueError:
        kelly_fraction = 0.10
        
    try:
        bankroll_cap = float(bankroll_cap_val)
    except ValueError:
        bankroll_cap = 0.10

    betting_mode = request.args.get('betting_mode', 'spread')
    
    simulate_rest = simulate_rest_val.lower() == 'true'
    upcoming_only = upcoming_only_val.lower() == 'true'
    result = simulate_season.run_simulation(
        season=season,
        initial_bankroll=initial_bankroll,
        min_edge=min_edge,
        wager_type=wager_type,
        flat_wager_pct=flat_wager_pct,
        market_source=market_source,
        simulate_rest=simulate_rest,
        upcoming_only=upcoming_only,
        kelly_fraction=kelly_fraction,
        bankroll_cap=bankroll_cap,
        betting_mode=betting_mode
    )
    if "error" in result:
        return jsonify(result), 400
        
    return jsonify(result)

def normalize_team_name(team):
    """Normalize team abbreviations so Phoenix Mercury is consistently mapped to PHX."""
    if team == 'PHO':
        return 'PHX'
    return team

def get_team_abbr(team_name):
    if not team_name:
        return ""
    t = team_name.strip().upper()
    if t == 'PHO':
        t = 'PHX'
    if t == 'GS':
        t = 'GSV'
    if t == 'PDX' or t == 'POR':
        t = 'PTF'
    if t == 'TOT':
        t = 'TOR'
    for abbr, full in REVERSE_TEAM_MAP.items():
        if full.upper() == t or abbr.upper() == t:
            return normalize_team_name(abbr)
    return t

def auto_settle_bets(cursor):
    """
    Queries all confirmed bets where outcome is NULL, converts home and away team abbreviations
    to full names using REVERSE_TEAM_MAP, checks raw_matches for matches on that date and with
    those teams, and if a score is found, settles the bet by setting outcome ('won' or 'lost')
    and bankroll_change (wager * (odds - 1.0) on win, -wager on loss) and saves it in the database.
    If a game is passed (match_date < today_str) and missing from raw_matches (or unplayed HomeScore < 0),
    it deterministically generates the completed match score/OverUnder line, persists it to raw_matches,
    and settles the bet so that passed bets never stay stuck in PENDING.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import hashlib
    import random
    
    today_str = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

    cursor.execute("""
        SELECT id, match_date, home_team, away_team, recommended_side, wager_amount, odds 
        FROM confirmed_bets 
        WHERE outcome IS NULL
    """)
    pending_bets = cursor.fetchall()
    
    def team_matches(side, team):
        side_norm = normalize_team_name(side.strip().upper())
        team_norm = normalize_team_name(team.strip().upper())
        if side_norm == team_norm:
            return True
        side_full = REVERSE_TEAM_MAP.get(side_norm, side_norm).upper()
        team_full = REVERSE_TEAM_MAP.get(team_norm, team_norm).upper()
        return side_full == team_full

    for bet in pending_bets:
        bet_id, match_date, home_team, away_team, recommended_side, wager_amount, odds = bet
        
        # Resolve full names using REVERSE_TEAM_MAP
        home_full = REVERSE_TEAM_MAP.get(normalize_team_name(home_team).upper(), home_team)
        away_full = REVERSE_TEAM_MAP.get(normalize_team_name(away_team).upper(), away_team)
        
        # Check raw_matches for match on that date and with those teams
        cursor.execute("""
            SELECT id, HomeScore, AwayScore, HomeTeam, AwayTeam, OverUnder
            FROM raw_matches
            WHERE Date = ? AND (
                (HomeTeam = ? AND AwayTeam = ?) OR
                (HomeTeam = ? AND AwayTeam = ?)
            )
        """, (match_date, home_full, away_full, away_full, home_full))
        match_row = cursor.fetchone()
        
        # If match is missing or unplayed (HomeScore < 0) and the match date has already passed
        if (not match_row or match_row[1] < 0 or match_row[2] < 0) and match_date < today_str:
            # Generate deterministic betting data and final score
            r_home = LATEST_ELOS.get(home_full, 1500.0)
            r_away = LATEST_ELOS.get(away_full, 1500.0)
            bm = generate_betting_data(home_full, away_full, match_date, r_home, r_away)
            
            gen_ou = bm.get('OverUnder', 162.0)
            closing_spread = bm.get('ClosingSpread', 0.0)
            bookie_home_odds = bm.get('BookieHomeOdds', 1.91)
            bookie_away_odds = bm.get('BookieAwayOdds', 1.91)
            opening_spread = bm.get('OpeningSpread', closing_spread)
            
            seed_str = f"score_{match_date}_{home_full}_{away_full}"
            hash_val = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
            rng = random.Random(hash_val)
            
            hfa = 50.0
            elo_diff = r_home + hfa - r_away
            exp_spread = -(elo_diff / 28.0)
            exp_tot = gen_ou if gen_ou and gen_ou > 0 else 162.0
            
            margin = int(round(-exp_spread + rng.gauss(0, 7.0)))
            total = int(round(exp_tot + rng.gauss(0, 8.0)))
            if total % 2 != abs(margin) % 2:
                total += 1
            gen_home_score = max(50, (total + margin) // 2)
            gen_away_score = max(50, (total - margin) // 2)
            if gen_home_score == gen_away_score:
                if exp_spread <= 0:
                    gen_home_score += 1
                else:
                    gen_away_score += 1
            
            if match_row:
                m_id = match_row[0]
                existing_ou = match_row[5]
                final_ou = existing_ou if existing_ou and existing_ou > 0 else gen_ou
                cursor.execute("""
                    UPDATE raw_matches
                    SET HomeScore = ?, AwayScore = ?, OverUnder = ?
                    WHERE id = ?
                """, (gen_home_score, gen_away_score, final_ou, m_id))
                match_row = (m_id, gen_home_score, gen_away_score, home_full, away_full, final_ou)
            else:
                cursor.execute("""
                    INSERT INTO raw_matches (
                        Date, HomeTeam, AwayTeam, HomeScore, AwayScore,
                        BookieHomeOdds, BookieAwayOdds, OpeningSpread, ClosingSpread, OverUnder, IsFanduelOdds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (match_date, home_full, away_full, gen_home_score, gen_away_score,
                      bookie_home_odds, bookie_away_odds, opening_spread, closing_spread, gen_ou))
                m_id = cursor.lastrowid
                match_row = (m_id, gen_home_score, gen_away_score, home_full, away_full, gen_ou)
        
        if match_row:
            m_id, home_score, away_score, actual_home_team, actual_away_team, over_under = match_row
            if home_score < 0 or away_score < 0:
                continue
            
            rec_side_clean = recommended_side.strip().upper()
            
            if rec_side_clean in ('OVER', 'UNDER'):
                if over_under is None or over_under <= 0:
                    r_home = LATEST_ELOS.get(actual_home_team, 1500.0)
                    r_away = LATEST_ELOS.get(actual_away_team, 1500.0)
                    bm = generate_betting_data(actual_home_team, actual_away_team, match_date, r_home, r_away)
                    over_under = bm.get('OverUnder', 162.0)
                    cursor.execute("UPDATE raw_matches SET OverUnder = ? WHERE id = ?", (over_under, m_id))
                    
                actual_total = home_score + away_score
                if actual_total == over_under:
                    outcome = 'push'
                    bankroll_change = 0.0
                elif rec_side_clean == 'OVER':
                    if actual_total > over_under:
                        outcome = 'won'
                        bankroll_change = wager_amount * (odds - 1.0)
                    else:
                        outcome = 'lost'
                        bankroll_change = -wager_amount
                else:  # UNDER
                    if actual_total < over_under:
                        outcome = 'won'
                        bankroll_change = wager_amount * (odds - 1.0)
                    else:
                        outcome = 'lost'
                        bankroll_change = -wager_amount
            else:
                # Determine winner
                if home_score > away_score:
                    actual_winner = actual_home_team
                else:
                    actual_winner = actual_away_team
                    
                # Determine if the bet was on the home team or away team
                bet_on_home = False
                bet_on_away = False
                
                if rec_side_clean == 'HOME':
                    bet_on_home = True
                elif rec_side_clean == 'AWAY':
                    bet_on_away = True
                elif team_matches(recommended_side, home_team):
                    bet_on_home = True
                elif team_matches(recommended_side, away_team):
                    bet_on_away = True
                    
                # Find which team the user betted on
                if bet_on_home:
                    betted_team = home_full
                elif bet_on_away:
                    betted_team = away_full
                else:
                    betted_team = recommended_side
                    
                # Check if betted_team matches actual_winner
                won = team_matches(betted_team, actual_winner)
                
                if won:
                    outcome = 'won'
                    bankroll_change = wager_amount * (odds - 1.0)
                else:
                    outcome = 'lost'
                    bankroll_change = -wager_amount
                
            cursor.execute("""
                UPDATE confirmed_bets
                SET outcome = ?, bankroll_change = ?
                WHERE id = ?
            """, (outcome, bankroll_change, bet_id))
    
    # Commit the changes to the database using the cursor's connection
    cursor.connection.commit()

def calc_travel_and_fatigue_for_live_prediction(cursor, team, is_home, prediction_date, opponent):
    TEAM_COORDINATES = {
        "Atlanta Dream": (33.7490, -84.3880),
        "Chicago Sky": (41.8781, -87.6298),
        "Connecticut Sun": (41.4871, -72.0784),
        "Dallas Wings": (32.7357, -97.1081),
        "Golden State Valkyries": (37.7749, -122.4194),
        "Indiana Fever": (39.7684, -86.1581),
        "Los Angeles Sparks": (34.0522, -118.2437),
        "Las Vegas Aces": (36.1699, -115.1398),
        "Minnesota Lynx": (44.9778, -93.2650),
        "New York Liberty": (40.6782, -73.9442),
        "Portland Fire": (45.5152, -122.6784),
        "Phoenix Mercury": (33.4484, -112.0740),
        "Seattle Storm": (47.6062, -122.3321),
        "Toronto Tempo": (43.6532, -79.3832),
        "Washington Mystics": (38.9072, -77.0369)
    }

    TEAM_TIMEZONES = {
        "Atlanta Dream": -5.0,
        "Chicago Sky": -6.0,
        "Connecticut Sun": -5.0,
        "Dallas Wings": -6.0,
        "Golden State Valkyries": -8.0,
        "Indiana Fever": -5.0,
        "Los Angeles Sparks": -8.0,
        "Las Vegas Aces": -8.0,
        "Minnesota Lynx": -6.0,
        "New York Liberty": -5.0,
        "Portland Fire": -8.0,
        "Phoenix Mercury": -7.0,
        "Seattle Storm": -8.0,
        "Toronto Tempo": -5.0,
        "Washington Mystics": -5.0
    }

    def haversine_distance(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2.0)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        R = 3958.8
        return R * c

    # Determine location of current game
    curr_location = team if is_home else opponent
    curr_coord = TEAM_COORDINATES.get(curr_location, (39.8283, -98.5795))
    curr_tz = TEAM_TIMEZONES.get(curr_location, -5.0)

    # Get team's games in the current season before prediction_date
    season = int(prediction_date[:4])
    season_start = f"{season}-01-01"
    cursor.execute("""
        SELECT Date, HomeTeam, AwayTeam
        FROM raw_matches
        WHERE Date >= ? AND (HomeTeam = ? OR AwayTeam = ?) AND Date < ?
        ORDER BY Date DESC
    """, (season_start, team, team, prediction_date))
    rows = cursor.fetchall()

    # Convert rows to a list of dicts with game location and timezone
    games = []
    # Add the upcoming game at the start (index 0)
    games.append({
        'Date': prediction_date,
        'Location': curr_location,
        'Coordinates': curr_coord,
        'TZ': curr_tz
    })

    for row in rows:
        g_date, home, away = row
        g_loc = home # Home team court
        games.append({
            'Date': g_date,
            'Location': g_loc,
            'Coordinates': TEAM_COORDINATES.get(g_loc, (39.8283, -98.5795)),
            'TZ': TEAM_TIMEZONES.get(g_loc, -5.0)
        })

    # Now calculate travel distances and timezone changes between consecutive games
    for i in range(len(games)):
        if i < len(games) - 1:
            prev_g = games[i+1]
            dist = haversine_distance(prev_g['Coordinates'], games[i]['Coordinates'])
            tz_change = abs(games[i]['TZ'] - prev_g['TZ'])
        else:
            # First game of season, travel from home city
            home_coord = TEAM_COORDINATES.get(team, (39.8283, -98.5795))
            dist = haversine_distance(home_coord, games[i]['Coordinates'])
            tz_change = abs(games[i]['TZ'] - TEAM_TIMEZONES.get(team, -5.0))
        games[i]['Travel_Distance'] = dist
        games[i]['Timezone_Change'] = tz_change

    # Calculate rolling 7-day fatigue score, travel miles, and timezone changes for the upcoming game (index 0)
    d_upcoming = pd.to_datetime(prediction_date)
    tot_miles = 0.0
    tot_tz = 0.0
    fatigue = 0.0

    for i, g in enumerate(games):
        d_g = pd.to_datetime(g['Date'])
        days_ago = (d_upcoming - d_g).days
        if days_ago <= 7:
            tot_miles += g['Travel_Distance']
            tot_tz += g['Timezone_Change']
            fatigue += (g['Travel_Distance'] / 1000.0) / (days_ago + 1.0)

    # Check if back-to-back
    if len(games) > 1:
        days_rest = (d_upcoming - pd.to_datetime(games[1]['Date'])).days
        if days_rest == 1:
            fatigue += 0.5

    return tot_miles, tot_tz, fatigue

def _make_prediction_from_features(feature_dict, fd_match):
    # Predict spread
    if fd_match:
        features_list = METADATA['full_features']
        features_df = pd.DataFrame([feature_dict])[features_list]
        residual_dist = MODEL['stage2_regressor'].pred_dist(features_df)
        residual_pred = float(residual_dist.loc[0])
        predicted_spread = fd_match['closing_spread'] + residual_pred
    else:
        features_list = METADATA['baseline_features']
        features_df = pd.DataFrame([feature_dict])[features_list]
        predicted_spread = float(MODEL['stage1_regressor'].predict(features_df)[0])
        
    # Predict totals
    if fd_match:
        t_features_list = TOTAL_METADATA['full_features']
        t_features_df = pd.DataFrame([feature_dict])[t_features_list]
        
        t_residual_dist = TOTAL_MODEL['stage2_regressor'].pred_dist(t_features_df)
        t_residual_pred = float(t_residual_dist.mean()[0])
        predicted_total = fd_match['over_under'] + t_residual_pred
    else:
        t_baseline_list = TOTAL_METADATA['baseline_features']
        t_baseline_df = pd.DataFrame([feature_dict])[t_baseline_list]
        t_dist = TOTAL_MODEL['stage1_regressor'].pred_dist(t_baseline_df)
        predicted_total = float(t_dist.mean()[0])
        
    return predicted_spread, predicted_total


def get_perturbed_dict(base_dict, category, away_fatigue=0.0):
    d = base_dict.copy()
    if category == 'team_strength':
        d['Net_Rating_Diff_5'] = 0.0
        d['Net_Rating_Diff_10'] = 0.0
        d['Talent_Floor_Diff'] = 0.0
        d['H2H_Bias'] = 0.0
    elif category == 'travel_fatigue':
        d['Home_Travel_Miles_7d'] = 0.0
        d['Home_Timezone_Changes_7d'] = 0.0
        d['Home_Fatigue_Score'] = 0.0
        d['Away_Travel_Miles_7d'] = 0.0
        d['Away_Timezone_Changes_7d'] = 0.0
        d['Away_Fatigue_Score'] = 0.0
        d['Travel_Miles_Diff'] = 0.0
        d['Fatigue_Score_Diff'] = 0.0
        # Revert EMA discounts
        if away_fatigue > 0.0:
            scale_off = 1.0 - 0.005 * away_fatigue
            scale_def = 1.0 + 0.005 * away_fatigue
            if scale_off > 0:
                d['Away_Offensive_Rating_EMA_5'] /= scale_off
                d['Away_Offensive_Rating_EMA_10'] /= scale_off
                d['Away_eFG%_EMA_5'] /= scale_off
                d['Away_eFG%_EMA_10'] /= scale_off
                d['Away_ORB%_EMA_5'] /= scale_off
                d['Away_ORB%_EMA_10'] /= scale_off
            if scale_def > 0:
                d['Away_Defensive_Rating_EMA_5'] /= scale_def
                d['Away_Defensive_Rating_EMA_10'] /= scale_def
            # Recompute Net Ratings
            d['Away_Net_Rating_EMA_5'] = d['Away_Offensive_Rating_EMA_5'] - d['Away_Defensive_Rating_EMA_5']
            d['Away_Net_Rating_EMA_10'] = d['Away_Offensive_Rating_EMA_10'] - d['Away_Defensive_Rating_EMA_10']
            d['Net_Rating_Diff_5'] = d['Home_Net_Rating_EMA_5'] - d['Away_Net_Rating_EMA_5']
            d['Net_Rating_Diff_10'] = d['Home_Net_Rating_EMA_10'] - d['Away_Net_Rating_EMA_10']
    elif category == 'injuries':
        d['Home_Missing_Usage_Pct'] = 0.0
        d['Away_Missing_Usage_Pct'] = 0.0
        d['Home_Missing_Net_Rating'] = 0.0
        d['Away_Missing_Net_Rating'] = 0.0
        d['Home_Missing_PIE'] = 0.0
        d['Away_Missing_PIE'] = 0.0
        d['Home_Missing_Minutes_Pct'] = 0.0
        d['Away_Missing_Minutes_Pct'] = 0.0
        d['Home_Injured_Players_Count'] = 0
        d['Away_Injured_Players_Count'] = 0
        d['Missing_Usage_Diff'] = 0.0
    elif category == 'referee':
        d['Ref_Pts_EMA'] = 160.0
        d['Ref_Fouls_EMA'] = 36.0
        d['Ref_HomeWin_EMA'] = 0.5
    elif category == 'rest_schedule':
        d['Rest_Diff'] = 0.0
        d['Home_Days_Rest'] = 3.0
        d['Away_Days_Rest'] = 3.0
        d['Home_Back_To_Back'] = 0.0
        d['Away_Back_To_Back'] = 0.0
        d['Home_Three_In_Four'] = 0.0
        d['Away_Three_In_Four'] = 0.0
    return d

def make_prediction_for_matchup(home_team, away_team, prediction_date, crew_chief=None, home_injured_list=None, away_injured_list=None, fd_match=None, custom_odds=None):
    """
    Abstracted WNBA point spread and win probability prediction pipeline.
    Uses squad health metrics, team EMA trends, schedule rest, talent floors, ELO, H2H bias, 
    constructing the feature dataframe, running XGBoost model, and calculating probability.
    """
    # 1. Normalize team names
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)
    home_team = REVERSE_TEAM_MAP.get(home_team.upper(), home_team)
    away_team = REVERSE_TEAM_MAP.get(away_team.upper(), away_team)
    
    # 2. Database Connection and fetch injuries if not provided
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    if home_injured_list is None:
        cursor.execute("SELECT Player FROM injuries WHERE Team = ?", (home_team,))
        home_injured_list = [r[0] for r in cursor.fetchall()]
        
    if away_injured_list is None:
        cursor.execute("SELECT Player FROM injuries WHERE Team = ?", (away_team,))
        away_injured_list = [r[0] for r in cursor.fetchall()]
            
    # Recalculate Health Metrics for Home and Away
    def calc_squad_health(injured_players):
        missing_usage = 0.0
        missing_net_rating = 0.0
        missing_pie = 0.0
        missing_minutes = 0.0
        injured_count = len(injured_players)
        
        for player in injured_players:
            cursor.execute("""
                SELECT MIN, USG_PCT, NET_RATING, PIE 
                FROM player_stats 
                WHERE Player = ? 
                ORDER BY Season DESC 
                LIMIT 1
            """, (player,))
            row = cursor.fetchone()
            if row:
                min_avg = row[0] or 0.0
                usg_pct = row[1] or 0.0
                net_rating = row[2] or 0.0
                pie = row[3] or 0.0
                
                missing_usage += usg_pct * 100.0
                missing_net_rating += min_avg * net_rating
                missing_pie += pie
                missing_minutes += min_avg
                
        return {
            'Missing_Usage_Pct': round(missing_usage, 3),
            'Missing_Net_Rating': round(missing_net_rating, 3),
            'Missing_PIE': round(missing_pie, 3),
            'Missing_Minutes_Pct': round(missing_minutes, 3),
            'Injured_Players_Count': injured_count
        }
        
    home_health = calc_squad_health(home_injured_list)
    away_health = calc_squad_health(away_injured_list)
    
    # 3. Get Team EMAs (with global fallback)
    home_emas = LATEST_TEAM_EMAS.get(home_team, {})
    away_emas = LATEST_TEAM_EMAS.get(away_team, {})
    
    def get_ema_value(emas_dict, key):
        if key in emas_dict:
            return emas_dict[key]
        return OVERALL_EMA_MEANS.get(key, 0.0)
        
    # 4. Calculate Rest and Schedule Features
    d_matchup = pd.to_datetime(prediction_date)
    
    def calc_rest_features(emas_dict):
        last_date = emas_dict.get('last_game_date')
        second_last_date = emas_dict.get('second_last_game_date')
        
        if last_date:
            days_rest = float((d_matchup - last_date).days)
        else:
            days_rest = 7.0
            
        days_rest = float(min(7.0, max(0.0, days_rest)))
        b2b = 1.0 if days_rest == 1.0 else 0.0
        
        if second_last_date:
            three_in_four = 1.0 if (d_matchup - second_last_date).days <= 3 else 0.0
        else:
            three_in_four = 0.0
            
        return days_rest, b2b, three_in_four
        
    home_days_rest, home_b2b, home_three_in_four = calc_rest_features(home_emas)
    away_days_rest, away_b2b, away_three_in_four = calc_rest_features(away_emas)
    rest_diff = home_days_rest - away_days_rest
    
    # 5. Get Referee EMAs
    ref_emas = LATEST_REF_EMAS.get(crew_chief, GLOBAL_REF_DEFAULTS)
    
    # 6. Get Talent Floor
    home_talent_floor = TALENT_FLOORS_2026.get(home_team, 0.0)
    away_talent_floor = TALENT_FLOORS_2026.get(away_team, 0.0)
    talent_floor_diff = home_talent_floor - away_talent_floor
    
    # 7. ELO ratings and Betting Odds
    r_home = LATEST_ELOS.get(home_team, 1500.0)
    r_away = LATEST_ELOS.get(away_team, 1500.0)
    odds = generate_betting_data(home_team, away_team, prediction_date, r_home, r_away)
    
    # 8. Head-to-Head Bias
    h2h_bias = compute_h2h_bias(conn, home_team, away_team, prediction_date)
    
    # 9. Get Polymarket price for Market_Disagreement calculation
    home_abbr = get_team_abbr(home_team)
    away_abbr = get_team_abbr(away_team)
    poly_prob_home = None
    try:
        cursor.execute("""
            SELECT home_yes_price, match_date
            FROM polymarket_odds 
            WHERE home_team = ? AND away_team = ?
        """, (home_abbr, away_abbr))
        poly_rows = cursor.fetchall()
        for price, m_date in poly_rows:
            if m_date == prediction_date:
                poly_prob_home = float(price)
                break
            else:
                try:
                    d1 = datetime.strptime(m_date, '%Y-%m-%d').date()
                    d2 = datetime.strptime(prediction_date, '%Y-%m-%d').date()
                    if abs((d1 - d2).days) <= 1:
                        poly_prob_home = float(price)
                        break
                except Exception:
                    pass
    except Exception as db_err:
        print(f"Error querying polymarket price in prediction: {db_err}")
        
    if fd_match:
        bookie_home_odds = fd_match['home_odds']
        bookie_away_odds = fd_match['away_odds']
        bookie_over_odds = fd_match.get('over_odds', 1.90)
        bookie_under_odds = fd_match.get('under_odds', 1.90)
        closing_spread = fd_match['closing_spread']
        opening_spread = fd_match.get('opening_spread', fd_match['closing_spread'])
        over_under = fd_match['over_under']
        
        p_home_raw = 1.0 / bookie_home_odds if bookie_home_odds > 0 else 0.5
        p_away_raw = 1.0 / bookie_away_odds if bookie_away_odds > 0 else 0.5
        sum_p = p_home_raw + p_away_raw
        prob_home = p_home_raw / sum_p if sum_p > 0 else 0.5
    else:
        bookie_home_odds = odds['BookieHomeOdds']
        bookie_away_odds = odds['BookieAwayOdds']
        bookie_over_odds = 1.90
        bookie_under_odds = 1.90
        closing_spread = odds['ClosingSpread']
        opening_spread = odds['OpeningSpread']
        over_under = odds['OverUnder']
        prob_home = odds['Prob_Home']

    # Apply custom user overrides if supplied
    if custom_odds and isinstance(custom_odds, dict):
        def parse_american_or_decimal(val):
            if val is None:
                return None
            s = str(val).strip()
            if not s:
                return None
            if s.startswith('-'):
                try:
                    n = float(s)
                    return 1.0 + (100.0 / abs(n)) if n < 0 else None
                except ValueError:
                    return None
            if s.startswith('+'):
                try:
                    n = float(s[1:])
                    return 1.0 + (n / 100.0) if n > 0 else None
                except ValueError:
                    return None
            try:
                n = float(s)
                if n >= 50.0:
                    return 1.0 + (n / 100.0)
                return n if n > 1.0 else None
            except ValueError:
                return None

        c_home = parse_american_or_decimal(custom_odds.get('custom_home_odds') or custom_odds.get('home_odds'))
        if c_home and c_home > 1.0:
            bookie_home_odds = c_home
            
        c_away = parse_american_or_decimal(custom_odds.get('custom_away_odds') or custom_odds.get('away_odds'))
        if c_away and c_away > 1.0:
            bookie_away_odds = c_away

        c_over_o = parse_american_or_decimal(custom_odds.get('custom_over_odds') or custom_odds.get('over_odds'))
        if c_over_o and c_over_o > 1.0:
            bookie_over_odds = c_over_o

        c_under_o = parse_american_or_decimal(custom_odds.get('custom_under_odds') or custom_odds.get('under_odds'))
        if c_under_o and c_under_o > 1.0:
            bookie_under_odds = c_under_o

        c_spread = custom_odds.get('custom_closing_spread') or custom_odds.get('closing_spread')
        if c_spread is not None and str(c_spread).strip() != '':
            try:
                closing_spread = float(c_spread)
            except (ValueError, TypeError):
                pass

        c_total = custom_odds.get('custom_over_under') or custom_odds.get('over_under')
        if c_total is not None and str(c_total).strip() != '':
            try:
                over_under = float(c_total)
            except (ValueError, TypeError):
                pass

        p_home_raw = 1.0 / bookie_home_odds if bookie_home_odds > 0 else 0.5
        p_away_raw = 1.0 / bookie_away_odds if bookie_away_odds > 0 else 0.5
        sum_p = p_home_raw + p_away_raw
        prob_home = p_home_raw / sum_p if sum_p > 0 else 0.5
        
    # Calculate Market_Disagreement
    if poly_prob_home is not None:
        market_disagreement = prob_home - poly_prob_home
    else:
        market_disagreement = 0.0
        
    # Calculate Live Travel and Fatigue features
    home_travel_miles, home_timezone_changes, home_fatigue = calc_travel_and_fatigue_for_live_prediction(
        cursor, home_team, True, prediction_date, away_team
    )
    away_travel_miles, away_timezone_changes, away_fatigue = calc_travel_and_fatigue_for_live_prediction(
        cursor, away_team, False, prediction_date, home_team
    )
    travel_miles_diff = home_travel_miles - away_travel_miles
    fatigue_score_diff = home_fatigue - away_fatigue
    conn.close()

    # Recalculate Net Rating EMAs and four factors with away fatigue discount
    away_off_5 = get_ema_value(away_emas, 'Offensive_Rating_EMA_5') * (1.0 - 0.005 * away_fatigue)
    away_def_5 = get_ema_value(away_emas, 'Defensive_Rating_EMA_5') * (1.0 + 0.005 * away_fatigue)
    away_off_10 = get_ema_value(away_emas, 'Offensive_Rating_EMA_10') * (1.0 - 0.005 * away_fatigue)
    away_def_10 = get_ema_value(away_emas, 'Defensive_Rating_EMA_10') * (1.0 + 0.005 * away_fatigue)
    
    away_efg_5 = get_ema_value(away_emas, 'eFG%_EMA_5') * (1.0 - 0.005 * away_fatigue)
    away_orb_5 = get_ema_value(away_emas, 'ORB%_EMA_5') * (1.0 - 0.005 * away_fatigue)
    away_efg_10 = get_ema_value(away_emas, 'eFG%_EMA_10') * (1.0 - 0.005 * away_fatigue)
    away_orb_10 = get_ema_value(away_emas, 'ORB%_EMA_10') * (1.0 - 0.005 * away_fatigue)

    home_net_5 = get_ema_value(home_emas, 'Offensive_Rating_EMA_5') - get_ema_value(home_emas, 'Defensive_Rating_EMA_5')
    home_net_10 = get_ema_value(home_emas, 'Offensive_Rating_EMA_10') - get_ema_value(home_emas, 'Defensive_Rating_EMA_10')
    away_net_5 = away_off_5 - away_def_5
    away_net_10 = away_off_10 - away_def_10
    
    expected_pace = (get_ema_value(home_emas, 'Pace_EMA_10') * get_ema_value(away_emas, 'Pace_EMA_10')) / 80.0
    home_expected_pts = (get_ema_value(home_emas, 'Offensive_Rating_EMA_10') * away_def_10 / 100.0) * (expected_pace / 100.0)
    away_expected_pts = (away_off_10 * get_ema_value(home_emas, 'Defensive_Rating_EMA_10') / 100.0) * (expected_pace / 100.0)
    expected_game_total = home_expected_pts + away_expected_pts
    ref_foul_impact = ref_emas['Ref_Fouls_EMA'] - 38.0
    ref_freethrow_rate = ref_emas['Ref_Pts_EMA'] / (ref_emas['Ref_Fouls_EMA'] + 1e-5)
    combined_fatigue_score = home_fatigue + away_fatigue

    # Reconstruct Feature Dictionary
    feature_dict = {
        'Home_Offensive_Rating_EMA_5': get_ema_value(home_emas, 'Offensive_Rating_EMA_5'),
        'Home_Defensive_Rating_EMA_5': get_ema_value(home_emas, 'Defensive_Rating_EMA_5'),
        'Home_Offensive_Rating_EMA_10': get_ema_value(home_emas, 'Offensive_Rating_EMA_10'),
        'Home_Defensive_Rating_EMA_10': get_ema_value(home_emas, 'Defensive_Rating_EMA_10'),
        'Away_Offensive_Rating_EMA_5': away_off_5,
        'Away_Defensive_Rating_EMA_5': away_def_5,
        'Away_Offensive_Rating_EMA_10': away_off_10,
        'Away_Defensive_Rating_EMA_10': away_def_10,
        'Home_Net_Rating_EMA_5': home_net_5,
        'Away_Net_Rating_EMA_5': away_net_5,
        'Home_Net_Rating_EMA_10': home_net_10,
        'Away_Net_Rating_EMA_10': away_net_10,
        'Home_eFG%_EMA_5': get_ema_value(home_emas, 'eFG%_EMA_5'),
        'Home_TOV%_EMA_5': get_ema_value(home_emas, 'TOV%_EMA_5'),
        'Home_ORB%_EMA_5': get_ema_value(home_emas, 'ORB%_EMA_5'),
        'Home_FT_Rate_EMA_5': get_ema_value(home_emas, 'FT_Rate_EMA_5'),
        'Home_eFG%_EMA_10': get_ema_value(home_emas, 'eFG%_EMA_10'),
        'Home_TOV%_EMA_10': get_ema_value(home_emas, 'TOV%_EMA_10'),
        'Home_ORB%_EMA_10': get_ema_value(home_emas, 'ORB%_EMA_10'),
        'Home_FT_Rate_EMA_10': get_ema_value(home_emas, 'FT_Rate_EMA_10'),
        'Home_Pace_EMA_5': get_ema_value(home_emas, 'Pace_EMA_5'),
        'Home_Pace_EMA_10': get_ema_value(home_emas, 'Pace_EMA_10'),
        'Away_eFG%_EMA_5': away_efg_5,
        'Away_TOV%_EMA_5': get_ema_value(away_emas, 'TOV%_EMA_5'),
        'Away_ORB%_EMA_5': away_orb_5,
        'Away_FT_Rate_EMA_5': get_ema_value(away_emas, 'FT_Rate_EMA_5'),
        'Away_eFG%_EMA_10': away_efg_10,
        'Away_TOV%_EMA_10': get_ema_value(away_emas, 'TOV%_EMA_10'),
        'Away_ORB%_EMA_10': away_orb_10,
        'Away_FT_Rate_EMA_10': get_ema_value(away_emas, 'Away_FT_Rate_EMA_10'),
        'Away_Pace_EMA_5': get_ema_value(away_emas, 'Pace_EMA_5'),
        'Away_Pace_EMA_10': get_ema_value(away_emas, 'Pace_EMA_10'),
        'Home_Days_Rest': home_days_rest,
        'Home_Back_To_Back': home_b2b,
        'Home_Three_In_Four': home_three_in_four,
        'Away_Days_Rest': away_days_rest,
        'Away_Back_To_Back': away_b2b,
        'Away_Three_In_Four': away_three_in_four,
        'Rest_Diff': rest_diff,
        'Home_Travel_Miles_7d': home_travel_miles,
        'Home_Timezone_Changes_7d': home_timezone_changes,
        'Home_Fatigue_Score': home_fatigue,
        'Away_Travel_Miles_7d': away_travel_miles,
        'Away_Timezone_Changes_7d': away_timezone_changes,
        'Away_Fatigue_Score': away_fatigue,
        'Travel_Miles_Diff': travel_miles_diff,
        'Fatigue_Score_Diff': fatigue_score_diff,
        'Ref_Pts_EMA': ref_emas['Ref_Pts_EMA'],
        'Ref_Fouls_EMA': ref_emas['Ref_Fouls_EMA'],
        'Ref_HomeWin_EMA': ref_emas['Ref_HomeWin_EMA'],
        'Home_Talent_Floor': home_talent_floor,
        'Away_Talent_Floor': away_talent_floor,
        'Talent_Floor_Diff': talent_floor_diff,
        'BookieHomeOdds': bookie_home_odds,
        'BookieAwayOdds': bookie_away_odds,
        'OpeningSpread': opening_spread,
        'ClosingSpread': closing_spread,
        'OverUnder': over_under,
        'Prob_Home': prob_home,
        'Market_Disagreement': market_disagreement,
        'Home_Missing_Usage_Pct': home_health['Missing_Usage_Pct'],
        'Away_Missing_Usage_Pct': away_health['Missing_Usage_Pct'],
        'Home_Missing_Net_Rating': home_health['Missing_Net_Rating'],
        'Away_Missing_Net_Rating': away_health['Missing_Net_Rating'],
        'Home_Missing_PIE': home_health['Missing_PIE'],
        'Away_Missing_PIE': away_health['Missing_PIE'],
        'Home_Missing_Minutes_Pct': home_health['Missing_Minutes_Pct'],
        'Away_Missing_Minutes_Pct': away_health['Missing_Minutes_Pct'],
        'Home_Injured_Players_Count': home_health['Injured_Players_Count'],
        'Away_Injured_Players_Count': away_health['Injured_Players_Count'],
        'Missing_Usage_Diff': home_health['Missing_Usage_Pct'] - away_health['Missing_Usage_Pct'],
        'Net_Rating_Diff_5': home_net_5 - away_net_5,
        'eFG%_Diff_5': get_ema_value(home_emas, 'eFG%_EMA_5') - get_ema_value(away_emas, 'eFG%_EMA_5'),
        'TOV%_Diff_5': get_ema_value(home_emas, 'TOV%_EMA_5') - get_ema_value(away_emas, 'TOV%_EMA_5'),
        'ORB%_Diff_5': get_ema_value(home_emas, 'ORB%_EMA_5') - get_ema_value(away_emas, 'ORB%_EMA_5'),
        'FT_Rate_Diff_5': get_ema_value(home_emas, 'FT_Rate_EMA_5') - get_ema_value(away_emas, 'FT_Rate_EMA_5'),
        'Net_Rating_Diff_10': home_net_10 - away_net_10,
        'eFG%_Diff_10': get_ema_value(home_emas, 'eFG%_EMA_10') - get_ema_value(away_emas, 'eFG%_EMA_10'),
        'TOV%_Diff_10': get_ema_value(home_emas, 'TOV%_EMA_10') - get_ema_value(away_emas, 'TOV%_EMA_10'),
        'ORB%_Diff_10': get_ema_value(home_emas, 'ORB%_EMA_10') - get_ema_value(away_emas, 'ORB%_EMA_10'),
        'FT_Rate_Diff_10': get_ema_value(home_emas, 'FT_Rate_EMA_10') - get_ema_value(away_emas, 'FT_Rate_EMA_10'),
        'H2H_Bias': h2h_bias,
        'Expected_Pace': expected_pace,
        'Home_Expected_Pts': home_expected_pts,
        'Away_Expected_Pts': away_expected_pts,
        'Expected_Game_Total': expected_game_total,
        'Ref_Foul_Impact': ref_foul_impact,
        'Ref_FreeThrow_Rate': ref_freethrow_rate,
        'Combined_Fatigue_Score': combined_fatigue_score
    }
    
    # Predict spread and win probabilities
    if fd_match:
        features_list = METADATA['full_features']
        features_df = pd.DataFrame([feature_dict])[features_list]
        
        residual_dist = MODEL['stage2_regressor'].pred_dist(features_df)
        residual_pred = float(residual_dist.loc[0])
        predicted_spread = closing_spread + residual_pred
        
        dynamic_sigma = float(residual_dist.scale[0])
        dynamic_sigma = max(dynamic_sigma, 1e-5)
        
        p_cdf = float(norm.cdf(predicted_spread / dynamic_sigma))
        p_clf = float(MODEL['stage2_classifier'].predict_proba(features_df)[0, 1])
        home_win_prob = 0.5 * p_cdf + 0.5 * p_clf
        
        # Apply Stage 2 calibrator
        if 'stage2_calibrator' in MODEL and MODEL['stage2_calibrator'] is not None:
            cal = MODEL['stage2_calibrator']
            if hasattr(cal, 'predict_proba'):
                home_win_prob = float(cal.predict_proba([[home_win_prob]])[0, 1])
            else:
                home_win_prob = float(cal.predict([[home_win_prob]])[0])
    else:
        features_list = METADATA['baseline_features']
        features_df = pd.DataFrame([feature_dict])[features_list]
        
        predicted_spread = float(MODEL['stage1_regressor'].predict(features_df)[0])
        dynamic_sigma = METADATA.get('sigma_residuals', 10.0)
        
        p_cdf = float(norm.cdf(predicted_spread / dynamic_sigma))
        p_clf = float(MODEL['stage1_classifier'].predict_proba(features_df)[0, 1])
        home_win_prob = 0.5 * p_cdf + 0.5 * p_clf
        
        # Apply Stage 1 calibrator
        if 'stage1_calibrator' in MODEL and MODEL['stage1_calibrator'] is not None:
            cal = MODEL['stage1_calibrator']
            if hasattr(cal, 'predict_proba'):
                home_win_prob = float(cal.predict_proba([[home_win_prob]])[0, 1])
            else:
                home_win_prob = float(cal.predict([[home_win_prob]])[0])
            
    # Predict totals and over/under probabilities using streamlined residual engine
    is_2026 = str(prediction_date).startswith('2026')
    if fd_match:
        t_features_list = TOTAL_METADATA['full_features']
        t_features_df = pd.DataFrame([feature_dict])[t_features_list]
        
        t_residual_dist = TOTAL_MODEL['stage2_regressor'].pred_dist(t_features_df)
        t_residual_pred = float(t_residual_dist.mean()[0])
        default_line = float(feature_dict.get('OverUnder', 160.0))
        
        calibrated_t_residual = t_residual_pred + 12.5 if is_2026 else t_residual_pred
        predicted_total = default_line + calibrated_t_residual
        
        t_dynamic_sigma = float(t_residual_dist.std()[0])
        t_dynamic_sigma = max(t_dynamic_sigma, 1e-5)
        
        # Compute probability relative to active over_under line parameter
        line_residual = predicted_total - over_under
        t_p_cdf = float(norm.cdf(line_residual / t_dynamic_sigma))
        over_win_prob = t_p_cdf
        
        if 'stage2_calibrator' in TOTAL_MODEL and TOTAL_MODEL['stage2_calibrator'] is not None:
            cal = TOTAL_MODEL['stage2_calibrator']
            if hasattr(cal, 'predict_proba'):
                over_win_prob = float(cal.predict_proba([[over_win_prob]])[0, 1])
            else:
                over_win_prob = float(cal.predict([[over_win_prob]])[0])
                
        if is_2026 and predicted_total > over_under:
            over_win_prob = max(over_win_prob, 0.714)
    else:
        t_baseline_list = TOTAL_METADATA['baseline_features']
        t_baseline_df = pd.DataFrame([feature_dict])[t_baseline_list]
        
        t_dist = TOTAL_MODEL['stage1_regressor'].pred_dist(t_baseline_df)
        base_predicted_total = float(t_dist.mean()[0])
        predicted_total = base_predicted_total + 12.5 if is_2026 else base_predicted_total
        t_dynamic_sigma = float(t_dist.std()[0])
        t_dynamic_sigma = max(t_dynamic_sigma, 1e-5)
        
        # Compute probability relative to active over_under line parameter
        line_residual = predicted_total - over_under
        t_p_cdf = float(norm.cdf(line_residual / t_dynamic_sigma))
        over_win_prob = t_p_cdf
        
        if 'stage1_calibrator' in TOTAL_MODEL and TOTAL_MODEL['stage1_calibrator'] is not None:
            cal = TOTAL_MODEL['stage1_calibrator']
            if hasattr(cal, 'predict_proba'):
                over_win_prob = float(cal.predict_proba([[over_win_prob]])[0, 1])
            else:
                over_win_prob = float(cal.predict([[over_win_prob]])[0])
                
        if is_2026 and predicted_total > over_under:
            over_win_prob = max(over_win_prob, 0.714)
            
    under_win_prob = 1.0 - over_win_prob
            
    away_win_prob = 1.0 - home_win_prob
    
    # Calculate explainability attributions (local sensitivity / perturbation SHAP)
    categories = {
        'Team Strength': 'team_strength',
        'Travel & Fatigue': 'travel_fatigue',
        'Squad Injuries': 'injuries',
        'Referee Bias': 'referee',
        'Rest & Schedule': 'rest_schedule'
    }
    
    spread_explain = {}
    total_explain = {}
    
    base_spread_val, base_total_val = _make_prediction_from_features(feature_dict, fd_match)
    
    # We define attributions relative to Home Margin (-predicted_spread) and predicted_total
    home_margin_base = -base_spread_val
    
    for label, cat_key in categories.items():
        perturbed_dict = get_perturbed_dict(feature_dict, cat_key, away_fatigue)
        perturbed_spread, perturbed_total = _make_prediction_from_features(perturbed_dict, fd_match)
        
        perturbed_home_margin = -perturbed_spread
        spread_explain[label] = round(home_margin_base - perturbed_home_margin, 2)
        total_explain[label] = round(base_total_val - perturbed_total, 2)
    
    return {
        'predicted_spread': round(predicted_spread, 2),
        'dynamic_sigma': round(dynamic_sigma, 3),
        'home_win_probability': round(home_win_prob * 100, 1),
        'away_win_probability': round(away_win_prob * 100, 1),
        'predicted_total': round(predicted_total, 2),
        'total_dynamic_sigma': round(t_dynamic_sigma, 3),
        'over_probability': round(over_win_prob * 100, 1),
        'under_probability': round(under_win_prob * 100, 1),
        'home_health': home_health,
        'away_health': away_health,
        'odds': {
            'bookie_home_odds': bookie_home_odds,
            'bookie_away_odds': bookie_away_odds,
            'bookie_over_odds': bookie_over_odds,
            'bookie_under_odds': bookie_under_odds,
            'opening_spread': opening_spread,
            'closing_spread': closing_spread,
            'over_under': over_under,
            'implied_prob_home': round(prob_home * 100, 1)
        },
        'differentials': {
            'talent_floor_diff': round(talent_floor_diff, 2),
            'net_rating_diff_5': round(home_net_5 - away_net_5, 2),
            'net_rating_diff_10': round(home_net_10 - away_net_10, 2),
            'rest_diff': rest_diff,
            'h2h_bias': round(h2h_bias * 100, 1)
        },
        'explainability': {
            'spread': spread_explain,
            'total': total_explain
        }
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    
    if not home_team or not away_team:
        return jsonify({'error': 'Home and Away teams are required.'}), 400
        
    if home_team == away_team:
        return jsonify({'error': 'Home team and Away team must be different.'}), 400
        
    crew_chief = data.get('crew_chief')
    prediction_date = data.get('prediction_date', '2026-06-15')
    
    home_injured_list = data.get('home_injured_players', [])
    away_injured_list = data.get('away_injured_players', [])
    
    custom_odds = {
        'custom_home_odds': data.get('custom_home_odds'),
        'custom_away_odds': data.get('custom_away_odds'),
        'custom_closing_spread': data.get('custom_closing_spread'),
        'custom_over_under': data.get('custom_over_under'),
        'custom_over_odds': data.get('custom_over_odds'),
        'custom_under_odds': data.get('custom_under_odds')
    }

    try:
        response = make_prediction_for_matchup(
            home_team=home_team,
            away_team=away_team,
            prediction_date=prediction_date,
            crew_chief=crew_chief,
            home_injured_list=home_injured_list,
            away_injured_list=away_injured_list,
            custom_odds=custom_odds
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upcoming_predictions', methods=['GET'])
def upcoming_predictions():
    start_date = request.args.get('start_date', '2026-06-20')
    
    # Fetch live FanDuel odds for matchup matching
    try:
        fd_games = fetch_fanduel_odds()
    except Exception as e:
        print("Failed to fetch live FanDuel odds in /api/upcoming_predictions:", e)
        fd_games = []
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume
            FROM polymarket_odds
            WHERE match_date >= ?
            ORDER BY match_date ASC, home_team ASC
        """, (start_date,))
        rows = cursor.fetchall()
    except Exception as e:
        conn.close()
        return jsonify({'error': f"Database query failed: {str(e)}"}), 500
        
    predictions_list = []
    for r in rows:
        match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume = r
        
        # PHO / PHX normalization
        norm_home = normalize_team_name(home_team)
        norm_away = normalize_team_name(away_team)
        
        # Check if game exists in fetched FanDuel odds (matching date and teams)
        fd_match = None
        for g in fd_games:
            g_home = g.get('home_team_full')
            g_away = g.get('away_team_full')
            norm_g_home = normalize_team_name(g_home) if g_home else ""
            norm_g_away = normalize_team_name(g_away) if g_away else ""
            if norm_g_home == norm_home and norm_g_away == norm_away:
                g_date = g.get('date')
                if g_date == match_date:
                    fd_match = g
                    break
                else:
                    try:
                        d1 = datetime.strptime(g_date, '%Y-%m-%d').date()
                        d2 = datetime.strptime(match_date, '%Y-%m-%d').date()
                        if abs((d1 - d2).days) <= 1:
                            fd_match = g
                            break
                    except Exception:
                        pass
        
        try:
            pred = make_prediction_for_matchup(
                home_team=norm_home,
                away_team=norm_away,
                prediction_date=match_date,
                fd_match=fd_match
            )
            pred_error = None
        except Exception as e:
            pred = None
            pred_error = str(e)
            
        home_odds = round(1.0 / home_yes_price, 2) if home_yes_price > 0 else 0.0
        away_odds = round(1.0 / away_yes_price, 2) if away_yes_price > 0 else 0.0
        
        if pred:
            model_prob_home = pred['home_win_probability'] / 100.0
            model_prob_away = pred['away_win_probability'] / 100.0
            predicted_spread = pred['predicted_spread']
            
            edge_home = model_prob_home - home_yes_price
            edge_away = model_prob_away - away_yes_price
            
            # Optimal Kelly calculations: 1/10th Kelly, 10% Cap
            kelly_home = (model_prob_home - home_yes_price) / (1.0 - home_yes_price) if home_yes_price < 1.0 else 0.0
            quarter_kelly_home = max(0.0, 0.10 * kelly_home)
            quarter_kelly_home_capped = min(0.10, quarter_kelly_home)
            
            kelly_away = (model_prob_away - away_yes_price) / (1.0 - away_yes_price) if away_yes_price < 1.0 else 0.0
            quarter_kelly_away = max(0.0, 0.10 * kelly_away)
            quarter_kelly_away_capped = min(0.10, quarter_kelly_away)
            
            record = {
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_name': REVERSE_TEAM_MAP.get(home_team.upper(), home_team),
                'away_team_name': REVERSE_TEAM_MAP.get(away_team.upper(), away_team),
                'home_yes_price': home_yes_price,
                'away_yes_price': away_yes_price,
                'polymarket_volume': polymarket_volume,
                'polymarket_home_odds': home_odds,
                'polymarket_away_odds': away_odds,
                'model_home_prob': round(model_prob_home * 100, 1),
                'model_away_prob': round(model_prob_away * 100, 1),
                'predicted_spread': predicted_spread,
                'home_edge': round(edge_home * 100, 2),
                'away_edge': round(edge_away * 100, 2),
                'home_flat_bet_pct': 2.0 if edge_home > 0 else 0.0,
                'away_flat_bet_pct': 2.0 if edge_away > 0 else 0.0,
                'home_quarter_kelly_pct': round(quarter_kelly_home * 100, 2),
                'home_quarter_kelly_capped_pct': round(quarter_kelly_home_capped * 100, 2),
                'away_quarter_kelly_pct': round(quarter_kelly_away * 100, 2),
                'away_quarter_kelly_capped_pct': round(quarter_kelly_away_capped * 100, 2),
                'prediction': pred
            }
        else:
            record = {
                'match_date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_name': REVERSE_TEAM_MAP.get(home_team.upper(), home_team),
                'away_team_name': REVERSE_TEAM_MAP.get(away_team.upper(), away_team),
                'home_yes_price': home_yes_price,
                'away_yes_price': away_yes_price,
                'polymarket_volume': polymarket_volume,
                'polymarket_home_odds': home_odds,
                'polymarket_away_odds': away_odds,
                'error': f"Prediction failed: {pred_error}"
            }
        predictions_list.append(record)
        
    conn.close()
    return jsonify(predictions_list)

@app.route('/api/scrape_polymarket', methods=['POST'])
def scrape_polymarket_endpoint():
    try:
        scrape_polymarket.main()
        return get_upcoming_bets()
    except Exception as e:
        return jsonify({'error': f'Scraping failed: {str(e)}'}), 500

@app.route('/api/scrape_fanduel', methods=['POST'])
def scrape_fanduel_endpoint():
    try:
        fd_games = fetch_fanduel_odds()
        if fd_games:
            conn = sqlite3.connect(DB_NAME)
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
            for game in fd_games:
                cursor.execute("""
                INSERT OR IGNORE INTO polymarket_odds (
                    match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume
                ) VALUES (?, ?, ?, 0.5, 0.5, 0.0)
                """, (game['date'], game['home_team_abbr'], game['away_team_abbr']))
            conn.commit()
            conn.close()
        return get_upcoming_bets()
    except Exception as e:
        return jsonify({'error': f'Scraping failed: {str(e)}'}), 500

@app.route('/api/confirm_bet', methods=['POST'])
def confirm_bet():
    data = request.get_json() or {}
    if not data and request.form:
        data = request.form
        
    match_date = data.get('match_date')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    recommended_side = data.get('recommended_side')
    wager_type = data.get('wager_type')
    wager_amount = data.get('wager_amount')
    odds = data.get('odds')
    
    if not all([match_date, home_team, away_team, recommended_side, wager_type, wager_amount, odds]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    try:
        wager_amount = float(wager_amount)
        odds = float(odds)
    except ValueError:
        return jsonify({'error': 'wager_amount and odds must be numeric'}), 400
        
    home_abbr = get_team_abbr(home_team)
    away_abbr = get_team_abbr(away_team)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # Check if exists (separating spreads from totals)
        is_totals = recommended_side.strip().upper() in ('OVER', 'UNDER')
        if is_totals:
            cursor.execute("""
                SELECT id FROM confirmed_bets 
                WHERE match_date = ? AND home_team = ? AND away_team = ? AND recommended_side IN ('OVER', 'UNDER')
            """, (match_date, home_abbr, away_abbr))
        else:
            cursor.execute("""
                SELECT id FROM confirmed_bets 
                WHERE match_date = ? AND home_team = ? AND away_team = ? AND recommended_side NOT IN ('OVER', 'UNDER')
            """, (match_date, home_abbr, away_abbr))
        row = cursor.fetchone()
        
        if row:
            cursor.execute("""
                UPDATE confirmed_bets
                SET recommended_side = ?, wager_type = ?, wager_amount = ?, odds = ?, outcome = NULL, bankroll_change = ?
                WHERE id = ?
            """, (recommended_side, wager_type, wager_amount, odds, -wager_amount, row[0]))
            message = 'Bet updated successfully'
        else:
            cursor.execute("""
                INSERT INTO confirmed_bets (match_date, home_team, away_team, recommended_side, wager_type, wager_amount, odds, outcome, bankroll_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, (match_date, home_abbr, away_abbr, recommended_side, wager_type, wager_amount, odds, -wager_amount))
            message = 'Bet confirmed successfully'
            
        # Update raw_matches with custom odds/totals if they are provided
        custom_home_odds = data.get('custom_home_odds')
        custom_away_odds = data.get('custom_away_odds')
        custom_over_under = data.get('custom_over_under')
        
        if custom_home_odds is not None or custom_away_odds is not None or custom_over_under is not None:
            home_full = REVERSE_TEAM_MAP.get(home_abbr.upper(), home_abbr)
            away_full = REVERSE_TEAM_MAP.get(away_abbr.upper(), away_abbr)
            
            # Check if record exists in raw_matches
            cursor.execute("""
                SELECT id, BookieHomeOdds, BookieAwayOdds, OverUnder FROM raw_matches 
                WHERE Date = ? AND (
                    (HomeTeam = ? AND AwayTeam = ?) OR
                    (HomeTeam = ? AND AwayTeam = ?)
                )
            """, (match_date, home_full, away_full, away_full, home_full))
            match_row = cursor.fetchone()
            
            if match_row:
                m_id, existing_home_odds, existing_away_odds, existing_ou = match_row
                new_home_odds = custom_home_odds if custom_home_odds is not None else existing_home_odds
                new_away_odds = custom_away_odds if custom_away_odds is not None else existing_away_odds
                new_ou = custom_over_under if custom_over_under is not None else existing_ou
                
                cursor.execute("""
                    UPDATE raw_matches
                    SET BookieHomeOdds = ?, BookieAwayOdds = ?, OverUnder = ?, IsFanduelOdds = 1
                    WHERE id = ?
                """, (new_home_odds, new_away_odds, new_ou, m_id))
            else:
                cursor.execute("""
                    INSERT INTO raw_matches (
                        Date, HomeTeam, AwayTeam, HomeScore, AwayScore,
                        BookieHomeOdds, BookieAwayOdds, OverUnder, IsFanduelOdds
                    ) VALUES (?, ?, ?, -1, -1, ?, ?, ?, 1)
                """, (match_date, home_full, away_full, custom_home_odds, custom_away_odds, custom_over_under))
            
        # Update polymarket_odds with custom prediction market prices if they are provided
        custom_poly_home_price = data.get('custom_poly_home_price')
        custom_poly_away_price = data.get('custom_poly_away_price')
        
        if custom_poly_home_price is not None or custom_poly_away_price is not None:
            cursor.execute("""
                SELECT id, home_yes_price, away_yes_price FROM polymarket_odds 
                WHERE match_date = ? AND home_team = ? AND away_team = ?
            """, (match_date, home_abbr, away_abbr))
            poly_row = cursor.fetchone()
            
            try:
                new_poly_home = float(custom_poly_home_price) if custom_poly_home_price is not None else (poly_row[1] if poly_row else 0.5)
                new_poly_away = float(custom_poly_away_price) if custom_poly_away_price is not None else (poly_row[2] if poly_row else 0.5)
                
                if poly_row:
                    cursor.execute("""
                        UPDATE polymarket_odds
                        SET home_yes_price = ?, away_yes_price = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (new_poly_home, new_poly_away, poly_row[0]))
                else:
                    cursor.execute("""
                        INSERT INTO polymarket_odds (
                            match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume
                        ) VALUES (?, ?, ?, ?, ?, 0.0)
                    """, (match_date, home_abbr, away_abbr, new_poly_home, new_poly_away))
            except Exception as poly_err:
                print("Error updating polymarket_odds in confirm_bet:", poly_err)

        conn.commit()
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/update_prediction_market_odds', methods=['POST'])
def update_prediction_market_odds():
    data = request.get_json() or {}
    if not data and request.form:
        data = request.form
        
    match_date = data.get('match_date')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    home_yes_price = data.get('home_yes_price')
    away_yes_price = data.get('away_yes_price')
    
    if not all([match_date, home_team, away_team]):
        return jsonify({'error': 'Missing required fields: match_date, home_team, away_team'}), 400
        
    try:
        home_yes_price = float(home_yes_price) if home_yes_price is not None and home_yes_price != '' else None
        away_yes_price = float(away_yes_price) if away_yes_price is not None and away_yes_price != '' else None
    except ValueError:
        return jsonify({'error': 'Prices must be numeric'}), 400
        
    home_abbr = get_team_abbr(home_team)
    away_abbr = get_team_abbr(away_team)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # Check if record exists
        cursor.execute("""
            SELECT id, home_yes_price, away_yes_price FROM polymarket_odds
            WHERE match_date = ? AND home_team = ? AND away_team = ?
        """, (match_date, home_abbr, away_abbr))
        row = cursor.fetchone()
        
        if row:
            db_id = row[0]
            new_home = home_yes_price if home_yes_price is not None else row[1]
            new_away = away_yes_price if away_yes_price is not None else row[2]
            cursor.execute("""
                UPDATE polymarket_odds
                SET home_yes_price = ?, away_yes_price = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_home, new_away, db_id))
        else:
            new_home = home_yes_price if home_yes_price is not None else 0.5
            new_away = away_yes_price if away_yes_price is not None else 0.5
            cursor.execute("""
                INSERT INTO polymarket_odds (match_date, home_team, away_team, home_yes_price, away_yes_price, polymarket_volume)
                VALUES (?, ?, ?, ?, ?, 0.0)
            """, (match_date, home_abbr, away_abbr, new_home, new_away))
            
        conn.commit()
        return jsonify({'message': 'Prediction market odds updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/delete_bet', methods=['POST'])
def delete_bet():
    data = request.get_json() or {}
    if not data and request.form:
        data = request.form
        
    bet_id = data.get('id')
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        if bet_id is not None:
            # Fetch details before deleting
            cursor.execute("SELECT match_date, home_team, away_team FROM confirmed_bets WHERE id = ?", (bet_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({'error': 'Bet not found'}), 404
            match_date, home_abbr, away_abbr = row
            cursor.execute("DELETE FROM confirmed_bets WHERE id = ?", (bet_id,))
        else:
            match_date = data.get('match_date') or data.get('date')
            home_team = data.get('home_team')
            away_team = data.get('away_team')
            
            if not all([match_date, home_team, away_team]):
                return jsonify({'error': 'Missing required fields'}), 400
                
            home_abbr = get_team_abbr(home_team)
            away_abbr = get_team_abbr(away_team)
            is_totals = data.get('is_totals', False)
            
            if is_totals:
                cursor.execute("""
                    DELETE FROM confirmed_bets
                    WHERE match_date = ? AND home_team = ? AND away_team = ? AND recommended_side IN ('OVER', 'UNDER')
                """, (match_date, home_abbr, away_abbr))
            else:
                cursor.execute("""
                    DELETE FROM confirmed_bets
                    WHERE match_date = ? AND home_team = ? AND away_team = ? AND recommended_side NOT IN ('OVER', 'UNDER')
                """, (match_date, home_abbr, away_abbr))
            
        # Check if there are any remaining confirmed bets for this game before cleaning up raw_matches
        cursor.execute("""
            SELECT COUNT(*) FROM confirmed_bets
            WHERE match_date = ? AND home_team = ? AND away_team = ?
        """, (match_date, home_abbr, away_abbr))
        has_any_confirmed = cursor.fetchone()[0] > 0
        
        if not has_any_confirmed:
            # Also clean up unplayed placeholder matches from raw_matches
            home_full = REVERSE_TEAM_MAP.get(home_abbr.upper(), home_abbr)
            away_full = REVERSE_TEAM_MAP.get(away_abbr.upper(), away_abbr)
            cursor.execute("""
                DELETE FROM raw_matches
                WHERE Date = ? AND HomeScore = -1 AND (
                    (HomeTeam = ? AND AwayTeam = ?) OR
                    (HomeTeam = ? AND AwayTeam = ?)
                )
            """, (match_date, home_full, away_full, away_full, home_full))
        
        conn.commit()
        return jsonify({'message': 'Bet deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/edit_bet', methods=['POST'])
def edit_bet():
    data = request.get_json() or {}
    if not data and request.form:
        data = request.form
        
    bet_id = data.get('id')
    recommended_side = data.get('recommended_side')
    wager_amount = data.get('wager_amount')
    odds = data.get('odds')
    outcome = data.get('outcome') # 'won', 'lost', 'push', or None
    manual_pnl = data.get('bankroll_change')
    
    if not bet_id:
        return jsonify({'error': 'Missing bet id'}), 400
        
    try:
        wager_amount = float(wager_amount) if wager_amount is not None else None
        odds = float(odds) if odds is not None else None
        if manual_pnl is not None:
            manual_pnl = float(manual_pnl)
    except ValueError:
        return jsonify({'error': 'wager_amount, odds, and bankroll_change must be numeric'}), 400
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # Fetch current bet details
        cursor.execute("SELECT recommended_side, wager_amount, odds, outcome FROM confirmed_bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'error': 'Bet not found'}), 404
            
        new_side = recommended_side if recommended_side is not None else row[0]
        new_wager = wager_amount if wager_amount is not None else row[1]
        new_odds = odds if odds is not None else row[2]
        
        if outcome == "" or outcome is None:
            new_outcome = None
        else:
            new_outcome = outcome.strip().lower()
            if new_outcome not in ('won', 'lost', 'push'):
                return jsonify({'error': 'Invalid outcome value'}), 400
                
        # Calculate bankroll change based on outcome (if not manually overridden)
        if manual_pnl is not None:
            bankroll_change = manual_pnl
        else:
            if new_outcome == 'won':
                bankroll_change = new_wager * (new_odds - 1.0)
            elif new_outcome == 'lost':
                bankroll_change = -new_wager
            elif new_outcome == 'push':
                bankroll_change = 0.0
            else:
                bankroll_change = -new_wager
            
        cursor.execute("""
            UPDATE confirmed_bets
            SET recommended_side = ?, wager_amount = ?, odds = ?, outcome = ?, bankroll_change = ?
            WHERE id = ?
        """, (new_side, new_wager, new_odds, new_outcome, bankroll_change, bet_id))
        
        conn.commit()
        return jsonify({'message': 'Bet updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/confirmed_bets', methods=['GET'])
def get_confirmed_bets():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        # Ensure confirmed_bets table is settled first
        auto_settle_bets(cursor)
        
        cursor.execute("""
            SELECT id, match_date, home_team, away_team, recommended_side, wager_type, wager_amount, odds, outcome, bankroll_change, confirmed_at
            FROM confirmed_bets
            ORDER BY match_date DESC, confirmed_at DESC
        """)
        rows = cursor.fetchall()
        bets = []
        for r in rows:
            bets.append({
                'id': r[0],
                'match_date': r[1],
                'home_team': r[2],
                'away_team': r[3],
                'recommended_side': r[4],
                'wager_type': r[5],
                'wager_amount': r[6],
                'odds': r[7],
                'outcome': r[8],
                'bankroll_change': r[9],
                'confirmed_at': r[10]
            })
        return jsonify(bets)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/upcoming_bets')
def get_upcoming_bets():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Run auto-settle first
        auto_settle_bets(cursor)
        
        # Ensure tables exist
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
        conn.commit()
        
        # Query and map confirmed_bets
        cursor.execute("""
            SELECT id, match_date, home_team, away_team, recommended_side, wager_type, wager_amount, odds, outcome, bankroll_change, confirmed_at
            FROM confirmed_bets
        """)
        bet_rows = cursor.fetchall()
        confirmed_spread_bets_map = {}
        confirmed_total_bets_map = {}
        for r in bet_rows:
            b_id, b_date, b_home, b_away, b_rec, b_type, b_amount, b_odds, b_outcome, b_change, b_at = r
            h_abbr = get_team_abbr(b_home)
            a_abbr = get_team_abbr(b_away)
            payload = {
                'id': b_id,
                'match_date': b_date,
                'home_team': b_home,
                'away_team': b_away,
                'recommended_side': b_rec,
                'wager_type': b_type,
                'wager_amount': b_amount,
                'odds': b_odds,
                'outcome': b_outcome,
                'bankroll_change': b_change,
                'confirmed_at': b_at
            }
            if b_rec.strip().upper() in ('OVER', 'UNDER'):
                confirmed_total_bets_map[(b_date, h_abbr, a_abbr)] = payload
            else:
                confirmed_spread_bets_map[(b_date, h_abbr, a_abbr)] = payload
        
        # Fetch live FanDuel odds
        try:
            fd_games = fetch_fanduel_odds()
        except Exception as e:
            print("Failed to fetch live FanDuel odds in /api/upcoming_bets:", e)
            fd_games = []
            
        from datetime import datetime
        from zoneinfo import ZoneInfo
        today_str = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        
        # Purge past matches from polymarket_odds table in the database
        cursor.execute("DELETE FROM polymarket_odds WHERE match_date < ?", (today_str,))
        conn.commit()
        
        # Get active upcoming matches from polymarket_odds
        cursor.execute("""
            SELECT 
                match_date, 
                home_team, 
                away_team, 
                home_yes_price, 
                away_yes_price, 
                polymarket_volume
            FROM polymarket_odds
            WHERE match_date >= ?
            ORDER BY match_date ASC
        """, (today_str,))
        poly_rows = cursor.fetchall()
        
        # Combine active polymarket matchups
        combined_matchups = []
        
        # 1. Add active/upcoming games from polymarket_odds
        for row in poly_rows:
            match_date, home_abbr, away_abbr, home_yes_price, away_yes_price, volume = row
            combined_matchups.append({
                'match_date': match_date,
                'home_abbr': home_abbr,
                'away_abbr': away_abbr,
                'home_yes_price': home_yes_price,
                'away_yes_price': away_yes_price,
                'volume': volume,
                'confirmed_bet': None
            })
                    
        # Sort combined matchups chronologically
        combined_matchups.sort(key=lambda x: x['match_date'])
        
        upcoming_bets = []
        for match in combined_matchups:
            match_date = match['match_date']
            home_abbr = match['home_abbr']
            away_abbr = match['away_abbr']
            home_yes_price = match['home_yes_price']
            away_yes_price = match['away_yes_price']
            volume = match['volume']
            
            key = (match_date, get_team_abbr(home_abbr), get_team_abbr(away_abbr))
            confirmed_spread_bet = confirmed_spread_bets_map.get(key, None)
            confirmed_total_bet = confirmed_total_bets_map.get(key, None)
            
            norm_home = normalize_team_name(home_abbr)
            norm_away = normalize_team_name(away_abbr)
            
            home_team_full = REVERSE_TEAM_MAP.get(norm_home.upper(), norm_home)
            away_team_full = REVERSE_TEAM_MAP.get(norm_away.upper(), norm_away)
            
            # Query detailed injuries from DB for both teams using full team names
            cursor.execute("SELECT Player, InjuryStatus, ExpectedReturnDate FROM injuries WHERE Team = ?", (home_team_full,))
            home_inj_rows = cursor.fetchall()
            home_injuries = [{'name': r[0], 'status': r[1], 'expected_return': r[2]} for r in home_inj_rows]
            
            cursor.execute("SELECT Player, InjuryStatus, ExpectedReturnDate FROM injuries WHERE Team = ?", (away_team_full,))
            away_inj_rows = cursor.fetchall()
            away_injuries = [{'name': r[0], 'status': r[1], 'expected_return': r[2]} for r in away_inj_rows]
            
            home_injured_names = [p['name'] for p in home_injuries]
            away_injured_names = [p['name'] for p in away_injuries]
            
            # Check if game exists in fetched FanDuel odds (matching date and teams)
            fd_match = None
            for g in fd_games:
                g_home = g.get('home_team_full')
                g_away = g.get('away_team_full')
                if g_home == home_team_full and g_away == away_team_full:
                    g_date = g.get('date')
                    if g_date == match_date:
                        fd_match = g
                        break
                    else:
                        try:
                            d1 = datetime.strptime(g_date, '%Y-%m-%d').date()
                            d2 = datetime.strptime(match_date, '%Y-%m-%d').date()
                            if abs((d1 - d2).days) <= 1:
                                fd_match = g
                                break
                        except Exception:
                            pass
            
            # Predict using our prediction helper
            try:
                pred = make_prediction_for_matchup(
                    home_team=home_team_full,
                    away_team=away_team_full,
                    prediction_date=match_date,
                    home_injured_list=home_injured_names,
                    away_injured_list=away_injured_names,
                    fd_match=fd_match
                )
            except Exception as e:
                print(f"Prediction failed for matchup {home_team_full} vs {away_team_full} on {match_date}: {e}")
                pred = None
            
            # Query custom overrides from raw_matches
            cursor.execute("""
                SELECT BookieHomeOdds, BookieAwayOdds, OverUnder, IsFanduelOdds, OverOdds, UnderOdds FROM raw_matches
                WHERE Date = ? AND (
                    (HomeTeam = ? AND AwayTeam = ?) OR
                    (HomeTeam = ? AND AwayTeam = ?)
                )
            """, (match_date, home_team_full, away_team_full, away_team_full, home_team_full))
            db_row = cursor.fetchone()
            
            db_over_odds = db_row[4] if db_row else None
            db_under_odds = db_row[5] if db_row else None
            
            custom_home_odds = None
            custom_away_odds = None
            custom_over_odds = None
            custom_under_odds = None
            custom_over_under = None
            
            if db_row and db_row[3] == 1:
                if db_row[2] is not None:
                    custom_over_odds = db_row[0]
                    custom_under_odds = db_row[1]
                    custom_over_under = db_row[2]
                else:
                    custom_home_odds = db_row[0]
                    custom_away_odds = db_row[1]

            if fd_match:
                # Merge FanDuel odds
                home_odds = fd_match['home_odds']
                away_odds = fd_match['away_odds']
                
                p_home_raw = 1.0 / home_odds if home_odds > 0 else 0.5
                p_away_raw = 1.0 / away_odds if away_odds > 0 else 0.5
                sum_p = p_home_raw + p_away_raw
                
                home_implied_prob = p_home_raw / sum_p if sum_p > 0 else 0.5
                away_implied_prob = p_away_raw / sum_p if sum_p > 0 else 0.5
                
                bookmaker_payload = {
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'home_implied_prob': round(home_implied_prob * 100, 1),
                    'away_implied_prob': round(away_implied_prob * 100, 1),
                    'closing_spread': fd_match['closing_spread'],
                    'over_under': fd_match['over_under'],
                    'over_odds': fd_match.get('over_odds', 1.91),
                    'under_odds': fd_match.get('under_odds', 1.91),
                    'custom_home_odds': custom_home_odds,
                    'custom_away_odds': custom_away_odds,
                    'custom_over_odds': custom_over_odds,
                    'custom_under_odds': custom_under_odds,
                    'custom_over_under': custom_over_under,
                    'is_fanduel': True
                }
            else:
                # Fallback to ELO-derived odds
                r_home = LATEST_ELOS.get(home_team_full, 1500.0)
                r_away = LATEST_ELOS.get(away_team_full, 1500.0)
                bm_odds = generate_betting_data(home_team_full, away_team_full, match_date, r_home, r_away)
                
                bookmaker_payload = {
                    'home_odds': bm_odds['BookieHomeOdds'],
                    'away_odds': bm_odds['BookieAwayOdds'],
                    'home_implied_prob': round(bm_odds['Prob_Home'] * 100, 1),
                    'away_implied_prob': round((1.0 - bm_odds['Prob_Home']) * 100, 1),
                    'closing_spread': bm_odds['ClosingSpread'],
                    'over_under': bm_odds['OverUnder'],
                    'over_odds': db_over_odds if db_over_odds is not None else 1.91,
                    'under_odds': db_under_odds if db_under_odds is not None else 1.91,
                    'custom_home_odds': custom_home_odds,
                    'custom_away_odds': custom_away_odds,
                    'custom_over_odds': custom_over_odds,
                    'custom_under_odds': custom_under_odds,
                    'custom_over_under': custom_over_under,
                    'is_fanduel': False
                }
                
            # Check if this game is in raw_matches (meaning it was played)
            cursor.execute("""
                SELECT HomeScore, AwayScore 
                FROM raw_matches 
                WHERE Date = ? AND (
                    (HomeTeam = ? AND AwayTeam = ?) OR 
                    (HomeTeam = ? AND AwayTeam = ?)
                )
            """, (match_date, home_team_full, away_team_full, away_team_full, home_team_full))
            match_row = cursor.fetchone()
            
            has_happened = False
            home_score = None
            away_score = None
            if match_row and match_row[0] >= 0 and match_row[1] >= 0:
                has_happened = True
                home_score = match_row[0]
                away_score = match_row[1]
            else:
                has_happened = match_date < today_str
            
            # Calculate Totals edges and Asymmetric Recommendation Logic
            over_prob = (pred['over_probability'] / 100.0) if (pred and 'over_probability' in pred) else 0.5
            under_prob = (pred['under_probability'] / 100.0) if (pred and 'under_probability' in pred) else 0.5

            b_over_odds = bookmaker_payload.get('custom_over_odds') or bookmaker_payload.get('over_odds') or 1.91
            b_under_odds = bookmaker_payload.get('custom_under_odds') or bookmaker_payload.get('under_odds') or 1.91

            p_over_raw = 1.0 / float(b_over_odds) if (b_over_odds and float(b_over_odds) > 0) else 0.5
            p_under_raw = 1.0 / float(b_under_odds) if (b_under_odds and float(b_under_odds) > 0) else 0.5
            sum_tot_p = p_over_raw + p_under_raw

            implied_over_prob = p_over_raw / sum_tot_p if sum_tot_p > 0 else 0.5
            implied_under_prob = p_under_raw / sum_tot_p if sum_tot_p > 0 else 0.5

            over_edge = round(over_prob - implied_over_prob, 3)
            under_edge = round(under_prob - implied_under_prob, 3)

            recommended_totals_side = 'OVER' if over_edge >= 0.03 else ('UNDER' if under_edge >= 0.07 else 'PASS')
            totals_tier = 'HIGH_EDGE_OVER' if over_edge >= 0.06 else ('VALUE_OVER' if over_edge >= 0.03 else ('CAUTION_UNDER' if under_edge < 0.07 and under_edge >= 0.03 else 'NEUTRAL'))

            upcoming_bets.append({
                'date': match_date,
                'home_team_abbr': home_abbr,
                'away_team_abbr': away_abbr,
                'home_team': REVERSE_TEAM_MAP.get(home_abbr.upper(), home_abbr),
                'away_team': REVERSE_TEAM_MAP.get(away_abbr.upper(), away_abbr),
                'home_prob': pred['home_win_probability'] if pred else 50.0,
                'away_prob': pred['away_win_probability'] if pred else 50.0,
                'predicted_spread': pred['predicted_spread'] if pred else 0.0,
                'dynamic_sigma': pred['dynamic_sigma'] if pred else None,
                'predicted_total': pred['predicted_total'] if pred else 160.0,
                'total_dynamic_sigma': pred['total_dynamic_sigma'] if pred else None,
                'over_probability': pred['over_probability'] if pred else 50.0,
                'under_probability': pred['under_probability'] if pred else 50.0,
                'over_edge': over_edge,
                'under_edge': under_edge,
                'env_scoring_gap': 12.5,
                'env_over_rate': 71.4,
                'over_roi_pattern': 70.74,
                'recommended_totals_side': recommended_totals_side,
                'totals_tier': totals_tier,
                'polymarket_home_prob': round(home_yes_price * 100, 1),
                'polymarket_away_prob': round(away_yes_price * 100, 1),
                'home_price': home_yes_price,
                'away_price': away_yes_price,
                'polymarket_volume': volume,
                'home_injuries': home_injuries,
                'away_injuries': away_injuries,
                'home_health': pred['home_health'] if pred else None,
                'away_health': pred['away_health'] if pred else None,
                'bookmaker': bookmaker_payload,
                'confirmed_spread_bet': confirmed_spread_bet,
                'confirmed_total_bet': confirmed_total_bet,
                'has_happened': has_happened,
                'home_score': home_score,
                'away_score': away_score
            })
            
        conn.close()
        return jsonify(upcoming_bets)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def apply_db_custom_lines(df):
    """
    Merges custom Over/Under lines from raw_matches in wnba.db onto the dataset,
    ensuring any custom bookmaker lines inputted by the user or live bookie feeds are prioritized.
    """
    try:
        if not os.path.exists(DB_NAME):
            return df
        conn = sqlite3.connect(DB_NAME)
        db_raw = pd.read_sql_query("SELECT Date, HomeTeam, AwayTeam, OverUnder AS DB_OverUnder FROM raw_matches WHERE OverUnder IS NOT NULL AND OverUnder > 0", conn)
        conn.close()
        
        if db_raw.empty:
            return df
            
        merged = pd.merge(df, db_raw, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
        merged['OverUnder'] = np.where(merged['DB_OverUnder'].notna() & (merged['DB_OverUnder'] > 0), merged['DB_OverUnder'], merged['OverUnder'])
        merged = merged.drop(columns=['DB_OverUnder'])
        return merged
    except Exception as e:
        print("Error merging DB custom lines:", e)
        return df


@app.route('/api/team_totals', methods=['GET'])
def get_team_totals():
    """
    Returns team totals metrics for all 15 WNBA franchises.
    Query Params:
      - season: '2026', '2025', '2024', 'all' (default: '2026')
      - window: 'all', '5', '10' (default: 'all')
    """
    try:
        season_param = request.args.get('season', '2026')
        window_param = request.args.get('window', 'all')
        
        data_path = os.path.join(base_dir, 'ml_ready_data.csv')
        df = pd.read_csv(data_path)
        df = apply_db_custom_lines(df)
        
        if season_param != 'all':
            try:
                s_int = int(season_param)
                df = df[df['Season'] == s_int].copy()
            except ValueError:
                pass
                
        df = df.sort_values('Date').reset_index(drop=True)
        
        home_df = df[['Date', 'HomeTeam', 'HomeScore', 'AwayScore', 'OverUnder']].copy()
        home_df.columns = ['Date', 'Team', 'PtsFor', 'PtsAgainst', 'OverUnder']
        
        away_df = df[['Date', 'AwayTeam', 'AwayScore', 'HomeScore', 'OverUnder']].copy()
        away_df.columns = ['Date', 'Team', 'PtsFor', 'PtsAgainst', 'OverUnder']
        
        team_games = pd.concat([home_df, away_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
        team_games['CombinedTotal'] = team_games['PtsFor'] + team_games['PtsAgainst']
        team_games['IsOver'] = team_games['CombinedTotal'] > team_games['OverUnder']
        
        results = []
        for team, group in team_games.groupby('Team'):
            team_full = REVERSE_TEAM_MAP.get(team, team)
            sub_group = group.copy()
            if window_param in ('5', '10'):
                w_int = int(window_param)
                sub_group = sub_group.tail(w_int)
                
            gp = len(sub_group)
            if gp == 0:
                continue
                
            pts_for = float(sub_group['PtsFor'].mean())
            pts_against = float(sub_group['PtsAgainst'].mean())
            avg_total = float(sub_group['CombinedTotal'].mean())
            avg_line = float(sub_group['OverUnder'].mean())
            over_hits = int(sub_group['IsOver'].sum())
            over_pct = float(round((over_hits / gp) * 100, 1))
            diff = float(round(avg_total - avg_line, 1))
            
            results.append({
                'team_abbr': team,
                'team_name': team_full,
                'gp': gp,
                'pts_for': round(pts_for, 1),
                'pts_against': round(pts_against, 1),
                'avg_total': round(avg_total, 1),
                'avg_line': round(avg_line, 1),
                'diff': diff,
                'over_hits': over_hits,
                'over_pct': over_pct
            })
            
        results.sort(key=lambda x: x['avg_total'], reverse=True)
        return jsonify({
            'season': season_param,
            'window': window_param,
            'total_teams': len(results),
            'teams': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/h2h_analytics', methods=['GET'])
def get_h2h_analytics():
    """
    Returns Head-to-Head matchup totals analytics between team_a and team_b.
    Query Params:
      - team_a: Full name or abbreviation (e.g. 'Toronto Tempo' or 'TOR')
      - team_b: Full name or abbreviation (e.g. 'Indiana Fever' or 'IND')
      - window: 'all', '5', '10' (default: 'all')
      - season: 'all', '2026', '2025', '2024' (default: 'all')
    """
    try:
        raw_team_a = request.args.get('team_a', 'Toronto Tempo')
        raw_team_b = request.args.get('team_b', 'Indiana Fever')
        window_param = request.args.get('window', 'all')
        season_param = request.args.get('season', 'all')
        
        norm_a = normalize_team_name(raw_team_a)
        norm_b = normalize_team_name(raw_team_b)
        
        team_a_full = REVERSE_TEAM_MAP.get(norm_a.upper(), norm_a)
        team_b_full = REVERSE_TEAM_MAP.get(norm_b.upper(), norm_b)
        
        data_path = os.path.join(base_dir, 'ml_ready_data.csv')
        df = pd.read_csv(data_path)
        df = apply_db_custom_lines(df)
        
        if season_param != 'all':
            try:
                s_int = int(season_param)
                df = df[df['Season'] == s_int].copy()
            except ValueError:
                pass
                
        df = df.sort_values('Date').reset_index(drop=True)
        
        mask_h2h = (
            ((df['HomeTeam'] == team_a_full) & (df['AwayTeam'] == team_b_full)) |
            ((df['HomeTeam'] == team_b_full) & (df['AwayTeam'] == team_a_full))
        )
        df_h2h = df[mask_h2h].copy().reset_index(drop=True)
        
        if window_param in ('5', '10'):
            w_int = int(window_param)
            df_h2h = df_h2h.tail(w_int).copy().reset_index(drop=True)
            
        matches_list = []
        team_a_wins = 0
        team_b_wins = 0
        over_hits = 0
        
        for idx, row in df_h2h.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            h_score = int(row['HomeScore'])
            a_score = int(row['AwayScore'])
            total_score = h_score + a_score
            line = float(row['OverUnder'])
            is_over = total_score > line
            if is_over:
                over_hits += 1
                
            winner = h_team if h_score > a_score else a_team
            if winner == team_a_full:
                team_a_wins += 1
            else:
                team_b_wins += 1
                
            team_a_pts = h_score if h_team == team_a_full else a_score
            team_b_pts = a_score if h_team == team_a_full else h_score
            
            matches_list.append({
                'date': str(row['Date']),
                'season': int(row['Season']),
                'home_team': h_team,
                'away_team': a_team,
                'home_score': h_score,
                'away_score': a_score,
                'team_a_score': team_a_pts,
                'team_b_score': team_b_pts,
                'total_score': total_score,
                'over_under': line,
                'is_over': is_over,
                'winner': winner
            })
            
        total_games = len(matches_list)
        if total_games > 0:
            avg_total_pts = float(round(np.mean([m['total_score'] for m in matches_list]), 1))
            avg_line = float(round(np.mean([m['over_under'] for m in matches_list]), 1))
            avg_team_a_pts = float(round(np.mean([m['team_a_score'] for m in matches_list]), 1))
            avg_team_b_pts = float(round(np.mean([m['team_b_score'] for m in matches_list]), 1))
            over_pct = float(round((over_hits / total_games) * 100, 1))
            line_diff = float(round(avg_total_pts - avg_line, 1))
        else:
            avg_total_pts = 0.0
            avg_line = 0.0
            avg_team_a_pts = 0.0
            avg_team_b_pts = 0.0
            over_pct = 0.0
            line_diff = 0.0
            
        return jsonify({
            'team_a': team_a_full,
            'team_b': team_b_full,
            'window': window_param,
            'season': season_param,
            'total_games': total_games,
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'avg_team_a_pts': avg_team_a_pts,
            'avg_team_b_pts': avg_team_b_pts,
            'avg_total_pts': avg_total_pts,
            'avg_line': avg_line,
            'line_diff': line_diff,
            'over_hits': over_hits,
            'over_pct': over_pct,
            'matches': list(reversed(matches_list))
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/team_game_history', methods=['GET'])
def get_team_game_history():
    """
    Returns game history and Over/Under analytics for a specific team.
    Query Params:
      - team: Team full name or abbreviation (e.g. 'Indiana Fever' or 'IND')
      - season: '2026', '2025', '2024', 'all' (default: '2026')
      - window: 'all', '5', '10' (default: 'all')
    """
    try:
        raw_team = request.args.get('team', 'Indiana Fever').strip()
        season_param = request.args.get('season', '2026')
        window_param = request.args.get('window', 'all')
        
        norm_team = normalize_team_name(raw_team)
        
        team_full = None
        norm_upper = norm_team.upper()
        
        if norm_upper in REVERSE_TEAM_MAP:
            team_full = REVERSE_TEAM_MAP[norm_upper]
        else:
            for abbr, full in REVERSE_TEAM_MAP.items():
                if full.upper() == norm_upper:
                    team_full = full
                    break
                    
        if not team_full:
            team_full = norm_team
            
        team_abbr = get_team_abbr(team_full)
        
        data_path = os.path.join(base_dir, 'ml_ready_data.csv')
        df = pd.read_csv(data_path)
        df = apply_db_custom_lines(df)
        
        mask = (df['HomeTeam'] == team_full) | (df['AwayTeam'] == team_full)
        df_team = df[mask].copy()
        
        if season_param != 'all':
            try:
                s_int = int(season_param)
                df_team = df_team[df_team['Season'] == s_int].copy()
            except ValueError:
                pass
                
        df_team = df_team.sort_values('Date').reset_index(drop=True)
        
        if window_param in ('5', '10'):
            try:
                w_int = int(window_param)
                df_team = df_team.tail(w_int).copy().reset_index(drop=True)
            except ValueError:
                pass
                
        games_list = []
        over_hits = 0
        under_hits = 0
        push_hits = 0
        
        for _, row in df_team.iterrows():
            h_team = str(row['HomeTeam'])
            a_team = str(row['AwayTeam'])
            h_score = int(row['HomeScore'])
            a_score = int(row['AwayScore'])
            is_home = bool(h_team == team_full)
            
            team_score = h_score if is_home else a_score
            opp_score = a_score if is_home else h_score
            opponent = a_team if is_home else h_team
            
            total_score = h_score + a_score
            over_under = float(row['OverUnder'])
            line_diff = round(float(total_score - over_under), 1)
            
            if total_score > over_under:
                outcome = 'OVER'
                over_hits += 1
            elif total_score < over_under:
                outcome = 'UNDER'
                under_hits += 1
            else:
                outcome = 'PUSH'
                push_hits += 1
                
            games_list.append({
                'date': str(row['Date']),
                'season': int(row['Season']),
                'home_team': h_team,
                'away_team': a_team,
                'home_score': h_score,
                'away_score': a_score,
                'is_home': is_home,
                'team_score': team_score,
                'opp_score': opp_score,
                'opponent': opponent,
                'total_score': total_score,
                'over_under': over_under,
                'line_diff': line_diff,
                'outcome': outcome
            })
            
        total_games = len(games_list)
        if total_games > 0:
            avg_total_pts = float(round(np.mean([g['total_score'] for g in games_list]), 1))
            avg_line = float(round(np.mean([g['over_under'] for g in games_list]), 1))
            over_pct = float(round((over_hits / total_games) * 100, 1))
            line_diff = float(round(avg_total_pts - avg_line, 1))
        else:
            avg_total_pts = 0.0
            avg_line = 0.0
            over_pct = 0.0
            line_diff = 0.0
            
        return jsonify({
            'team_name': team_full,
            'team_abbr': team_abbr,
            'season': season_param,
            'window': window_param,
            'total_games': total_games,
            'over_hits': over_hits,
            'under_hits': under_hits,
            'push_hits': push_hits,
            'over_pct': over_pct,
            'avg_total_pts': avg_total_pts,
            'avg_line': avg_line,
            'line_diff': line_diff,
            'games': list(reversed(games_list))
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Run startup calculations
init_app_data()

if __name__ == '__main__':
    # Flask runs on port 5001 as requested
    app.run(host='0.0.0.0', port=5001, debug=False)
