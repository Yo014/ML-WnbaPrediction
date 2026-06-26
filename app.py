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

# Global cache variables
MODEL = None
METADATA = None
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
    global MODEL, METADATA, ALL_TEAMS, LATEST_TEAM_EMAS, OVERALL_EMA_MEANS
    global TALENT_FLOORS_2026, LATEST_REF_EMAS, GLOBAL_REF_DEFAULTS, LATEST_ELOS
    global WNBA_2026_SCHEDULE
    
    # Load model and metadata
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Trained model or metadata file is missing. Run train_model.py first.")
        
    with open(MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
        
    with open(METADATA_PATH, 'r') as f:
        METADATA = json.load(f)
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Fetch all teams
    cursor.execute("SELECT DISTINCT Team FROM player_stats WHERE Season = 2026 ORDER BY Team")
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
            'PF': row['HomePF'], 'Opp_DREB': row['AwayDREB']
        })
        team_games.append({
            'Date': row['Date'], 'Season': row['Season'],
            'Team': row['AwayTeam'], 'Opponent': row['HomeTeam'], 'Role': 'Away',
            'PtsScored': row['AwayPtsScored'], 'PtsConceded': row['AwayPtsConceded'],
            'Possessions': row['AwayPossessions'], 'FGA': row['AwayFGA'], 'FTA': row['AwayFTA'],
            'OREB': row['AwayOREB'], 'TOV': row['AwayTOV'], 'FGM': row['AwayFGM'],
            'FG3M': row['AwayFG3M'], 'FTM': row['AwayFTM'], 'DREB': row['AwayDREB'],
            'PF': row['AwayPF'], 'Opp_DREB': row['HomeDREB']
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
    
    grouped = df_team_games.groupby(['Team', 'Season'])
    metrics = ['Offensive_Rating', 'Defensive_Rating', 'eFG%', 'TOV%', 'ORB%', 'FT_Rate']
    
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
    
    # Query players on the team in 2026
    cursor.execute("""
        SELECT Player, MIN, USG_PCT, BPM 
        FROM player_stats 
        WHERE Team = ? AND Season = 2026 
        ORDER BY Player
    """, (team_name,))
    rows = cursor.fetchall()
    
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
            'bpm': r[3] or 0.0,
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
        upcoming_only=upcoming_only
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
    """
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
            SELECT HomeScore, AwayScore, HomeTeam, AwayTeam
            FROM raw_matches
            WHERE Date = ? AND (
                (HomeTeam = ? AND AwayTeam = ?) OR
                (HomeTeam = ? AND AwayTeam = ?)
            )
        """, (match_date, home_full, away_full, away_full, home_full))
        match_row = cursor.fetchone()
        
        if match_row:
            home_score, away_score, actual_home_team, actual_away_team = match_row
            if home_score < 0 or away_score < 0:
                continue
            
            # Determine winner
            if home_score > away_score:
                actual_winner = actual_home_team
            else:
                actual_winner = actual_away_team
                
            # Determine if the bet was on the home team or away team
            bet_on_home = False
            bet_on_away = False
            
            rec_side_clean = recommended_side.strip().upper()
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

def make_prediction_for_matchup(home_team, away_team, prediction_date, crew_chief=None, home_injured_list=None, away_injured_list=None):
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
        missing_bpm = 0.0
        missing_minutes = 0.0
        injured_count = len(injured_players)
        
        for player in injured_players:
            cursor.execute("""
                SELECT MIN, USG_PCT, BPM 
                FROM player_stats 
                WHERE Player = ? 
                ORDER BY Season DESC 
                LIMIT 1
            """, (player,))
            row = cursor.fetchone()
            if row:
                min_avg = row[0] or 0.0
                usg_pct = row[1] or 0.0
                bpm = row[2] or 0.0
                
                missing_usage += usg_pct * 100.0
                missing_bpm += bpm
                missing_minutes += (min_avg / 200.0) * 100.0
                
        return {
            'Missing_Usage_Pct': round(missing_usage, 3),
            'Missing_BPM_Pct': round(missing_bpm, 3),
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
    conn.close()
    
    # Recalculate Net Rating EMAs
    home_net_5 = get_ema_value(home_emas, 'Offensive_Rating_EMA_5') - get_ema_value(home_emas, 'Defensive_Rating_EMA_5')
    home_net_10 = get_ema_value(home_emas, 'Offensive_Rating_EMA_10') - get_ema_value(home_emas, 'Defensive_Rating_EMA_10')
    away_net_5 = get_ema_value(away_emas, 'Offensive_Rating_EMA_5') - get_ema_value(away_emas, 'Defensive_Rating_EMA_5')
    away_net_10 = get_ema_value(away_emas, 'Offensive_Rating_EMA_10') - get_ema_value(away_emas, 'Defensive_Rating_EMA_10')
    
    # Reconstruct Feature Dictionary
    feature_dict = {
        'Home_Offensive_Rating_EMA_5': get_ema_value(home_emas, 'Offensive_Rating_EMA_5'),
        'Home_Defensive_Rating_EMA_5': get_ema_value(home_emas, 'Defensive_Rating_EMA_5'),
        'Home_Offensive_Rating_EMA_10': get_ema_value(home_emas, 'Offensive_Rating_EMA_10'),
        'Home_Defensive_Rating_EMA_10': get_ema_value(home_emas, 'Defensive_Rating_EMA_10'),
        'Away_Offensive_Rating_EMA_5': get_ema_value(away_emas, 'Offensive_Rating_EMA_5'),
        'Away_Defensive_Rating_EMA_5': get_ema_value(away_emas, 'Defensive_Rating_EMA_5'),
        'Away_Offensive_Rating_EMA_10': get_ema_value(away_emas, 'Offensive_Rating_EMA_10'),
        'Away_Defensive_Rating_EMA_10': get_ema_value(away_emas, 'Defensive_Rating_EMA_10'),
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
        'Away_eFG%_EMA_5': get_ema_value(away_emas, 'eFG%_EMA_5'),
        'Away_TOV%_EMA_5': get_ema_value(away_emas, 'TOV%_EMA_5'),
        'Away_ORB%_EMA_5': get_ema_value(away_emas, 'ORB%_EMA_5'),
        'Away_FT_Rate_EMA_5': get_ema_value(away_emas, 'FT_Rate_EMA_5'),
        'Away_eFG%_EMA_10': get_ema_value(away_emas, 'eFG%_EMA_10'),
        'Away_TOV%_EMA_10': get_ema_value(away_emas, 'TOV%_EMA_10'),
        'Away_ORB%_EMA_10': get_ema_value(away_emas, 'ORB%_EMA_10'),
        'Away_FT_Rate_EMA_10': get_ema_value(away_emas, 'FT_Rate_EMA_10'),
        'Home_Days_Rest': home_days_rest,
        'Home_Back_To_Back': home_b2b,
        'Home_Three_In_Four': home_three_in_four,
        'Away_Days_Rest': away_days_rest,
        'Away_Back_To_Back': away_b2b,
        'Away_Three_In_Four': away_three_in_four,
        'Rest_Diff': rest_diff,
        'Ref_Pts_EMA': ref_emas['Ref_Pts_EMA'],
        'Ref_Fouls_EMA': ref_emas['Ref_Fouls_EMA'],
        'Ref_HomeWin_EMA': ref_emas['Ref_HomeWin_EMA'],
        'Home_Talent_Floor': home_talent_floor,
        'Away_Talent_Floor': away_talent_floor,
        'Talent_Floor_Diff': talent_floor_diff,
        'BookieHomeOdds': odds['BookieHomeOdds'],
        'BookieAwayOdds': odds['BookieAwayOdds'],
        'OpeningSpread': odds['OpeningSpread'],
        'ClosingSpread': odds['ClosingSpread'],
        'OverUnder': odds['OverUnder'],
        'Prob_Home': odds['Prob_Home'],
        'Home_Missing_Usage_Pct': home_health['Missing_Usage_Pct'],
        'Away_Missing_Usage_Pct': away_health['Missing_Usage_Pct'],
        'Home_Missing_BPM_Pct': home_health['Missing_BPM_Pct'],
        'Away_Missing_BPM_Pct': away_health['Missing_BPM_Pct'],
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
        'H2H_Bias': h2h_bias
    }
    
    # Build dataframe matching features list order
    features_list = METADATA['features']
    features_df = pd.DataFrame([feature_dict])[features_list]
    
    # Predict spread and win probabilities
    if isinstance(MODEL, dict) and 'regressor' in MODEL and 'classifier' in MODEL:
        predicted_spread = float(MODEL['regressor'].predict(features_df)[0])
        home_win_prob = float(MODEL['classifier'].predict_proba(features_df)[0, 1])
    else:
        predicted_spread = float(MODEL.predict(features_df)[0])
        sigma_residuals = METADATA['sigma_residuals']
        home_win_prob = float(norm.cdf(predicted_spread / sigma_residuals))
        
    away_win_prob = 1.0 - home_win_prob
    
    return {
        'predicted_spread': round(predicted_spread, 2),
        'home_win_probability': round(home_win_prob * 100, 1),
        'away_win_probability': round(away_win_prob * 100, 1),
        'home_health': home_health,
        'away_health': away_health,
        'odds': {
            'bookie_home_odds': odds['BookieHomeOdds'],
            'bookie_away_odds': odds['BookieAwayOdds'],
            'opening_spread': odds['OpeningSpread'],
            'closing_spread': odds['ClosingSpread'],
            'over_under': odds['OverUnder'],
            'implied_prob_home': round(odds['Prob_Home'] * 100, 1)
        },
        'differentials': {
            'talent_floor_diff': round(talent_floor_diff, 2),
            'net_rating_diff_5': round(home_net_5 - away_net_5, 2),
            'net_rating_diff_10': round(home_net_10 - away_net_10, 2),
            'rest_diff': rest_diff,
            'h2h_bias': round(h2h_bias * 100, 1)
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
    
    try:
        response = make_prediction_for_matchup(
            home_team=home_team,
            away_team=away_team,
            prediction_date=prediction_date,
            crew_chief=crew_chief,
            home_injured_list=home_injured_list,
            away_injured_list=away_injured_list
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upcoming_predictions', methods=['GET'])
def upcoming_predictions():
    start_date = request.args.get('start_date', '2026-06-20')
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
        
        try:
            pred = make_prediction_for_matchup(
                home_team=norm_home,
                away_team=norm_away,
                prediction_date=match_date
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
            
            # Kelly calculations: kelly = (p - m) / (1 - m)
            kelly_home = (model_prob_home - home_yes_price) / (1.0 - home_yes_price) if home_yes_price < 1.0 else 0.0
            quarter_kelly_home = max(0.0, 0.25 * kelly_home)
            quarter_kelly_home_capped = min(0.15, quarter_kelly_home)
            
            kelly_away = (model_prob_away - away_yes_price) / (1.0 - away_yes_price) if away_yes_price < 1.0 else 0.0
            quarter_kelly_away = max(0.0, 0.25 * kelly_away)
            quarter_kelly_away_capped = min(0.15, quarter_kelly_away)
            
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
        # Check if exists
        cursor.execute("""
            SELECT id FROM confirmed_bets 
            WHERE match_date = ? AND home_team = ? AND away_team = ?
        """, (match_date, home_abbr, away_abbr))
        row = cursor.fetchone()
        
        if row:
            cursor.execute("""
                UPDATE confirmed_bets
                SET recommended_side = ?, wager_type = ?, wager_amount = ?, odds = ?, outcome = NULL, bankroll_change = ?
                WHERE match_date = ? AND home_team = ? AND away_team = ?
            """, (recommended_side, wager_type, wager_amount, odds, -wager_amount, match_date, home_abbr, away_abbr))
            message = 'Bet updated successfully'
        else:
            cursor.execute("""
                INSERT INTO confirmed_bets (match_date, home_team, away_team, recommended_side, wager_type, wager_amount, odds, outcome, bankroll_change)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, (match_date, home_abbr, away_abbr, recommended_side, wager_type, wager_amount, odds, -wager_amount))
            message = 'Bet confirmed successfully'
            
        # Update raw_matches with custom odds if they are provided
        custom_home_odds = data.get('custom_home_odds')
        custom_away_odds = data.get('custom_away_odds')
        
        if custom_home_odds is not None or custom_away_odds is not None:
            home_full = REVERSE_TEAM_MAP.get(home_abbr.upper(), home_abbr)
            away_full = REVERSE_TEAM_MAP.get(away_abbr.upper(), away_abbr)
            
            # Check if record exists in raw_matches
            cursor.execute("""
                SELECT id FROM raw_matches 
                WHERE Date = ? AND (
                    (HomeTeam = ? AND AwayTeam = ?) OR
                    (HomeTeam = ? AND AwayTeam = ?)
                )
            """, (match_date, home_full, away_full, away_full, home_full))
            match_row = cursor.fetchone()
            
            if match_row:
                cursor.execute("""
                    UPDATE raw_matches
                    SET BookieHomeOdds = ?, BookieAwayOdds = ?, IsFanduelOdds = 1
                    WHERE id = ?
                """, (custom_home_odds, custom_away_odds, match_row[0]))
            else:
                cursor.execute("""
                    INSERT INTO raw_matches (
                        Date, HomeTeam, AwayTeam, HomeScore, AwayScore,
                        BookieHomeOdds, BookieAwayOdds, IsFanduelOdds
                    ) VALUES (?, ?, ?, -1, -1, ?, ?, 1)
                """, (match_date, home_full, away_full, custom_home_odds, custom_away_odds))
            
        conn.commit()
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/delete_bet', methods=['POST'])
def delete_bet():
    data = request.get_json() or {}
    if not data and request.form:
        data = request.form
        
    match_date = data.get('match_date') or data.get('date')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    
    if not all([match_date, home_team, away_team]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    home_abbr = get_team_abbr(home_team)
    away_abbr = get_team_abbr(away_team)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM confirmed_bets
            WHERE match_date = ? AND home_team = ? AND away_team = ?
        """, (match_date, home_abbr, away_abbr))
        
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
        confirmed_bets_map = {}
        for r in bet_rows:
            b_id, b_date, b_home, b_away, b_rec, b_type, b_amount, b_odds, b_outcome, b_change, b_at = r
            h_abbr = get_team_abbr(b_home)
            a_abbr = get_team_abbr(b_away)
            confirmed_bets_map[(b_date, h_abbr, a_abbr)] = {
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
            confirmed_bet = confirmed_bets_map.get(key, None)
            
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
            
            # Predict using our prediction helper
            try:
                pred = make_prediction_for_matchup(
                    home_team=home_team_full,
                    away_team=away_team_full,
                    prediction_date=match_date,
                    home_injured_list=home_injured_names,
                    away_injured_list=away_injured_names
                )
            except Exception as e:
                print(f"Prediction failed for matchup {home_team_full} vs {away_team_full} on {match_date}: {e}")
                pred = None
            
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
            
            upcoming_bets.append({
                'date': match_date,
                'home_team_abbr': home_abbr,
                'away_team_abbr': away_abbr,
                'home_team': REVERSE_TEAM_MAP.get(home_abbr.upper(), home_abbr),
                'away_team': REVERSE_TEAM_MAP.get(away_abbr.upper(), away_abbr),
                'home_prob': pred['home_win_probability'] if pred else 50.0,
                'away_prob': pred['away_win_probability'] if pred else 50.0,
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
                'confirmed_bet': confirmed_bet,
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




# Run startup calculations
init_app_data()

if __name__ == '__main__':
    # Flask runs on port 5001 as requested
    app.run(host='0.0.0.0', port=5001, debug=True)
