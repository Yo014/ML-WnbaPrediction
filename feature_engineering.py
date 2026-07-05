import sqlite3
import pandas as pd
import numpy as np
from db_manager import DB_NAME, get_connection

def get_chronological_global_means(df, date_col, cols):
    """
    Computes league-wide chronological means for each column in cols.
    For any row, the mean uses only data from rows with date_col < current_row_date.
    """
    # Group by Date to get daily sums and counts
    daily_sum = df.groupby(date_col)[cols].sum()
    daily_count = df.groupby(date_col)[cols].count()
    
    cum_sum = daily_sum.cumsum().shift(1)
    cum_count = daily_count.cumsum().shift(1)
    
    prior_mean = cum_sum / cum_count
    
    # Fallback for the first date: use the mean of the first date's games
    first_date_means = df.groupby(date_col)[cols].mean().iloc[0]
    prior_mean = prior_mean.fillna(first_date_means)
    
    return prior_mean

def calculate_talent_floors(conn):
    """
    Computes the Talent Floor for each team-season as the sum of
    the previous-season Win Shares of players on the roster.
    """
    # Load player stats
    df_players = pd.read_sql_query("SELECT Season, Player, Team, WS FROM player_stats", conn)
    
    # Aggregate player WS by Player and Season (in case of trades or duplicate listings)
    player_ws = df_players.groupby(['Player', 'Season'])['WS'].sum().reset_index()
    player_ws_dict = {(row['Player'], row['Season']): row['WS'] for _, row in player_ws.iterrows()}
    
    # Calculate Talent Floor for each team-season
    talent_floors = []
    for (team, season), group in df_players.groupby(['Team', 'Season']):
        prev_season = season - 1
        sum_prev_ws = 0.0
        for player in group['Player'].unique():
            # Get the player's Win Shares from the previous season
            prev_ws = player_ws_dict.get((player, prev_season), 0.0)
            sum_prev_ws += prev_ws
        talent_floors.append({
            'Team': team,
            'Season': season,
            'Talent_Floor': round(sum_prev_ws, 2)
        })
    return pd.DataFrame(talent_floors)

def calculate_referee_stats(df_matches):
    """
    Calculates rolling EMA (span=20, shifted by 1) of Crew Chief officiating statistics:
    - Average total points scored in games officiated.
    - Average personal fouls called per game.
    - Crew chief historical home team win percentage.
    """
    # Sort matches chronologically to ensure no future information leaks
    df_matches = df_matches.sort_values('Date').reset_index(drop=True)
    
    # Calculate game-level referee metrics
    df_matches['Game_Total_Points'] = df_matches['HomeScore'] + df_matches['AwayScore']
    df_matches['Game_Total_Fouls'] = df_matches['HomePF'] + df_matches['AwayPF']
    df_matches['Game_Home_Win'] = (df_matches['HomeScore'] > df_matches['AwayScore']).astype(float)
    
    # Group by CrewChief
    ref_groups = df_matches.groupby('CrewChief')
    
    # Create target columns
    df_matches['Ref_Pts_EMA'] = np.nan
    df_matches['Ref_Fouls_EMA'] = np.nan
    df_matches['Ref_HomeWin_EMA'] = np.nan
    
    # Compute EMA span=20, shifted by 1 to prevent data leakage
    for chief, group in ref_groups:
        idx = group.index
        # Compute EWM on the chief's games
        pts_ema = group['Game_Total_Points'].ewm(span=20, adjust=False).mean().shift(1)
        fouls_ema = group['Game_Total_Fouls'].ewm(span=20, adjust=False).mean().shift(1)
        homewin_ema = group['Game_Home_Win'].ewm(span=20, adjust=False).mean().shift(1)
        
        df_matches.loc[idx, 'Ref_Pts_EMA'] = pts_ema
        df_matches.loc[idx, 'Ref_Fouls_EMA'] = fouls_ema
        df_matches.loc[idx, 'Ref_HomeWin_EMA'] = homewin_ema
        
    # Global fallbacks for Crew Chiefs officiating their first games (no history)
    # Using chronological global means to avoid future data leakage
    chrono_ref_means = get_chronological_global_means(df_matches, 'Date', ['Game_Total_Points', 'Game_Total_Fouls', 'Game_Home_Win'])
    df_matches['Ref_Pts_EMA'] = df_matches['Ref_Pts_EMA'].fillna(chrono_ref_means['Game_Total_Points'])
    df_matches['Ref_Fouls_EMA'] = df_matches['Ref_Fouls_EMA'].fillna(chrono_ref_means['Game_Total_Fouls'])
    df_matches['Ref_HomeWin_EMA'] = df_matches['Ref_HomeWin_EMA'].fillna(chrono_ref_means['Game_Home_Win'])
    
    # Drop intermediate columns
    df_matches = df_matches.drop(columns=['Game_Total_Points', 'Game_Total_Fouls', 'Game_Home_Win'])
    return df_matches

def calculate_h2h_bias(df_matches):
    """
    Computes Head-to-Head (H2H) Bias: Home team win rate against this specific away team
    over the last 2 seasons (current and previous season) prior to the match date.
    """
    h2h_win_rates = []
    
    # Sort chronologically to prevent any potential leakage
    df_matches = df_matches.sort_values('Date').reset_index(drop=True)
    
    for idx, row in df_matches.iterrows():
        h = row['HomeTeam']
        a = row['AwayTeam']
        d = row['Date']
        s = int(d[:4])
        
        # Historical matches in the last 2 seasons (season >= s - 1) before current date d
        cond_h_home = (df_matches['HomeTeam'] == h) & (df_matches['AwayTeam'] == a) & (df_matches['Date'] < d) & (df_matches['Date'].str[:4].astype(int) >= s - 1)
        cond_h_away = (df_matches['HomeTeam'] == a) & (df_matches['AwayTeam'] == h) & (df_matches['Date'] < d) & (df_matches['Date'].str[:4].astype(int) >= s - 1)
        
        prev_home = df_matches[cond_h_home]
        prev_away = df_matches[cond_h_away]
        
        total_games = len(prev_home) + len(prev_away)
        if total_games == 0:
            win_rate = 0.5  # Neutral default when no prior meetings
        else:
            wins_home = (prev_home['HomeScore'] > prev_home['AwayScore']).sum()
            wins_away = (prev_away['AwayScore'] > prev_away['HomeScore']).sum()
            win_rate = (wins_home + wins_away) / total_games
            
        h2h_win_rates.append(win_rate)
        
    df_matches['H2H_Bias'] = h2h_win_rates
    return df_matches

def main():
    print("Connecting to database to load matches...")
    conn = get_connection()
    df_matches = pd.read_sql_query("SELECT * FROM raw_matches WHERE HomeScore >= 0 AND AwayScore >= 0", conn)
    
    # Parse Date and add Season
    df_matches['Season'] = df_matches['Date'].str[:4].astype(int)
    
    # 1. Melt matches to team games long format
    print("Constructing team-game long format for rolling calculations...")
    team_games = []
    for idx, row in df_matches.iterrows():
        # Home team record
        team_games.append({
            'Match_Id': row['id'],
            'Date': row['Date'],
            'Season': row['Season'],
            'Team': row['HomeTeam'],
            'Opponent': row['AwayTeam'],
            'Role': 'Home',
            'PtsScored': row['HomePtsScored'],
            'PtsConceded': row['HomePtsConceded'],
            'Possessions': row['HomePossessions'],
            'FGA': row['HomeFGA'],
            'FTA': row['HomeFTA'],
            'OREB': row['HomeOREB'],
            'TOV': row['HomeTOV'],
            'FGM': row['HomeFGM'],
            'FG3M': row['HomeFG3M'],
            'FTM': row['HomeFTM'],
            'DREB': row['HomeDREB'],
            'PF': row['HomePF'],
            'MIN': row['HomeMIN'],
            'Opp_DREB': row['AwayDREB']
        })
        # Away team record
        team_games.append({
            'Match_Id': row['id'],
            'Date': row['Date'],
            'Season': row['Season'],
            'Team': row['AwayTeam'],
            'Opponent': row['HomeTeam'],
            'Role': 'Away',
            'PtsScored': row['AwayPtsScored'],
            'PtsConceded': row['AwayPtsConceded'],
            'Possessions': row['AwayPossessions'],
            'FGA': row['AwayFGA'],
            'FTA': row['AwayFTA'],
            'OREB': row['AwayOREB'],
            'TOV': row['AwayTOV'],
            'FGM': row['AwayFGM'],
            'FG3M': row['AwayFG3M'],
            'FTM': row['AwayFTM'],
            'DREB': row['AwayDREB'],
            'PF': row['AwayPF'],
            'MIN': row['AwayMIN'],
            'Opp_DREB': row['HomeDREB']
        })
    df_team_games = pd.DataFrame(team_games)
    
    # Sort chronologically by team and season
    df_team_games = df_team_games.sort_values(['Team', 'Season', 'Date']).reset_index(drop=True)
    
    # Calculate game-level rate metrics
    print("Calculating team game-level rating and four factors metrics...")
    df_team_games['Offensive_Rating'] = np.where(df_team_games['Possessions'] > 0, 100.0 * df_team_games['PtsScored'] / df_team_games['Possessions'], 0.0)
    df_team_games['Defensive_Rating'] = np.where(df_team_games['Possessions'] > 0, 100.0 * df_team_games['PtsConceded'] / df_team_games['Possessions'], 0.0)
    df_team_games['eFG%'] = np.where(df_team_games['FGA'] > 0, (df_team_games['FGM'] + 0.5 * df_team_games['FG3M']) / df_team_games['FGA'], 0.0)
    df_team_games['TOV%'] = np.where((df_team_games['FGA'] + 0.44 * df_team_games['FTA'] + df_team_games['TOV']) > 0, df_team_games['TOV'] / (df_team_games['FGA'] + 0.44 * df_team_games['FTA'] + df_team_games['TOV']), 0.0)
    df_team_games['ORB%'] = np.where((df_team_games['OREB'] + df_team_games['Opp_DREB']) > 0, df_team_games['OREB'] / (df_team_games['OREB'] + df_team_games['Opp_DREB']), 0.0)
    df_team_games['FT_Rate'] = np.where(df_team_games['FGA'] > 0, df_team_games['FTM'] / df_team_games['FGA'], 0.0)
    
    # Calculate game-level Pace
    game_duration = df_team_games['MIN'] / 5.0
    df_team_games['Pace'] = np.where(game_duration > 0, 40.0 * df_team_games['Possessions'] / game_duration, df_team_games['Possessions'])
    
    # Group by Team and Season to compute EMAs with start-of-season carry-over
    print("Computing EMAs (span=5 and span=10) with start-of-season carry-over...")
    metrics = ['Offensive_Rating', 'Defensive_Rating', 'eFG%', 'TOV%', 'ORB%', 'FT_Rate', 'Pace']
    
    # Precompute Talent Floors and changes for early-season stats Bayesian prior blending
    df_tf = calculate_talent_floors(conn)
    tf_dict = {(row['Team'], row['Season']): row['Talent_Floor'] for _, row in df_tf.iterrows()}
    tf_change_dict = {}
    for _, row in df_tf.iterrows():
        team = row['Team']
        season = row['Season']
        tf_curr = row['Talent_Floor']
        tf_prev = tf_dict.get((team, season - 1))
        if tf_prev is None:
            tf_change = 0.0
        else:
            tf_change = tf_curr - tf_prev
        tf_change_dict[(team, season)] = tf_change

    # Precompute game number in season from df_team_games
    df_team_games['Game_Number_In_Season'] = df_team_games.groupby(['Team', 'Season']).cumcount() + 1
    gn_map = df_team_games.set_index(['Team', 'Season', 'Date'])['Game_Number_In_Season'].to_dict()
    
    # Precompute chronological global means for all metrics to use as starting EMA fallback for 2018 / new teams
    chrono_means_unique = get_chronological_global_means(df_team_games, 'Date', metrics)
    chrono_team_means = df_team_games[['Date']].join(chrono_means_unique, on='Date')
    
    # Initialize columns in df_team_games
    for span in [5, 10]:
        for col in metrics:
            df_team_games[f'{col}_EMA_{span}'] = np.nan
            
    seasons = sorted(df_team_games['Season'].unique())
    prev_season_final_ema = {}
    
    for span in [5, 10]:
        for col in metrics:
            for S in seasons:
                if S > 2018:
                    league_mean_prev = df_team_games[df_team_games['Season'] == S - 1][col].mean()
                else:
                    league_mean_prev = None
                    
                teams_in_season = df_team_games[df_team_games['Season'] == S]['Team'].unique()
                for T in teams_in_season:
                    team_season_mask = (df_team_games['Team'] == T) & (df_team_games['Season'] == S)
                    team_season_df = df_team_games[team_season_mask]
                    
                    vals = team_season_df[col].values
                    
                    if S > 2018:
                        prev_ema = prev_season_final_ema.get((span, col, T))
                        if prev_ema is not None:
                            EMA_start = 0.75 * prev_ema + 0.25 * league_mean_prev
                        else:
                            EMA_start = league_mean_prev
                    else:
                        # 2018: use chronological mean prior to team's first game of the season
                        EMA_start = chrono_team_means.loc[team_season_df.index[0], col]
                        
                    # Determine prior value based on column and Talent_Floor_Change
                    tf_change = tf_change_dict.get((T, S), 0.0)
                    if col == 'Offensive_Rating':
                        Prior = EMA_start + 0.15 * tf_change
                    elif col == 'Defensive_Rating':
                        Prior = EMA_start - 0.15 * tf_change
                    elif col == 'eFG%':
                        Prior = EMA_start + 0.0005 * tf_change
                    elif col == 'TOV%':
                        Prior = EMA_start - 0.0002 * tf_change
                    elif col == 'ORB%':
                        Prior = EMA_start + 0.0005 * tf_change
                    elif col == 'FT_Rate':
                        Prior = EMA_start + 0.0005 * tf_change
                    elif col == 'Pace':
                        Prior = EMA_start
                    else:
                        Prior = EMA_start

                    # Prepend EMA_start to the season's feature series before running ewm
                    prepended_vals = np.insert(vals, 0, EMA_start)
                    s_prepended = pd.Series(prepended_vals)
                    ewm_series = s_prepended.ewm(span=span, adjust=False).mean()
                    
                    # shift and slice back
                    ewm_shifted = ewm_series.shift(1)
                    ewm_final = ewm_shifted.iloc[1:].values
                    
                    # Calculate Bayesian credibility weight
                    n_vals = team_season_df['Game_Number_In_Season'].values
                    W = np.where(n_vals <= 8, n_vals / (n_vals + 4.0), 1.0)
                    
                    # Blend the computed ewm_final with the prior
                    ewm_final_blended = (1.0 - W) * Prior + W * ewm_final
                    
                    df_team_games.loc[team_season_mask, f'{col}_EMA_{span}'] = ewm_final_blended
                    
                    # Store final EMA for the next season carry over (applying blending)
                    N = len(vals)
                    W_final = N / (N + 4.0) if N <= 8 else 1.0
                    carry_over_blended = (1.0 - W_final) * Prior + W_final * ewm_series.iloc[-1]
                    prev_season_final_ema[(span, col, T)] = carry_over_blended
            
    # Compute Rest & Fatigue
    print("Computing rest & fatigue features...")
    df_team_games['Date_dt'] = pd.to_datetime(df_team_games['Date'])
    df_team_games['Days_Rest'] = df_team_games.groupby(['Team', 'Season'])['Date_dt'].diff().dt.days
    df_team_games['Days_Rest'] = df_team_games['Days_Rest'].fillna(7.0).clip(upper=7.0)
    df_team_games['Back_To_Back'] = np.where(df_team_games['Days_Rest'] == 1.0, 1.0, 0.0)
    
    # 3 games in 4 nights
    df_team_games['Three_In_Four_Days'] = df_team_games.groupby(['Team', 'Season'])['Date_dt'].diff(2).dt.days
    df_team_games['Three_In_Four'] = np.where(df_team_games['Three_In_Four_Days'] <= 3.0, 1.0, 0.0)
    
    # Split back into Home and Away subsets
    df_home_feats = df_team_games[df_team_games['Role'] == 'Home'].copy()
    df_away_feats = df_team_games[df_team_games['Role'] == 'Away'].copy()
    
    # Build columns mapping to merge back
    home_cols = {'Match_Id': 'id', 'Days_Rest': 'Home_Days_Rest', 'Back_To_Back': 'Home_Back_To_Back', 'Three_In_Four': 'Home_Three_In_Four'}
    away_cols = {'Match_Id': 'id', 'Days_Rest': 'Away_Days_Rest', 'Back_To_Back': 'Away_Back_To_Back', 'Three_In_Four': 'Away_Three_In_Four'}
    
    for span in [5, 10]:
        for col in metrics:
            home_cols[f'{col}_EMA_{span}'] = f'Home_{col}_EMA_{span}'
            away_cols[f'{col}_EMA_{span}'] = f'Away_{col}_EMA_{span}'
            
    df_home_subset = df_home_feats[list(home_cols.keys())].rename(columns=home_cols)
    df_away_subset = df_away_feats[list(away_cols.keys())].rename(columns=away_cols)
    
    print("Merging features back to match-level dataframe...")
    df_matches = pd.merge(df_matches, df_home_subset, on='id', how='left')
    df_matches = pd.merge(df_matches, df_away_subset, on='id', how='left')
    
    # Calculate fallbacks using chronological global means instead of future-leaking overall_mean
    chrono_match_means = df_matches[['Date']].join(chrono_means_unique, on='Date')
    for span in [5, 10]:
        for col in metrics:
            df_matches[f'Home_{col}_EMA_{span}'] = df_matches[f'Home_{col}_EMA_{span}'].fillna(chrono_match_means[col])
            df_matches[f'Away_{col}_EMA_{span}'] = df_matches[f'Away_{col}_EMA_{span}'].fillna(chrono_match_means[col])
            
    # Calculate Net Ratings and Net Rating EMAs
    for span in [5, 10]:
        df_matches[f'Home_Net_Rating_EMA_{span}'] = df_matches[f'Home_Offensive_Rating_EMA_{span}'] - df_matches[f'Home_Defensive_Rating_EMA_{span}']
        df_matches[f'Away_Net_Rating_EMA_{span}'] = df_matches[f'Away_Offensive_Rating_EMA_{span}'] - df_matches[f'Away_Defensive_Rating_EMA_{span}']
        
    # 2. Merge Squad Health
    print("Calculating dynamic squad health from historical_inactives, injuries, and player_stats...")
    
    # Load player stats, inactives and injuries
    df_players = pd.read_sql_query("SELECT * FROM player_stats", conn)
    df_hist_inactives = pd.read_sql_query("SELECT * FROM historical_inactives", conn)
    df_active_injuries = pd.read_sql_query("SELECT * FROM injuries", conn)
    
    # Build dictionary for fast player stats lookup: (Player, Season) -> (MIN, USG_PCT, BPM)
    player_stats_dict = {}
    for _, row in df_players.iterrows():
        player_stats_dict[(row['Player'], int(row['Season']))] = (
            float(row['MIN']),
            float(row['USG_PCT']),
            float(row['BPM'])
        )
        
    def get_player_season_stats(player, season):
        """Looks up player stats for a given season, with fallback to closest season or 0.0."""
        stats = player_stats_dict.get((player, season))
        if stats is not None:
            return stats
        
        # Closest season fallback
        all_seasons = [s for (p, s) in player_stats_dict.keys() if p == player]
        if all_seasons:
            closest_season = min(all_seasons, key=lambda x: abs(x - season))
            return player_stats_dict[(player, closest_season)]
            
        return (0.0, 0.0, 0.0)


    # Initialize health columns
    health_metrics = ['Missing_Usage_Pct', 'Missing_BPM_Pct', 'Missing_Minutes_Pct', 'Injured_Players_Count']
    for col in health_metrics:
        df_matches[f'Home_{col}'] = 0.0
        df_matches[f'Away_{col}'] = 0.0
        
    current_date_str = '2026-06-29'
    
    # Loop over matches to calculate health features dynamically
    for idx, row in df_matches.iterrows():
        d = row['Date']
        season = row['Season']
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        
        # Look up game numbers
        h_game_num = gn_map.get((h_team, season, d), 1)
        a_game_num = gn_map.get((a_team, season, d), 1)
        
        # Start of season threshold (first 5 games): use season - 1 stats
        h_season_for_stats = season - 1 if h_game_num <= 5 else season
        a_season_for_stats = season - 1 if a_game_num <= 5 else season
        
        # Get Home Team Inactives
        h_inactives = df_hist_inactives[(df_hist_inactives['Date'] == d) & (df_hist_inactives['Team'] == h_team)]['Player'].tolist()
        if not h_inactives and d >= current_date_str:
            h_inactives = df_active_injuries[df_active_injuries['Team'] == h_team]['Player'].tolist()
            
        # Get Away Team Inactives
        a_inactives = df_hist_inactives[(df_hist_inactives['Date'] == d) & (df_hist_inactives['Team'] == a_team)]['Player'].tolist()
        if not a_inactives and d >= current_date_str:
            a_inactives = df_active_injuries[df_active_injuries['Team'] == a_team]['Player'].tolist()
            
        # Home team metrics
        h_usg, h_bpm, h_min = 0.0, 0.0, 0.0
        for p in h_inactives:
            p_min, p_usg, p_bpm = get_player_season_stats(p, h_season_for_stats)
            h_usg += p_usg * 100.0
            h_bpm += p_bpm
            h_min += (p_min / 2.0)
            
        df_matches.loc[idx, 'Home_Missing_Usage_Pct'] = round(h_usg, 3)
        df_matches.loc[idx, 'Home_Missing_BPM_Pct'] = round(h_bpm, 3)
        df_matches.loc[idx, 'Home_Missing_Minutes_Pct'] = round(h_min, 3)
        df_matches.loc[idx, 'Home_Injured_Players_Count'] = len(h_inactives)
        
        # Away team metrics
        a_usg, a_bpm, a_min = 0.0, 0.0, 0.0
        for p in a_inactives:
            p_min, p_usg, p_bpm = get_player_season_stats(p, a_season_for_stats)
            a_usg += p_usg * 100.0
            a_bpm += p_bpm
            a_min += (p_min / 2.0)
            
        df_matches.loc[idx, 'Away_Missing_Usage_Pct'] = round(a_usg, 3)
        df_matches.loc[idx, 'Away_Missing_BPM_Pct'] = round(a_bpm, 3)
        df_matches.loc[idx, 'Away_Missing_Minutes_Pct'] = round(a_min, 3)
        df_matches.loc[idx, 'Away_Injured_Players_Count'] = len(a_inactives)
            
    # 3. Talent Floor
    print("Merging Talent Floor from player win shares...")
    df_matches = pd.merge(df_matches, df_tf.rename(columns={'Talent_Floor': 'Home_Talent_Floor'}), left_on=['HomeTeam', 'Season'], right_on=['Team', 'Season'], how='left').drop(columns=['Team'])
    df_matches = pd.merge(df_matches, df_tf.rename(columns={'Talent_Floor': 'Away_Talent_Floor'}), left_on=['AwayTeam', 'Season'], right_on=['Team', 'Season'], how='left').drop(columns=['Team'])
    df_matches['Home_Talent_Floor'] = df_matches['Home_Talent_Floor'].fillna(0.0)
    df_matches['Away_Talent_Floor'] = df_matches['Away_Talent_Floor'].fillna(0.0)
    
    # 4. Referee Stats
    print("Calculating Crew Chief officiating stats...")
    df_matches = calculate_referee_stats(df_matches)
    
    # 5. Market Data
    print("Calculating market probability...")
    df_matches['Prob_Home'] = (1.0 / df_matches['BookieHomeOdds']) / ((1.0 / df_matches['BookieHomeOdds']) + (1.0 / df_matches['BookieAwayOdds']))
    
    # Merge Polymarket historical odds
    print("Merging Polymarket historical odds...")
    POLY_TO_FULL = {
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
        'WAS': 'Washington Mystics'
    }
    try:
        df_poly = pd.read_sql_query("SELECT match_date, home_team, away_team, home_yes_price FROM polymarket_odds", conn)
        df_poly['HomeTeam'] = df_poly['home_team'].map(POLY_TO_FULL)
        df_poly['AwayTeam'] = df_poly['away_team'].map(POLY_TO_FULL)
        df_poly = df_poly.rename(columns={
            'match_date': 'Date',
            'home_yes_price': 'Poly_Prob_Home'
        })
        df_poly = df_poly.dropna(subset=['HomeTeam', 'AwayTeam'])
        df_poly = df_poly[['Date', 'HomeTeam', 'AwayTeam', 'Poly_Prob_Home']]
        df_matches = pd.merge(df_matches, df_poly, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    except Exception as e:
        print(f"Error merging Polymarket odds: {e}")
        df_matches['Poly_Prob_Home'] = np.nan
        
    df_matches['Poly_Prob_Home'] = df_matches['Poly_Prob_Home'].fillna(df_matches['Prob_Home'])
    df_matches['Market_Disagreement'] = df_matches['Prob_Home'] - df_matches['Poly_Prob_Home']
    
    # 6. Differentials
    print("Calculating feature differentials (Home - Away)...")
    for span in [5, 10]:
        df_matches[f'Net_Rating_Diff_{span}'] = df_matches[f'Home_Net_Rating_EMA_{span}'] - df_matches[f'Away_Net_Rating_EMA_{span}']
        df_matches[f'eFG%_Diff_{span}'] = df_matches[f'Home_eFG%_EMA_{span}'] - df_matches[f'Away_eFG%_EMA_{span}']
        df_matches[f'TOV%_Diff_{span}'] = df_matches[f'Home_TOV%_EMA_{span}'] - df_matches[f'Away_TOV%_EMA_{span}']
        df_matches[f'ORB%_Diff_{span}'] = df_matches[f'Home_ORB%_EMA_{span}'] - df_matches[f'Away_ORB%_EMA_{span}']
        df_matches[f'FT_Rate_Diff_{span}'] = df_matches[f'Home_FT_Rate_EMA_{span}'] - df_matches[f'Away_FT_Rate_EMA_{span}']
        
    df_matches['Rest_Diff'] = df_matches['Home_Days_Rest'] - df_matches['Away_Days_Rest']
    df_matches['Missing_Usage_Diff'] = df_matches['Home_Missing_Usage_Pct'] - df_matches['Away_Missing_Usage_Pct']
    df_matches['Talent_Floor_Diff'] = df_matches['Home_Talent_Floor'] - df_matches['Away_Talent_Floor']
    
    # 7. Head-to-Head Bias
    print("Calculating Head-to-Head (H2H) win bias...")
    df_matches = calculate_h2h_bias(df_matches)
    
    # Target variables
    df_matches['Home_Win'] = (df_matches['HomeScore'] > df_matches['AwayScore']).astype(int)
    df_matches['Score_Diff'] = df_matches['HomeScore'] - df_matches['AwayScore']
    
    # Save the output
    output_file = "ml_ready_data.csv"
    df_matches.to_csv(output_file, index=False)
    print(f"Feature engineering complete! Saved dataset with {len(df_matches)} matches to {output_file}.")
    conn.close()

if __name__ == "__main__":
    main()
