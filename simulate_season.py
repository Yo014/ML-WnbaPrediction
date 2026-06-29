import os
import pickle
import json
import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_metrics(probs, actuals):
    """Calculates Accuracy, Brier Score, and Log Loss for a series of probabilities and outcomes."""
    if len(probs) == 0:
        return {"accuracy": None, "brier_score": None, "log_loss": None}
    
    probs = np.array(probs, dtype=float)
    actuals = np.array(actuals, dtype=float)
    
    # Filter out any NaNs
    mask = ~np.isnan(probs) & ~np.isnan(actuals)
    probs = probs[mask]
    actuals = actuals[mask]
    
    if len(probs) == 0:
        return {"accuracy": None, "brier_score": None, "log_loss": None}
    
    # Accuracy (Home Win if Prob >= 0.5)
    preds = (probs >= 0.5).astype(float)
    accuracy = float(np.mean(preds == actuals)) * 100.0
    
    # Brier Score
    brier = float(np.mean((probs - actuals) ** 2))
    
    # Log Loss (clipped to avoid log(0) or log(1))
    clipped_probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
    logloss = float(-np.mean(actuals * np.log(clipped_probs) + (1.0 - actuals) * np.log(1.0 - clipped_probs)))
    
    return {
        "accuracy": round(accuracy, 2),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4)
    }

def run_simulation(season, initial_bankroll=1000.0, min_edge=0.03, wager_type='flat', flat_wager_pct=0.02, market_source='bookie', simulate_rest=False, upcoming_only=False, kelly_fraction=0.10, bankroll_cap=0.10):
    """
    Runs a season simulation:
    1. Loads the XGBoost model, metadata, and ready-to-use dataset.
    2. Filters for the given season.
    3. Evaluates predictions and calculates comparative metrics for the model, bookie, and Polymarket.
    4. Runs a 1,000-trial Monte Carlo simulation of standings.
    5. Simulates placing bets on games where the model has an edge >= min_edge over the selected market source.
    """
    # 1. Resolve file paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "wnba_spread_model.pkl")
    metadata_path = os.path.join(base_dir, "model_metadata.json")
    data_path = os.path.join(base_dir, "ml_ready_data.csv")
    
    # 2. Check for file existence
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}
    if not os.path.exists(metadata_path):
        return {"error": f"Metadata file not found at {metadata_path}"}
    if not os.path.exists(data_path):
        return {"error": f"Data file not found at {data_path}"}
        
    # 3. Load model and metadata
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load model or metadata: {str(e)}"}
        
    # 4. Load dataset and filter for the season
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}
        
    df_season = df[df['Season'] == season].copy()
    if df_season.empty:
        return {"error": f"No games found for season {season}"}
        
    # Sort chronologically
    df_season = df_season.sort_values(by='Date').reset_index(drop=True)
    
    # 5. Make predictions
    features_list = metadata.get('features', [])
    missing_features = [feat for feat in features_list if feat not in df_season.columns]
    if missing_features:
        return {"error": f"Data file is missing features: {missing_features}"}
        
    X_season = df_season[features_list]
    sigma_residuals = metadata.get('sigma_residuals', 10.0)
    if isinstance(model, dict) and 'regressor' in model and 'classifier' in model:
        predicted_spreads = model['regressor'].predict(X_season)
        model_probs_home = model['classifier'].predict_proba(X_season)[:, 1]
    else:
        predicted_spreads = model.predict(X_season)
        model_probs_home = norm.cdf(predicted_spreads / sigma_residuals)
    
    actual_home_wins = (df_season['HomeScore'] > df_season['AwayScore']).astype(float)
    
    # Bookie probabilities (stored in Prob_Home)
    bookie_probs_home = df_season['Prob_Home'].values
    
    # Polymarket probabilities (stored in Poly_Prob_Home, might be NaN)
    poly_probs_home = df_season['Poly_Prob_Home'].values
    
    # 6. Calculate comparative metrics
    model_metrics = compute_metrics(model_probs_home, actual_home_wins)
    bookie_metrics = compute_metrics(bookie_probs_home, actual_home_wins)
    poly_metrics = compute_metrics(poly_probs_home, actual_home_wins)
    
    # Prepare df_combined (copy of df_season with predictions added)
    df_combined = df_season.copy()
    df_combined['predicted_spread'] = predicted_spreads
    df_combined['model_prob_home'] = model_probs_home
    
    # 7. Fetch and simulate remaining unplayed games
    unplayed_games = []
    if (simulate_rest or upcoming_only) and season == 2026:
        import sys
        import sqlite3
        from datetime import datetime
        from zoneinfo import ZoneInfo
        
        app_module = sys.modules.get('app')
        if app_module is None:
            import app as app_module
            
        today_str = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
        db_path = os.path.join(base_dir, "wnba.db")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    match_date, 
                    home_team, 
                    away_team
                FROM polymarket_odds
                WHERE match_date >= ?
                ORDER BY match_date ASC
            """, (today_str,))
            poly_rows = cursor.fetchall()
            conn.close()
        except Exception as db_e:
            print(f"Failed to query upcoming bets from DB: {db_e}")
            poly_rows = []
            
        # Deduplicate upcoming games
        seen_games = set()
        upcoming_matches = []
        for match_date, home_abbr, away_abbr in poly_rows:
            game_key = (match_date, home_abbr, away_abbr)
            if game_key not in seen_games:
                seen_games.add(game_key)
                upcoming_matches.append((match_date, home_abbr, away_abbr))
                
        # Fetch live FanDuel odds
        try:
            from fanduel_odds import fetch_fanduel_odds
            fd_games = fetch_fanduel_odds()
        except Exception as fd_e:
            print("Failed to fetch live FanDuel odds in simulate_season:", fd_e)
            fd_games = []
            
        active_teams = set(df_season['HomeTeam'].unique()) | set(df_season['AwayTeam'].unique())
        
        for date_str, home_abbr, away_abbr in upcoming_matches:
            norm_home = app_module.normalize_team_name(home_abbr)
            norm_away = app_module.normalize_team_name(away_abbr)
            
            home_team = app_module.REVERSE_TEAM_MAP.get(norm_home.upper(), norm_home)
            away_team = app_module.REVERSE_TEAM_MAP.get(norm_away.upper(), norm_away)
            
            if home_team not in active_teams or away_team not in active_teams:
                continue
                
            try:
                # Always predict using model
                pred_res = app_module.make_prediction_for_matchup(home_team, away_team, date_str)
                pred_spread = pred_res['predicted_spread']
                model_prob = pred_res['home_win_probability'] / 100.0
                
                # Check for FanDuel match
                fd_match = None
                for g in fd_games:
                    g_home = g.get('home_team_full')
                    g_away = g.get('away_team_full')
                    if g_home == home_team and g_away == away_team:
                        g_date = g.get('date')
                        if g_date == date_str:
                            fd_match = g
                            break
                        else:
                            try:
                                d1 = datetime.strptime(g_date, '%Y-%m-%d').date()
                                d2 = datetime.strptime(date_str, '%Y-%m-%d').date()
                                if abs((d1 - d2).days) <= 1:
                                    fd_match = g
                                    break
                            except Exception:
                                pass
                                
                if fd_match:
                    bookie_home_odds = fd_match['home_odds']
                    bookie_away_odds = fd_match['away_odds']
                    
                    p_home_raw = 1.0 / bookie_home_odds if bookie_home_odds > 0 else 0.5
                    p_away_raw = 1.0 / bookie_away_odds if bookie_away_odds > 0 else 0.5
                    sum_p = p_home_raw + p_away_raw
                    prob_home = p_home_raw / sum_p if sum_p > 0 else 0.5
                    
                    opening_spread = fd_match.get('closing_spread', 0.0)
                    closing_spread = fd_match.get('closing_spread', 0.0)
                    over_under = fd_match.get('over_under', 160.0)
                    is_fd = 1
                else:
                    # Fallback to prediction ELO-derived odds
                    bookie_home_odds = pred_res['odds']['bookie_home_odds']
                    bookie_away_odds = pred_res['odds']['bookie_away_odds']
                    prob_home = pred_res['odds']['implied_prob_home'] / 100.0
                    opening_spread = pred_res['odds']['opening_spread']
                    closing_spread = pred_res['odds']['closing_spread']
                    over_under = pred_res['odds']['over_under']
                    is_fd = 0
                    
                # Simulate outcome
                sim_margin = np.random.normal(loc=pred_spread, scale=sigma_residuals)
                sim_home_score = int(round((over_under + sim_margin) / 2.0))
                sim_away_score = int(round(over_under - sim_home_score))
                
                if sim_margin > 0:
                    if sim_home_score <= sim_away_score:
                        sim_home_score = sim_away_score + 1
                else:
                    if sim_away_score <= sim_home_score:
                        sim_away_score = sim_home_score + 1
                        
                unplayed_games.append({
                    'Date': date_str,
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'HomeScore': sim_home_score,
                    'AwayScore': sim_away_score,
                    'Prob_Home': prob_home,
                    'Poly_Prob_Home': np.nan,
                    'BookieHomeOdds': bookie_home_odds,
                    'BookieAwayOdds': bookie_away_odds,
                    'OpeningSpread': opening_spread,
                    'ClosingSpread': closing_spread,
                    'OverUnder': over_under,
                    'predicted_spread': pred_spread,
                    'model_prob_home': model_prob,
                    'IsFanduelOdds': is_fd,
                    'Season': 2026
                })
            except Exception as pred_e:
                print(f"Prediction failed for matchup {home_team} vs {away_team} on {date_str}: {pred_e}")
                    
    if upcoming_only:
        if len(unplayed_games) == 0:
            return {"error": "No upcoming games found to simulate."}
        df_combined = pd.DataFrame(unplayed_games)
        df_combined = df_combined.sort_values(by='Date').reset_index(drop=True)
    else:
        if len(unplayed_games) > 0:
            df_unplayed = pd.DataFrame(unplayed_games)
            df_combined = pd.concat([df_combined, df_unplayed], ignore_index=True)
            # Sort chronologically by date
            df_combined = df_combined.sort_values(by='Date').reset_index(drop=True)
            
    # 8. Run Monte Carlo standings simulation (1,000 trials) using combined games
    np.random.seed(42)
    num_trials = 1000
    num_games = len(df_combined)
    
    # Get all unique teams in this season
    teams = sorted(list(set(df_combined['HomeTeam'].unique()) | set(df_combined['AwayTeam'].unique())))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    num_teams = len(teams)
    
    # Pre-allocate array to count wins for each team in each trial
    trial_wins = np.zeros((num_trials, num_teams), dtype=int)
    
    # Vectorized Monte Carlo simulation
    model_probs_home_combined = df_combined['model_prob_home'].values
    draws = np.random.rand(num_trials, num_games)
    home_wins = (draws < model_probs_home_combined).astype(int)
    
    # Map games to team indices
    home_team_indices = np.array([team_to_idx[team] for team in df_combined['HomeTeam']])
    away_team_indices = np.array([team_to_idx[team] for team in df_combined['AwayTeam']])
    
    # Accumulate wins for each team across trials
    for t in range(num_trials):
        np.add.at(trial_wins[t], home_team_indices, home_wins[t])
        np.add.at(trial_wins[t], away_team_indices, 1 - home_wins[t])
        
    # Calculate actual standings and simulated averages
    actual_wins = {team: 0 for team in teams}
    actual_losses = {team: 0 for team in teams}
    
    for _, row in df_combined.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        if row['HomeScore'] > row['AwayScore']:
            actual_wins[home] += 1
            actual_losses[away] += 1
        else:
            actual_wins[away] += 1
            actual_losses[home] += 1
            
    standings = []
    for team in teams:
        idx = team_to_idx[team]
        sim_wins_for_team = trial_wins[:, idx]
        mean_sim_wins = float(np.mean(sim_wins_for_team))
        
        # Calculate total games played by the team in this season
        total_games = df_combined[(df_combined['HomeTeam'] == team) | (df_combined['AwayTeam'] == team)].shape[0]
        mean_sim_losses = total_games - mean_sim_wins
        
        standings.append({
            "team": team,
            "actual_wins": actual_wins[team],
            "actual_losses": actual_losses[team],
            "simulated_wins": round(mean_sim_wins, 2),
            "simulated_losses": round(mean_sim_losses, 2)
        })
        
    standings = sorted(standings, key=lambda x: (-x['actual_wins'], -x['simulated_wins']))
    
    # 9. Run Betting Simulation using combined games
    bankroll = float(initial_bankroll)
    bankroll_history = [{"date": "Start", "bankroll": bankroll, "cumulative_profit": 0.0}]
    
    total_bets_count = 0
    won_bets_count = 0
    total_amount_wagered = 0.0
    
    games = []
    for i, row in df_combined.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        home_score = int(row['HomeScore'])
        away_score = int(row['AwayScore'])
        actual_winner = home if home_score > away_score else away
        
        # Resolve market probability and decimal odds
        p_home_market = None
        p_away_market = None
        odds_home = None
        odds_away = None
        
        if market_source == 'polymarket':
            poly_val = row['Poly_Prob_Home']
            if not pd.isna(poly_val):
                p_home_market = float(poly_val)
                p_away_market = 1.0 - p_home_market
                if p_home_market > 0:
                    odds_home = 1.0 / p_home_market
                if p_away_market > 0:
                    odds_away = 1.0 / p_away_market
        else: # bookie
            p_home_market = row['Prob_Home']
            if not pd.isna(p_home_market):
                p_home_market = float(p_home_market)
                p_away_market = 1.0 - p_home_market
                
                # Get actual decimal odds from table, fall back to 1 / prob
                if not pd.isna(row['BookieHomeOdds']):
                    odds_home = float(row['BookieHomeOdds'])
                elif p_home_market > 0:
                    odds_home = 1.0 / p_home_market
                    
                if not pd.isna(row['BookieAwayOdds']):
                    odds_away = float(row['BookieAwayOdds'])
                elif p_away_market > 0:
                    odds_away = 1.0 / p_away_market
                    
        # Model predictions
        model_prob_home = float(row['model_prob_home'])
        model_prob_away = 1.0 - model_prob_home
        model_predicted_winner = home if model_prob_home >= 0.5 else away
        
        # Simulate betting
        bet_placed = False
        bet_team = None
        bet_wager = 0.0
        bet_odds = None
        bet_win = None
        bet_payout = 0.0
        bet_edge = 0.0
        
        if p_home_market is not None and odds_home is not None and odds_away is not None and bankroll > 0:
            edge_home = model_prob_home - p_home_market
            edge_away = model_prob_away - p_away_market
            
            if edge_home >= min_edge:
                bet_placed = True
                bet_team = home
                bet_odds = odds_home
                bet_edge = edge_home
            elif edge_away >= min_edge:
                bet_placed = True
                bet_team = away
                bet_odds = odds_away
                bet_edge = edge_away
                
            if bet_placed:
                # Calculate wager size
                if wager_type == 'kelly':
                    p_bet = model_prob_home if bet_team == home else model_prob_away
                    if bet_odds > 1.0:
                        f_star = (p_bet * bet_odds - 1.0) / (bet_odds - 1.0)
                        kelly_frac = kelly_fraction * f_star
                        kelly_frac = max(0.0, min(bankroll_cap, kelly_frac))
                        bet_wager = kelly_frac * bankroll
                    else:
                        bet_wager = 0.0
                else: # flat
                    bet_wager = flat_wager_pct * initial_bankroll
                    
                bet_wager = round(bet_wager, 2)
                if bet_wager > bankroll:
                    bet_wager = bankroll
                    
                if bet_wager > 0:
                    # Resolve bet
                    if bet_team == actual_winner:
                        bet_win = True
                        bet_payout = round(bet_wager * (bet_odds - 1.0), 2)
                        bankroll += bet_payout
                    else:
                        bet_win = False
                        bet_payout = -bet_wager
                        bankroll += bet_payout
                        
                    bankroll = round(max(0.0, bankroll), 2)
                    total_bets_count += 1
                    if bet_win:
                        won_bets_count += 1
                    total_amount_wagered += bet_wager
                    
                    bankroll_history.append({
                        "date": str(row['Date']),
                        "bankroll": bankroll,
                        "cumulative_profit": round(bankroll - initial_bankroll, 2)
                    })
                    
        # Game log record
        games.append({
            "date": str(row['Date']),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "predicted_spread": round(float(row['predicted_spread']), 2),
            "model_prob_home": round(model_prob_home, 4),
            "bookie_prob_home": round(float(row['Prob_Home']), 4) if not pd.isna(row['Prob_Home']) else None,
            "poly_prob_home": round(float(row['Poly_Prob_Home']), 4) if not pd.isna(row['Poly_Prob_Home']) else None,
            "actual_winner": actual_winner,
            "model_predicted_winner": model_predicted_winner,
            
            # Betting details
            "bet_placed": bet_placed,
            "bet_team": bet_team,
            "bet_wager": bet_wager if bet_placed else None,
            "bet_odds": round(bet_odds, 2) if bet_placed and bet_odds is not None else None,
            "bet_win": bet_win,
            "bet_payout": bet_payout if bet_placed else None,
            "bet_edge": round(bet_edge, 4) if bet_placed else None,
            "is_fanduel_odds": int(row['IsFanduelOdds']) if 'IsFanduelOdds' in row and not pd.isna(row['IsFanduelOdds']) else 0
        })
        
    betting_metrics = {
        "initial_bankroll": round(initial_bankroll, 2),
        "final_bankroll": round(bankroll, 2),
        "total_bets": total_bets_count,
        "won_bets": won_bets_count,
        "win_rate": round((won_bets_count / total_bets_count * 100.0), 2) if total_bets_count > 0 else 0.0,
        "total_wagered": round(total_amount_wagered, 2),
        "net_profit": round(bankroll - initial_bankroll, 2),
        "roi": round((bankroll - initial_bankroll) / total_amount_wagered * 100.0, 2) if total_amount_wagered > 0 else 0.0
    }
    
    return {
        "metrics": {
            "model": model_metrics,
            "bookie": bookie_metrics,
            "polymarket": poly_metrics
        },
        "standings": standings,
        "games": games,
        "betting_metrics": betting_metrics,
        "bankroll_history": bankroll_history
    }
