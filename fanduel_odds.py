import requests
import hashlib
import random

AN_TO_CANONICAL_ABBR = {
    'GS': 'GSV',
    'LVA': 'LVA',
    'WSH': 'WAS',
    'MIN': 'MIN',
    'NY': 'NYL',
    'LA': 'LAS',
    'IND': 'IND',
    'CHI': 'CHI',
    'PHX': 'PHX',
    'PHO': 'PHX',
    'DAL': 'DAL',
    'ATL': 'ATL',
    'CON': 'CON',
    'SEA': 'SEA'
}

CANONICAL_ABBR_TO_FULL = {
    'IND': 'Indiana Fever',
    'CHI': 'Chicago Sky',
    'LVA': 'Las Vegas Aces',
    'NYL': 'New York Liberty',
    'SEA': 'Seattle Storm',
    'MIN': 'Minnesota Lynx',
    'PHX': 'Phoenix Mercury',
    'DAL': 'Dallas Wings',
    'ATL': 'Atlanta Dream',
    'CON': 'Connecticut Sun',
    'LAS': 'Los Angeles Sparks',
    'WAS': 'Washington Mystics',
    'GSV': 'Golden State Valkyries'
}

def american_to_decimal(american_odds):
    if american_odds is None:
        return 1.90
    if american_odds > 0:
        return round(1.0 + (american_odds / 100.0), 2)
    elif american_odds < 0:
        return round(1.0 - (100.0 / american_odds), 2)
    return 1.90

def fetch_fanduel_odds():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    try:
        r = requests.get('https://api.actionnetwork.com/web/v1/scoreboard/wnba', headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        games = data.get('games', [])
        parsed_games = []
        for g in games:
            home_id = g.get('home_team_id')
            away_id = g.get('away_team_id')
            home_team_abbr = None
            away_team_abbr = None
            for t in g.get('teams', []):
                if t.get('id') == home_id:
                    home_team_abbr = AN_TO_CANONICAL_ABBR.get(t.get('abbr'))
                elif t.get('id') == away_id:
                    away_team_abbr = AN_TO_CANONICAL_ABBR.get(t.get('abbr'))
            
            if not home_team_abbr or not away_team_abbr:
                continue
                
            home_team_full = CANONICAL_ABBR_TO_FULL.get(home_team_abbr)
            away_team_full = CANONICAL_ABBR_TO_FULL.get(away_team_abbr)
            
            # Find FanDuel (book_id = 30) pre-match game odds
            fd_odds = None
            for o in g.get('odds', []):
                if o.get('book_id') == 30 and o.get('type') == 'game':
                    fd_odds = o
                    break
            
            if fd_odds:
                ml_home = fd_odds.get('ml_home')
                ml_away = fd_odds.get('ml_away')
                spread_home = fd_odds.get('spread_home')
                total = fd_odds.get('total')
                
                # Convert date from start_time
                start_time = g.get('start_time')
                match_date = start_time.split('T')[0] if start_time else None
                
                if ml_home is not None and ml_away is not None:
                    home_odds = american_to_decimal(ml_home)
                    away_odds = american_to_decimal(ml_away)
                    parsed_games.append({
                        'date': match_date,
                        'home_team_full': home_team_full,
                        'away_team_full': away_team_full,
                        'home_team_abbr': home_team_abbr,
                        'away_team_abbr': away_team_abbr,
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'closing_spread': spread_home,
                        'over_under': total
                    })
        return parsed_games
    except Exception as e:
        print('Error fetching FanDuel odds:', e)
        return []
