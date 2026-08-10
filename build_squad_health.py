import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from data_processing import TEAM_NAME_MAP

ALL_TEAMS = [
    "Atlanta Dream", "Chicago Sky", "Connecticut Sun", "Dallas Wings",
    "Golden State Valkyries", "Indiana Fever", "Los Angeles Sparks",
    "Las Vegas Aces", "Minnesota Lynx", "New York Liberty",
    "Portland Fire", "Phoenix Mercury", "Seattle Storm",
    "Toronto Tempo", "Washington Mystics"
]

def clean_team_name(team_str):
    name = team_str.strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Check substring matches
    for key, val in TEAM_NAME_MAP.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    return name

def scrape_injuries():
    url = "https://www.espn.com/wnba/injuries"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    print(f"Scraping injuries from {url}...")
    injuries = []
    try:
        r = requests.get(url, headers=headers, verify=False, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        for table in soup.find_all('table'):
            heading = table.find_previous(class_='Table__Title')
            if not heading:
                heading = table.find_previous(['h4', 'h3', 'h2', 'h1'])
            team_name = heading.text.strip() if heading else 'Unknown'
            canonical_team = clean_team_name(team_name)
            
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if not cols:
                    continue
                player_name = cols[0].text.strip()
                status = cols[3].text.strip() if len(cols) > 3 else "Out"
                
                injuries.append({
                    'Team': canonical_team,
                    'Player': player_name,
                    'Status': status
                })
                
        print(f"Successfully scraped {len(injuries)} active injuries.")
    except Exception as e:
        print(f"Warning: Live injury scraping failed ({e}). Falling back to database injuries table...")
        try:
            conn = sqlite3.connect("wnba.db")
            cursor = conn.cursor()
            cursor.execute("SELECT Team, Player, InjuryStatus FROM injuries")
            for row in cursor.fetchall():
                injuries.append({
                    'Team': clean_team_name(row[0]),
                    'Player': row[1],
                    'Status': row[2]
                })
            conn.close()
            print(f"Loaded {len(injuries)} injury records from database.")
        except Exception as db_err:
            print(f"Warning: Failed to load backup injuries from database: {db_err}")

    return injuries

def build_squad_health():
    # 1. Scrape current injuries
    injuries = scrape_injuries()
    
    # 2. Query player stats from the database
    conn = sqlite3.connect("wnba.db")
    cursor = conn.cursor()
    
    # Initialize team metrics dictionary
    team_metrics = {team: {
        'Missing_Usage_Pct': 0.0,
        'Missing_Net_Rating': 0.0,
        'Missing_PIE': 0.0,
        'Missing_Minutes_Pct': 0.0,
        'Injured_Players_Count': 0
    } for team in ALL_TEAMS}
    
    for injury in injuries:
        team = injury['Team']
        player = injury['Player']
        
        # Ensure team is in our list of tracked teams
        if team not in team_metrics:
            print(f"Warning: Team {team} is not in the tracked teams list. Adding it dynamically.")
            team_metrics[team] = {
                'Missing_Usage_Pct': 0.0,
                'Missing_Net_Rating': 0.0,
                'Missing_PIE': 0.0,
                'Missing_Minutes_Pct': 0.0,
                'Injured_Players_Count': 0
            }
            
        # Query player stats (most recent season)
        cursor.execute("""
            SELECT MIN, USG_PCT, NET_RATING, PIE, Season, GP
            FROM player_stats
            WHERE Player = ?
            ORDER BY Season DESC
            LIMIT 1
        """, (player,))
        row = cursor.fetchone()
        
        if row:
            min_avg = row[0]
            usg_pct = row[1]
            net_rating = row[2]
            pie = row[3]
            season = row[4]
            gp = row[5]
            print(f"Found stats for {player} (Season: {season}, Team: {team}): MIN={min_avg}, USG%={usg_pct}, NET_RATING={net_rating}, PIE={pie}")
        else:
            print(f"Warning: No stats found in database for {player}. Using default 0.0 values.")
            min_avg = 0.0
            usg_pct = 0.0
            net_rating = 0.0
            pie = 0.0
            gp = 0
            
        # Calculate metric contributions
        # USG_PCT is stored as a fraction (e.g., 0.289 = 28.9%). Missing_Usage_Pct is sum of USG%
        player_usg = usg_pct * 100.0
        # Minutes-weighted NET_RATING and PIE
        player_net_rating_weighted = min_avg * net_rating
        player_pie_weighted = min_avg * pie
        player_min_pct = (min_avg / 200.0) * 100.0
        
        team_metrics[team]['Missing_Usage_Pct'] += player_usg
        team_metrics[team]['Missing_Net_Rating'] += player_net_rating_weighted
        team_metrics[team]['Missing_PIE'] += player_pie_weighted
        team_metrics[team]['Missing_Minutes_Pct'] += player_min_pct
        team_metrics[team]['Injured_Players_Count'] += 1
        
    conn.close()
    
    # 3. Format as DataFrame and export
    records = []
    for team, metrics in team_metrics.items():
        records.append({
            'Team': team,
            'Missing_Usage_Pct': round(metrics['Missing_Usage_Pct'], 3),
            'Missing_Net_Rating': round(metrics['Missing_Net_Rating'], 3),
            'Missing_PIE': round(metrics['Missing_PIE'], 3),
            'Missing_Minutes_Pct': round(metrics['Missing_Minutes_Pct'], 3),
            'Injured_Players_Count': metrics['Injured_Players_Count']
        })
        
    df_health = pd.DataFrame(records)
    # Sort teams alphabetically
    df_health = df_health.sort_values(by='Team').reset_index(drop=True)
    
    output_file = "current_squad_health.csv"
    df_health.to_csv(output_file, index=False)
    print(f"Squad health summary exported successfully to {output_file}.")
    print(df_health)

if __name__ == "__main__":
    build_squad_health()
