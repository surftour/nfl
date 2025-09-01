import requests
import pandas as pd
import json
from ..utils.data_io import save_to_csv

def fetch_teams():
    url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        teams_data = []
        
        if 'sports' in data and len(data['sports']) > 0:
            leagues = data['sports'][0].get('leagues', [])
            if leagues and 'teams' in leagues[0]:
                for team in leagues[0]['teams']:
                    team_info = team.get('team', {})
                    team_entry = {
                        'id': team_info.get('id', ''),
                        'uid': team_info.get('uid', ''),
                        'slug': team_info.get('slug', ''),
                        'abbreviation': team_info.get('abbreviation', ''),
                        'displayName': team_info.get('displayName', ''),
                        'shortDisplayName': team_info.get('shortDisplayName', ''),
                        'name': team_info.get('name', ''),
                        'nickname': team_info.get('nickname', ''),
                        'location': team_info.get('location', ''),
                        'color': team_info.get('color', ''),
                        'alternateColor': team_info.get('alternateColor', ''),
                        'isActive': team_info.get('isActive', False),
                        'isAllStar': team_info.get('isAllStar', False)
                    }
                    
                    # Add logo URLs if available
                    if 'logos' in team_info and team_info['logos']:
                        team_entry['logo_url'] = team_info['logos'][0].get('href', '')
                    else:
                        team_entry['logo_url'] = ''
                    
                    teams_data.append(team_entry)
        
        df = pd.DataFrame(teams_data)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching teams data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing teams JSON: {e}")
        return None

def fetch_team_stats(year, team_id):
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/types/2/teams/{team_id}/statistics"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        stats_data = []
        
        if 'splits' in data and 'categories' in data['splits']:
            for category in data['splits']['categories']:
                category_name = category.get('name', 'Unknown')
                
                if 'stats' in category:
                    for stat in category['stats']:
                        stat_entry = {
                            'category': category_name,
                            'name': stat.get('name', ''),
                            'displayName': stat.get('displayName', ''),
                            'shortDisplayName': stat.get('shortDisplayName', ''),
                            'description': stat.get('description', ''),
                            'value': stat.get('value', 0),
                            'displayValue': stat.get('displayValue', '')
                        }
                        stats_data.append(stat_entry)
        
        df = pd.DataFrame(stats_data)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def fetch_game_results(year, week=None, season_type=2):
    base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    
    params = {
        'seasontype': season_type
    }
    
    if week:
        params['week'] = week
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        games_data = []
        
        if 'events' in data:
            for event in data['events']:
                game_info = {
                    'id': event.get('id', ''),
                    'uid': event.get('uid', ''),
                    'date': event.get('date', ''),
                    'name': event.get('name', ''),
                    'shortName': event.get('shortName', ''),
                    'season': event.get('season', {}).get('year', year),
                    'week': event.get('week', {}).get('number', ''),
                    'season_type': event.get('season', {}).get('type', season_type),
                    'status': event.get('status', {}).get('type', {}).get('name', ''),
                    'completed': event.get('status', {}).get('type', {}).get('completed', False)
                }
                
                # Extract team information and scores
                if 'competitions' in event and event['competitions']:
                    competition = event['competitions'][0]
                    
                    if 'competitors' in competition:
                        competitors = competition['competitors']
                        
                        home_team = next((team for team in competitors if team.get('homeAway') == 'home'), {})
                        away_team = next((team for team in competitors if team.get('homeAway') == 'away'), {})
                        
                        if home_team and 'team' in home_team:
                            game_info.update({
                                'home_team_id': home_team['team'].get('id', ''),
                                'home_team_name': home_team['team'].get('displayName', ''),
                                'home_team_abbr': home_team['team'].get('abbreviation', ''),
                                'home_team_score': home_team.get('score', 0),
                                'home_team_winner': home_team.get('winner', False)
                            })
                        
                        if away_team and 'team' in away_team:
                            game_info.update({
                                'away_team_id': away_team['team'].get('id', ''),
                                'away_team_name': away_team['team'].get('displayName', ''),
                                'away_team_abbr': away_team['team'].get('abbreviation', ''),
                                'away_team_score': away_team.get('score', 0),
                                'away_team_winner': away_team.get('winner', False)
                            })
                    
                    # Add venue information
                    if 'venue' in competition:
                        venue = competition['venue']
                        game_info.update({
                            'venue_name': venue.get('fullName', ''),
                            'venue_city': venue.get('address', {}).get('city', ''),
                            'venue_state': venue.get('address', {}).get('state', '')
                        })
                
                games_data.append(game_info)
        
        df = pd.DataFrame(games_data)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game results: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing game results JSON: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    teams_df = save_to_csv(fetch_teams, 'data/nfl_teams.csv')
    stats_df = save_to_csv(fetch_team_stats, 'data/team_stats.csv', 2024, 25)
    games_df = save_to_csv(fetch_game_results, 'data/game_results_week1.csv', 2024, 1)