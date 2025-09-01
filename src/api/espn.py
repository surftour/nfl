import json

import requests
import pandas as pd

from ..utils.data_io import save_to_csv

def fetch_teams():
    url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"

    try:
        response = requests.get(url, timeout=10)
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
        response = requests.get(url, timeout=10)
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

def _extract_team_info(competitors, home_away):
    """Helper function to extract team information from competitors"""
    team = next((t for t in competitors if t.get('homeAway') == home_away), {})
    if not team or 'team' not in team:
        return {}

    prefix = f"{home_away}_team"
    return {
        f'{prefix}_id': team['team'].get('id', ''),
        f'{prefix}_name': team['team'].get('displayName', ''),
        f'{prefix}_abbr': team['team'].get('abbreviation', ''),
        f'{prefix}_score': team.get('score', 0),
        f'{prefix}_winner': team.get('winner', False)
    }

def _extract_venue_info(competition):
    """Helper function to extract venue information"""
    if 'venue' not in competition:
        return {}

    venue = competition['venue']
    return {
        'venue_name': venue.get('fullName', ''),
        'venue_city': venue.get('address', {}).get('city', ''),
        'venue_state': venue.get('address', {}).get('state', '')
    }

def fetch_game_results(year, week=None, season_type=2):
    """
    Fetch game results for a specific year and optionally a specific week

    Args:
        year (int): Year (e.g., 2024)
        week (int, optional): Week number (1-18 for regular season). If None, fetches all weeks.
        season_type (int): Season type (1=preseason, 2=regular season, 3=postseason)

    Returns:
        pandas.DataFrame with game results
    """
    base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {'seasontype': season_type}

    if week:
        params['week'] = week

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        games_data = []

        for event in data.get('events', []):
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

            # Extract team and venue information
            competitions = event.get('competitions', [])
            if competitions:
                competition = competitions[0]
                competitors = competition.get('competitors', [])

                # Add team information
                game_info.update(_extract_team_info(competitors, 'home'))
                game_info.update(_extract_team_info(competitors, 'away'))

                # Add venue information
                game_info.update(_extract_venue_info(competition))

            games_data.append(game_info)

        return pd.DataFrame(games_data)

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
