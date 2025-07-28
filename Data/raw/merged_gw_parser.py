import csv
import io
import pandas as pd
import os

def load_team_strengths(teams_file_path: str) -> dict:
    """
    Load team strength data from CSV file.
    
    Returns:
        Dictionary mapping team IDs to their strength metrics
    """
    teams_df = pd.read_csv(teams_file_path)
    
    team_strengths = {}
    for _, row in teams_df.iterrows():
        team_id = str(row['id'])  # Convert to string for consistent lookup
        team_strengths[team_id] = {
            'name': row['name'],
            'attack_home': row['strength_attack_home'],
            'attack_away': row['strength_attack_away'],
            'defence_home': row['strength_defence_home'],
            'defence_away': row['strength_defence_away'],
            'overall_home': row['strength_overall_home'],
            'overall_away': row['strength_overall_away']
        }
    
    return team_strengths

def calculate_opponent_strength(opponent_team_id: str, is_home: bool, team_strengths: dict) -> dict:
    """
    Calculate opponent strength metrics based on opponent team ID and home/away status.
    
    Args:
        opponent_team_id: ID of the opponent team (as string)
        is_home: Whether the current team is playing at home (True) or away (False)
        team_strengths: Dictionary of team strength data
    
    Returns:
        Dictionary with opponent strength metrics
    """
    if str(opponent_team_id) not in team_strengths:
        # Default values if team not found
        return {
            'opponent_attack_strength': 1150.0,
            'opponent_defence_strength': 1150.0,
            'opponent_overall_strength': 1150.0,
            'fixture_attractiveness': 0.5
        }
    
    opponent_data = team_strengths[str(opponent_team_id)]
    
    # If current team is home, opponent is away (and vice versa)
    if is_home:
        # Current team at home, opponent away
        attack_strength = opponent_data['attack_away']
        defence_strength = opponent_data['defence_away']
        overall_strength = opponent_data['overall_away']
    else:
        # Current team away, opponent at home  
        attack_strength = opponent_data['attack_home']
        defence_strength = opponent_data['defence_home']
        overall_strength = opponent_data['overall_home']
    
    # Calculate fixture attractiveness (normalized opponent strength)
    # Higher opponent strength = lower attractiveness for scoring
    fixture_attractiveness = max(0, min(1, (1400 - overall_strength) / 400))
    
    return {
        'opponent_attack_strength': float(attack_strength),
        'opponent_defence_strength': float(defence_strength),
        'opponent_overall_strength': float(overall_strength),
        'fixture_attractiveness': round(fixture_attractiveness, 3)
    }

def load_player_codes(players_file_path: str) -> dict:
    """
    Load player code mapping from CSV file.
    
    Returns:
        Dictionary mapping element IDs to player codes for cross-season consistency
    """
    players_df = pd.read_csv(players_file_path)
    
    player_codes = {}
    for _, row in players_df.iterrows():
        element_id = str(row['id'])  # This is the element ID from historical data
        code = str(row['code'])      # This is the consistent cross-season identifier
        player_codes[element_id] = {
            'code': code,
            'name': row['web_name'],
            'full_name': f"{row['first_name']} {row['second_name']}"
        }
    
    return player_codes

def process_csv_file(file_path: str) -> str:
    """
    Manipulates a CSV file to keep specified columns and the last column of each row.
    Now also adds player code information for cross-season consistency.

    Args:
        file_path: Path to the input CSV file.

    Returns:
        A string containing the manipulated CSV data.
    """
    output_buffer = io.StringIO()
    csv_writer = csv.writer(output_buffer)

    # Load team strength data
    teams_file = os.path.join(os.path.dirname(file_path), 'teams2425.csv')
    try:
        team_strengths = load_team_strengths(teams_file)
        print(f"✅ Loaded team strengths for {len(team_strengths)} teams")
    except Exception as e:
        print(f"⚠️ Warning: Could not load team strengths from {teams_file}: {e}")
        print("   Using default opponent strength values")
        team_strengths = {}

    # Load player code mapping
    players_file = os.path.join(os.path.dirname(file_path), 'players_raw2425.csv')
    try:
        player_codes = load_player_codes(players_file)
        print(f"✅ Loaded player codes for {len(player_codes)} players")
    except Exception as e:
        print(f"⚠️ Warning: Could not load player codes from {players_file}: {e}")
        print("   Player code information will not be available")
        player_codes = {}

    # Define the columns to keep by name (prefix columns)
    # These columns are selected for their relevance to player performance prediction
    columns_to_keep_prefix = [
        # Player identification (now includes code for cross-season consistency)
        'name', 'position', 'team', 'element', 'player_code',
        
        # Performance metrics (target and features)
        'total_points', 'xP', 
        
        # Goal-related stats
        'goals_scored', 'expected_goals', 'assists', 'expected_assists',
        'expected_goal_involvements',
        
        # Defensive stats
        'clean_sheets', 'goals_conceded', 'expected_goals_conceded', 'saves',
        
        # Advanced performance metrics
        'bonus', 'bps', 'ict_index', 'influence', 'creativity', 'threat',
        
        # Playing time and form
        'minutes', 'starts', 'selected',
        
        # Disciplinary
        'yellow_cards', 'red_cards', 'own_goals',
        
        # Penalties
        'penalties_missed', 'penalties_saved',
        
        # Market metrics (player popularity/form indicators)
        'transfers_in', 'transfers_out', 'transfers_balance', 'value',
        
        # Match context
        'opponent_team', 'was_home', 'team_a_score', 'team_h_score', 'fixture',
        
        # Opponent strength features (calculated from team strength data)
        'opponent_attack_strength', 'opponent_defence_strength', 
        'opponent_overall_strength', 'fixture_attractiveness'
    ]

    try:
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as infile:
            csv_reader = csv.reader(infile)
            
            original_header = next(csv_reader, None)
            
            if original_header is None:
                raise ValueError("Input CSV file is empty or lacks a header.")

            if not original_header: # Defensive check
                raise ValueError("Original header row is an empty list, which is invalid.")

            # Determine the name for the "last GW column" from the original header
            name_for_last_gw_column = original_header[-1]

            # Prepare the new header for the output
            new_header = columns_to_keep_prefix + [name_for_last_gw_column]
            csv_writer.writerow(new_header)

            # Get indices for opponent strength calculation
            opponent_team_idx = original_header.index('opponent_team')
            was_home_idx = original_header.index('was_home')
            element_idx = original_header.index('element')

            # Process each data row
            for row_data in csv_reader:
                if not row_data:  # Skip empty rows
                    continue
                    
                output_row_values = []
                
                # Build the row according to the new header structure
                for col_name in columns_to_keep_prefix:
                    if col_name == 'player_code':
                        # Add player code based on element ID lookup
                        try:
                            element_id = row_data[element_idx] if element_idx < len(row_data) else ''
                            player_info = player_codes.get(str(element_id), {})
                            player_code = player_info.get('code', '')
                            output_row_values.append(player_code)
                        except Exception as e:
                            print(f"⚠️ Warning: Could not get player code for element {element_id}: {e}")
                            output_row_values.append('')
                    elif col_name in ['opponent_attack_strength', 'opponent_defence_strength', 
                                    'opponent_overall_strength', 'fixture_attractiveness']:
                        # These will be added later in the opponent strength calculation
                        continue
                    else:
                        # Regular column from original data
                        try:
                            col_idx = original_header.index(col_name)
                            if col_idx < len(row_data):
                                output_row_values.append(row_data[col_idx])
                            else:
                                output_row_values.append('')
                        except ValueError:
                            # Column not found in original data
                            output_row_values.append('')
                
                # Calculate opponent strength features
                try:
                    opponent_team_id = row_data[opponent_team_idx] if opponent_team_idx < len(row_data) else ''
                    was_home = row_data[was_home_idx] if was_home_idx < len(row_data) else 'True'
                    is_home = was_home.lower() == 'true'
                    
                    opponent_strength = calculate_opponent_strength(opponent_team_id, is_home, team_strengths)
                    
                    # Add opponent strength values to output
                    output_row_values.extend([
                        opponent_strength['opponent_attack_strength'],
                        opponent_strength['opponent_defence_strength'],
                        opponent_strength['opponent_overall_strength'],
                        opponent_strength['fixture_attractiveness']
                    ])
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not calculate opponent strength for row: {e}")
                    # Add default values
                    output_row_values.extend([1150.0, 1150.0, 1150.0, 0.5])
                
                # Extract data for the "last GW column" (actual last element of the current row)
                if row_data: # Check if row_data is not empty
                    output_row_values.append(row_data[-1])
                else:
                    # This should not be reached if empty rows are skipped above
                    output_row_values.append('') 
                
                csv_writer.writerow(output_row_values)
        
        return output_buffer.getvalue()

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    except ValueError as ve:
        raise ValueError(f"CSV processing error: {str(ve)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during CSV manipulation: {str(e)}")

# The user uploaded 'merged_gw_2425.csv'.
# In a real environment, 'files[0]' would be the path provided by the system.
# For this execution, I will use the placeholder name.
file_id = "/Users/owen/src/Personal/fpl-team-picker/Data/raw/merged_gw_2425.csv" 

try:
    # Process the CSV
    manipulated_csv_data = process_csv_file(file_id)
    
    # Define the output filename
    output_filename = "/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw_2425.csv"
    
    # In a typical environment, this string data would be written to a file
    # that can then be offered for download.
    # For now, I will confirm success and provide the name of the intended output file.
    # If running locally, you'd uncomment the next lines:
    # with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
    #     f_out.write(manipulated_csv_data)
    
    print(f"Successfully processed the CSV file.")
    print(f"The manipulated data is ready and would be saved as '{output_filename}'.")
    # To display a snippet (optional and only if small):
    # print("\nFirst few lines of the manipulated CSV:\n")
    # print('\n'.join(manipulated_csv_data.splitlines()[:5]))

    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
            f_out.write(manipulated_csv_data)
        print(f"Successfully saved the manipulated data to '{output_filename}'")
    except IOError:
        print(f"Error: Could not write to the file '{output_filename}'. Please check permissions or disk space.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")


except Exception as e:
    print(f"An error occurred: {e}")