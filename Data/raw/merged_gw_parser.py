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

def process_csv_file(file_path: str) -> str:
    """
    Manipulates a CSV file to keep specified columns and the last column of each row.

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

    # Define the columns to keep by name (prefix columns)
    # These columns are selected for their relevance to player performance prediction
    columns_to_keep_prefix = [
        # Player identification
        'name', 'position', 'team', 'element',
        
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

            # Get indices of the base columns (excluding opponent strength columns)
            base_columns = [col for col in columns_to_keep_prefix 
                          if col not in ['opponent_attack_strength', 'opponent_defence_strength', 
                                       'opponent_overall_strength', 'fixture_attractiveness']]
            
            prefix_column_indices = []
            missing_columns_in_header = []
            for col_name in base_columns:
                try:
                    prefix_column_indices.append(original_header.index(col_name))
                except ValueError:
                    missing_columns_in_header.append(col_name)
            
            if missing_columns_in_header:
                raise ValueError(
                    f"The following specified columns were not found in the CSV header: {', '.join(missing_columns_in_header)}"
                )
            
            # Get indices for opponent strength calculation
            opponent_team_idx = original_header.index('opponent_team')
            was_home_idx = original_header.index('was_home')

            # Process each data row
            for row_data in csv_reader:
                if not row_data:  # Skip empty rows
                    continue
                    
                output_row_values = []
                # Extract data for the base prefix columns
                for col_idx in prefix_column_indices:
                    if col_idx < len(row_data):
                        output_row_values.append(row_data[col_idx])
                    else:
                        # Row is shorter than expected for a prefix column
                        output_row_values.append('') # Placeholder for missing data
                
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

# The user uploaded 'merged_gw.csv'.
# In a real environment, 'files[0]' would be the path provided by the system.
# For this execution, I will use the placeholder name.
file_id = "/Users/owen/src/Personal/fpl-team-picker/Data/raw/merged_gw.csv" 

try:
    # Process the CSV
    manipulated_csv_data = process_csv_file(file_id)
    
    # Define the output filename
    output_filename = "/Users/owen/src/Personal/fpl-team-picker/Data/raw/parsed_gw.csv"
    
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