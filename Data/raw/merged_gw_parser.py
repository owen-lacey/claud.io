import csv
import io

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

    # Define the columns to keep by name (prefix columns)
    columns_to_keep_prefix = [
        'name', 'position', 'team', 'xP', 'assists', 'bonus', 'bps',
        'clean_sheets', 'creativity', 'element'
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

            # Get indices of the prefix columns from the original header
            prefix_column_indices = []
            missing_columns_in_header = []
            for col_name in columns_to_keep_prefix:
                try:
                    prefix_column_indices.append(original_header.index(col_name))
                except ValueError:
                    missing_columns_in_header.append(col_name)
            
            if missing_columns_in_header:
                raise ValueError(
                    f"The following specified columns were not found in the CSV header: {', '.join(missing_columns_in_header)}"
                )

            # Process each data row
            for row_data in csv_reader:
                if not row_data:  # Skip empty rows
                    continue
                    
                output_row_values = []
                # Extract data for the prefix columns
                for col_idx in prefix_column_indices:
                    if col_idx < len(row_data):
                        output_row_values.append(row_data[col_idx])
                    else:
                        # Row is shorter than expected for a prefix column
                        output_row_values.append('') # Placeholder for missing data
                
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