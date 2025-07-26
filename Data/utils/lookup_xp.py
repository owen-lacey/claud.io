def lookup_fpl_data(element_id, gameweek, xpData):
  # Ensure 'element' and 'GW' columns are of appropriate data type for comparison
  xpData['element'] = xpData['element'].astype(int)
  xpData['GW'] = xpData['GW'].astype(int)

  # Look up the row based on the criteria
  # Using boolean indexing to filter the DataFrame
  matching_rows = xpData[(xpData['element'] == element_id) & (xpData['GW'] == gameweek)]

  if not matching_rows.empty:
      # If there are matching rows, return the first one (assuming element-gw combination is unique)
      return matching_rows.iloc[0]
  else:
      return None # No matching row found