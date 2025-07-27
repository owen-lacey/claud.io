# Data Pipeline Notebooks 📊

## Overview
Core data extraction and processing notebooks that feed the modeling pipeline.

## Files
- `extract.ipynb` - **Main data extraction pipeline**
  - Extracts `players.json`, `teams.json`, `fixtures.json` from FPL API
  - Simplified from complex analysis to focus on essential data
  - **Status**: ✅ Complete and streamlined

## Usage
Run `extract.ipynb` to refresh the JSON data files in the `database/` directory with the latest FPL API data. This should be run regularly to keep predictions current.

## Dependencies
- Requires internet connection for FPL API calls
- Outputs to `../database/` directory
- Uses minimal dependencies (requests, pandas, json)
