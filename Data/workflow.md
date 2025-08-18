Every gameweek:
1. Run `extract.ipynb` to update current player info such as cost / availability
2. Update FBRef DB to get recent player performance info
   1. (NB: happens outside of )
3. Update team strengths
   1. `.venv/bin/python Data/update_team_strengths.py --gameweek XXX`
4. Create GW fixtures
5. Run `generate_predictions.py` for the GW