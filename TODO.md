# FPL Team Picker - TODO List

## MVP:
- [ ] Figure out who we don't have predictions for, by price desc
  - [ ] Group by league in previous season & tackle like that
  - [ ] Integrate FBref - should we use that rather than FPL?
- [ ] Make UI nice
- [ ] Defensive contributions - FBref?
- [ ] AI chatbot
  - [ ] Tool for getting my predictions
  - [ ] Tool for getting my info
  - [ ] Tool for getting the optimal wildcard team
- [ ] External prediction integration
  - [ ] Create a web scraping tool from popular FPL blogs
  - [ ] Aggregate it all and put it in a document store


## 🎯 **NEXT TO INVESTIGATE**

### Model Validation
- [ ] Why are predictions consistent across GWs?
- [ ] Should we estimate expected defensive interactions? We might want to pull from a different data store here
- [ ] Potential: Looking at season total minutes as a threshold for minutes predictions, should we not use % of minutes played in the last few gameweeks?
- [ ] Learning curve for existing models? Do we have enough training data?
- [ ] Should we be using `form`?
- [ ] Put players we don't have predictions for into buckets and tackle them

### Model Calibration Issues (From Prediction vs Reality Analysis)
- [ ] **Clean sheet probability validation** - A few defenders have >40% CS probability (Calafiori 50.3%, Fofana 49.3%) - check team defense model
- [ ] **Elite player identification** - Under-predicting top performers (Salah: pred 6.0 vs actual 9.1, Isak: pred 3.8 vs actual 6.2)
- [ ] Why, when predicting, do we do 43 games, then 5 games for a player. 43 seems too high.

### OR Tools model
- [ ] Penalise:
  - [ ] players selected from the same team in attacking positions
  - [ ] players playing each other in the same gameweek (e.g don't have a defender of a team if you have a mid/fwd playing them)
- [ ] Change the JSON files to a proper data store, and hook the API into it

### DB
- [ ] include more team information in teams.json, including difficulty score?
