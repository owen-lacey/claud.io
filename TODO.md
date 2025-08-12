# FPL Team Picker - TODO List

## MVP:
- [x] Figure out who we don't have predictions for, by price desc
  - [x] Group by league in previous season & tackle like that
  - [x] Integrate FBref - should we use that rather than FPL?
- [x] Make UI nice
- [x] Defensive contributions - FBref?
- [x] AI chatbot
  - [x] Tool for getting my predictions
  - [x] Tool for getting my info
  - [x] Tool for getting the optimal wildcard team
- [ ] External prediction integration
  - [ ] Create a web scraping tool from popular FPL blogs
  - [ ] Aggregate it all and put it in a document store
- [x] Change the JSON files to a proper data store, and hook the API into it

### OR Tools model
- [ ] Penalise:
  - [ ] players selected from the same team in attacking positions
  - [ ] players playing each other in the same gameweek (e.g don't have a defender of a team if you have a mid/fwd playing them)


### DB
- [x] include more team information in teams.json, including difficulty score?
- [ ] Save strength info to teams so we can show it in the UI
