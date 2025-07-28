# FPL Team Picker - TODO List

## 🎯 **NEXT TO INVESTIGATE**

### Model Validation
- [x] ~~Why are MID/FWDs predicted lower than others?~~ **FIXED:** Double clean sheet calculation bug was inflating GK/DEF predictions by ~2pts
- [ ] Why are predictions consistent across GWs?
- [x] ~~Why is Travers so popular?~~ **IDENTIFIED:** Name matching failure between "Travers" (db) vs "Mark Travers" (historical) causes fallback to season averages instead of recent 0-minute form
- [x] **CRITICAL: Fix historical data matching** - Use ID mapping instead of names (historical.element ≠ players.id, e.g., Mo Salah: 328→381)
- [ ] Can we calculated expected goals conceded instead?
- [ ] Should we estimate expected defensive interactions? We might want to pull from a different data store here
- [x] ~~Bonus points - are we doing it? Do we want to enable always?~~ **CONFIRMED:** Bonus points are implemented and being calculated (include_bonus=True in prediction engine)
- [ ] Potential: Looking at season total minutes as a threshold for minutes predictions, should we not use % of minutes played in the last few gameweeks?
- [ ] Defensive coding everywhere! I would much rather fail obviously from a data issue. A good example of this is default parameter values.
- [ ] Bring in data from other leagues if we can!
- [x] ~~Quite a lot of defenders are predicted to score highly?~~ **FIXED:** Implemented Poisson-based goal conceded penalty system - now +1.24 pts over-prediction vs previous +1.58 pts
- [ ] Learning curve for existing models? Do we have enough training data?
- [ ] Should we be using form?

### Model Calibration Issues (From Prediction vs Reality Analysis)
- [x] ~~**CRITICAL: Implement negative penalties**~~ **COMPLETED:** Implemented Poisson-based goal conceded penalties (2-3 goals: -1pt, 4-5: -2pts, 6-7: -3pts, 8-9: -4pts) with proper non-overlapping buckets
- [x] ~~**Defender goal expectations too high**~~ **COMPLETED:** Retrained expected goals model to include defenders with position-aware features (is_defender: -0.028 coefficient penalty, DEF: 0.067 vs 0.067 actual goals in test set)
- [ ] **Clean sheet probability validation** - A few defenders have >40% CS probability (Calafiori 50.3%, Fofana 49.3%) - check team defense model
- [ ] **Elite player identification** - Under-predicting top performers (Salah: pred 6.0 vs actual 9.1, Isak: pred 3.8 vs actual 6.2)
- [x] ~~**Playing time estimation for young/squad players**~~ **PARTIALLY IMPROVED:** Still over-predicting rotation players (Lewis-Skelly: pred 4.7 vs actual 1.6) but penalty system now differentiates strong vs weak defenses (+1.08 pt separation)
