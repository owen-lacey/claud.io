# FPL Team Picker - TODO List

## 🎯 **NEXT TO INVESTIGATE**

### Model Validation
- [x] ~~Why are MID/FWDs predicted lower than others?~~ **FIXED:** Double clean sheet calculation bug was inflating GK/DEF predictions by ~2pts
- [ ] Why are predictions consistent across GWs?
- [x] ~~Why is Travers so popular?~~ **IDENTIFIED:** Name matching failure between "Travers" (db) vs "Mark Travers" (historical) causes fallback to season averages instead of recent 0-minute form
- [ ] **CRITICAL: Fix historical data matching** - Use ID mapping instead of names (historical.element ≠ players.id, e.g., Mo Salah: 328→381)
- [ ] Can we calculated expected goals conceded instead?
- [ ] Should we estimate expected defensive interactions? We might want to pull from a different data store here
- [ ] Bonus points - are we doing it? Do we want to enable always?
- [ ] Potential: Looking at season total minutes as a threshold for minutes predictions, should we not use % of minutes played in the last few gameweeks?
- [ ] Defensive coding everywhere! I would much rather fail obviously from a data issue. A good example of this is default parameter values.
- [ ] Bring in data from other leagues if we can!
- [ ] Quite a lot of defenders are predicted to score highly?

### Model Calibration Issues (From Prediction vs Reality Analysis)
- [ ] **Defender prediction logic needs refinement** - Model is 88.1% too optimistic for defenders (+1.58 pts average)
- [ ] **Elite player identification** - Under-predicting top performers (Salah: pred 6.0 vs actual 9.1, Isak: pred 3.8 vs actual 6.2)
- [ ] **Playing time estimation for young/squad players** - Over-predicting rotation-prone players (Lewis-Skelly: pred 5.05 vs actual 1.61)
- [ ] **Investigate defender over-prediction** - Why are defenders consistently predicted too high? Check clean sheet probability calculations
- [ ] **Improve ceiling detection** - Model may not be capturing the upside potential of premium players

### Bug Fixes Completed ✅
- [x] **Clean Sheet Double Calculation:** Fixed duplicate clean sheet points in `_calculate_base_fpl_points()` - defenders/GKs were getting 4pts twice instead of once