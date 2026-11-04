# FPL Points & Captain Prediction

A Fantasy Premier League points prediction and squad optimizer.

Pulls historical match/player data, trains models to predict xG, xA, and saves per player, keeps a running "current form" state for every player and team, simulates a gameweek thousands of times to get expected points, then picks the best 15-man squad under the real budget and formation rules.

## Data

Match and player stats come from [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights), a community-maintained archive of per-gameweek Premier League CSVs. Big thanks to whoever maintains it. Not affiliated with that repo or the official Premier League / FPL.

Prices come from the official FPL API.

## How it works

1. Preprocessing turns the raw per-gameweek CSVs into training data and the state tables the live system runs on.
2. Training: three XGBoost models (xG, xA, saves), Poisson objective, tuned with Optuna.
3. State and features: each real gameweek updates every player's and team's rolling form (EWMA), then that state becomes the features for the next gameweek.
4. Simulation: Monte Carlo, 50,000 runs, fully vectorized in numpy. For every player, draw minutes played, look up the model output for those minutes, draw goals/assists/saves, add up FPL points.
5. Optimization: a mixed-integer linear program (scipy.optimize.milp) picks the squad maximizing expected points under the £100m budget and 2 GK / 5 DEF / 5 MID / 3 FWD rules.

## Decisions worth knowing about

Players are tracked by a stable cross-season ID, not the raw dataset's season-scoped one, which gets reused for different people between seasons. Minutes played feeds into the models directly rather than just folded into per-90 rates, so a player back from injury isn't predicted off their old minutes pattern. Minutes are simulated in two layers: a per-player chance of 0 / partial / full minutes, then an exact minute drawn from a league-wide distribution within that bucket. Clean sheets and goals conceded fall out of summing everyone's simulated goals per team per run, rather than being modeled directly.

## What it doesn't do (yet)

- Clean sheets are checked over the whole match, not just while a player was on the pitch, so a defender subbed off at 0-0 who sees their team concede afterward still loses it here.
- Double gameweeks aren't handled, the pipeline assumes one match per player per gameweek.
- No bonus points, no own goals.
- Only predicts one gameweek at a time, no multi-week-ahead simulation for transfer or chip planning.
- No backtesting yet: prices use current values instead of historical ones, and there's no run comparing it against a dumb baseline, so I can't say how much the models actually add over something simple.