#!/bin/bash
# Daily path: pick up finished games, rebuild features, re-predict both models.
# Does not refit the xG model -- run fit_xg.py yourself if you want that.
set -e
cd "$(dirname "$0")"
PY=./.venv/bin/python
$PY fetch.py games
$PY fetch.py shots          # play-by-play for new games (resumable)
$PY fetch_shifts.py         # shift charts for new games (resumable)
$PY build_features.py
$PY live.py                 # game predictions
$PY live_props.py           # shots-on-goal prices
$PY make_csvs.py
$PY make_csvs2.py
echo "done -- out/*.csv refreshed"
