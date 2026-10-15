#!/bin/bash
# Everything the page needs, in dependency order, once a day.
#
# Each stage is cheap except the backtest (~4 min), which has to re-run
# because the page's finished-game panels come from it: a game played last
# night needs a prediction made by a model that never saw it.
#
# Exits non-zero on the first failure and says which stage, so a scheduler
# can tell "no new games" from "the NHL changed its API again".
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
LOG=data/daily.log

stage() { printf '\n=== %s  %s ===\n' "$1" "$(date '+%H:%M:%S')"; }

{
  printf '\n\n######## %s ########\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  stage "new results and play-by-play"
  $PY fetch.py games
  $PY fetch.py shots
  $PY fetch_shifts.py
  $PY fetch_lineups.py || echo "  (lineups failed; not fatal)"

  stage "features"
  $PY build_features.py

  stage "walk-forward backtest (the page's past games)"
  $PY backtest.py > data/backtest_run.log 2>&1
  tail -3 data/backtest_run.log

  stage "shots-on-goal caches"
  $PY build_props.py

  stage "predictions for the upcoming slate"
  $PY live.py | tail -3
  $PY live_props.py | tail -2
  $PY live_saves.py | tail -2

  stage "live prices and the value finder"
  # Prices are the one stage allowed to come back empty: books post NHL
  # lines a day or two out, so an early-morning run may find nothing.
  $PY live_edge.py 8 | tail -4 || echo "  (no prices available this run)"

  stage "page"
  $PY export_tool.py
  $PY tool/build.py

  stage "tests"
  $PY tool/linetest.py | tail -1
  $JSC tool/mathtest.js | tail -1
  $JSC tool/daytest.js | tail -1

  stage "frozen prediction log"
  $PY freeze.py log  | tail -3
  $PY freeze.py score | tail -6

  printf '\nOK  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee -a "$LOG"
