"""Build the truth file rectest.js checks the page's record against.

Written down rather than improvised each run, because the record now has
two sources: a prediction logged before puck drop where one exists, and the
walk-forward backtest for games played before the log began. A truth file
derived from only one of them would disagree with the page for the right
reason, which is the worst kind of failing test.
"""
import json

import pandas as pd

d = json.load(open("out/tool_data.json"))
days = sorted({g["d"] for g in d["past"]})

cur = pd.read_pickle("data/backtest_preds.pkl")
cur = cur[cur.season == "20262027"].copy()
try:
    log = pd.read_csv("data/frozen_log.csv")[["game_id", "p_home"]]
    log = log.rename(columns={"p_home": "p_logged"}).drop_duplicates("game_id")
    cur = cur.merge(log, on="game_id", how="left")
    cur["p_home"] = cur.p_logged.fillna(cur.p_home)
except FileNotFoundError:
    pass
cur = cur[cur.date.isin(days)]
cur["hit"] = (cur.p_home > .5) == (cur.home_win == 1)

json.dump({"byday": {k: [int(g.hit.sum()), int((~g.hit).sum())]
                     for k, g in cur.groupby("date")},
           "total": [int(cur.hit.sum()), int((~cur.hit).sum())]},
          open("/tmp/truth.json", "w"))
print(f"truth: {int(cur.hit.sum())}-{int((~cur.hit).sum())} over {len(cur)} games, "
      f"{len(days)} days")
