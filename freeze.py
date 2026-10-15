"""Step 4: freeze the chosen model, then record predictions before puck drop.

Everything measured so far has been measured on the same 1,414 holdout
games, repeatedly. Nothing in the pipeline fits on them, but the *choices* --
which features to keep, which encodings to try -- have now been made partly
in the light of how they scored there. That is a slow leak which no leakage
check can detect, and the only clean fix is a fresh sample.

So: freeze the configuration, fingerprint the code that produced it, and
append predictions to a log that is never rewritten. Each row is written
before the game is played and carries the timestamp that proves it.

  python freeze.py freeze        record the configuration and code hashes
  python freeze.py log           append predictions for upcoming games
  python freeze.py score         score every logged game that has finished

The log is append-only by construction: a game_id already present is left
alone, so re-running `log` after a line move or a goalie announcement cannot
quietly improve a past prediction.
"""
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

DB = "data/nhl.sqlite3"
MANIFEST = "data/frozen.json"
LOG = "data/frozen_log.csv"
CODE = ["model.py", "features.py", "xg.py", "live.py", "backtest.py"]

# The configuration chosen in step 1. Recorded here so that a later run can
# assert it has not drifted.
CONFIG = {
    "frame": "per60",
    "tie_retune": True,
    "scoring_mult": True,
    "extra_features": [],
    "recalibration": False,
    "carry_stale_rating": True,
    "impute_ice_time": False,
    "notes": "per-60 rates from shift charts; causal tie retune; a game with "
             "no published shift chart keeps the team's carried-forward "
             "rating rather than a zero-filled one; no goalie or lineup "
             "extras (every one measured neutral in steps 1-3)",
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def code_hashes():
    return {f: sha(f) for f in CODE if os.path.exists(f)}


def cmd_freeze(force=False):
    if os.path.exists(MANIFEST) and not force:
        m = json.load(open(MANIFEST))
        print(f"already frozen {m['frozen_at']} as {m['id']}")
        print("pass --force to replace it (this invalidates the log's "
              "provenance)")
        return m
    df = pd.read_pickle("data/features.pkl")
    reg = df[df.game_type == 2]
    m = {
        "id": "nhl-v2-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": CONFIG,
        "code": code_hashes(),
        "trained_through": str(reg.date.max()),
        "train_games": int(len(reg)),
        "holdout_used_for_selection": ["20252026", "20262027"],
    }
    json.dump(m, open(MANIFEST, "w"), indent=2)
    print(json.dumps(m, indent=2))
    return m


def check_code():
    if not os.path.exists(MANIFEST):
        print("not frozen yet; run `python freeze.py freeze` first")
        return None
    m = json.load(open(MANIFEST))
    now = code_hashes()
    drift = {k: (v, now.get(k)) for k, v in m["code"].items()
             if now.get(k) != v}
    if drift:
        print("WARNING: code has changed since the freeze:")
        for k, (was, is_) in drift.items():
            print(f"  {k}: {was} -> {is_}")
        print("predictions logged from here are not the frozen model")
    return m


def cmd_log(days=8):
    m = check_code()
    if m is None:
        return
    import live
    out = live.main(days=days)
    if out is None or not len(out):
        print("no upcoming games to log")
        return
    cols = ["game_id", "date", "home", "away", "p_home", "exp_total", "p_ot",
            "p_over_55", "p_home_m15", "h_goalie_name", "a_goalie_name",
            "h_goalie_label", "a_goalie_label"]
    new = out[[c for c in cols if c in out.columns]].copy()
    new["model_id"] = m["id"]
    new["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        already = set(old.game_id)
        new = new[~new.game_id.isin(already)]
        if new.empty:
            print(f"nothing new; log holds {len(old)} predictions")
            return
        pd.concat([old, new], ignore_index=True).to_csv(LOG, index=False)
    else:
        new.to_csv(LOG, index=False)
    print(f"appended {len(new)} predictions; log now "
          f"{len(pd.read_csv(LOG))} rows")
    print(new[["date", "away", "home", "p_home"]].to_string(index=False))


def required_n(sd, delta, power=0.80, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sd / delta) ** 2))


def cmd_score():
    if not os.path.exists(LOG):
        print("no log yet")
        return
    log = pd.read_csv(LOG)
    con = sqlite3.connect(DB, timeout=60)
    res = pd.read_sql("""SELECT game_id, home_score, away_score, state,
                         last_period_type FROM games""", con)
    d = log.merge(res, on="game_id", how="left")
    done = d[d.state.isin(["OFF", "FINAL"]) & d.home_score.notna()].copy()
    print(f"logged {len(d)} predictions, {len(done)} games played")
    if len(done) == 0:
        return

    done["y"] = (done.home_score > done.away_score).astype(int)
    p = np.clip(done.p_home.values, 1e-6, 1 - 1e-6)
    y = done.y.values
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"  accuracy {(((p > .5) == (y == 1)).mean()):.4f}")
    print(f"  brier    {((p - y) ** 2).mean():.5f}")
    print(f"  log loss {ll.mean():.5f}  (sd per game {ll.std(ddof=1):.3f})")

    try:
        od = pd.read_pickle("data/odds_joined.pkl")[["game_id", "mkt_h"]]
        j = done.merge(od, on="game_id", how="inner")
    except FileNotFoundError:
        j = pd.DataFrame()
    if len(j) >= 20:
        q = np.clip(j.mkt_h.values, 1e-6, 1 - 1e-6)
        yy = j.y.values
        pp = np.clip(j.p_home.values, 1e-6, 1 - 1e-6)
        lm = -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
        lp = -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
        dd = lp - lm
        t, pv = stats.ttest_1samp(dd, 0)
        print(f"\n  vs market on {len(j)} of them: model {lp.mean():.5f} "
              f"market {lm.mean():.5f}  diff {dd.mean():+.5f} "
              f"t={t:+.2f} p={pv:.3f}")
        sd = dd.std(ddof=1)
        print(f"  games needed to call a {0.010:.3f} log-loss difference: "
              f"{required_n(sd, 0.010)}")
        print(f"  games needed to call a {0.005:.3f} difference: "
              f"{required_n(sd, 0.005)}")
    print("\nDo not change the model in response to these numbers until the "
          "sample above is reached.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "freeze":
        cmd_freeze("--force" in sys.argv)
    elif cmd == "log":
        cmd_log()
    elif cmd == "score":
        cmd_score()
    else:
        print(__doc__)
