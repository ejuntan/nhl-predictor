"""Predict upcoming games using each team's current rolling state.

Fits the final model on every completed regular-season game, then applies it
to scheduled games whose features come from the team/goalie state as of the
last completed game.
"""
import json
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import features as F
from backtest import causal_temperature
from model import NHLModel, elo_ratings

DB = "data/nhl.sqlite3"
TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes", "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils", "NSH": "Nashville Predators",
    "NYI": "New York Islanders", "NYR": "New York Rangers",
    "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks", "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals", "ARI": "Arizona Coyotes",
}


def probable_starter(gg, team, game_date, played):
    """Guess who starts: the recent workhorse, unless this is a back-to-back.

    Real books price the confirmed starter, which is announced a few hours
    before puck drop. Before that, this is the best available guess and the
    single largest source of uncertainty in a pre-announcement prediction.
    """
    hist = gg[(gg.team == team) & (gg.date < game_date)].sort_values("date")
    if hist.empty:
        return None, "unknown", 0.0
    recent = hist[hist.date >= (hist.date.max() if len(hist) < 2 else
                                sorted(hist.date.unique())[-12:][0])]
    starts = recent[recent.is_starter == 1]
    if starts.empty:
        starts = recent
    counts = starts.goalie.value_counts()
    top = int(counts.index[0])
    share = float(counts.iloc[0] / counts.sum())

    last = hist.date.max()
    gap = (pd.Timestamp(game_date) - pd.Timestamp(last)).days
    if gap <= 1:
        last_start = hist[(hist.date == last) & (hist.is_starter == 1)]
        if len(last_start) and int(last_start.goalie.iloc[0]) == top:
            if len(counts) > 1:
                return int(counts.index[1]), "backup (back-to-back)", 0.55
            return top, "starter (b2b, no clear backup)", 0.5
    return top, "probable starter", share


def main(days=8):
    con = sqlite3.connect(DB, timeout=60)
    df = pd.read_pickle("data/features.pkl")
    reg = df[df.game_type == 2].copy()
    state = pickle.load(open("data/current_state.pkl", "rb"))
    gg = pd.read_pickle("data/goalie_games.pkl")
    rg_state = state["goalie"]
    sf = pd.read_pickle("data/sched.pkl")

    model = NHLModel().fit(reg)
    print(f"fitted on {len(reg)} games  tau={model.tau:+.3f} "
          f"kappa={model.kappa:.3f}  asof={state['asof']}")

    # Causal scoring-level multiplier from the most recent 250 games.
    tail = reg.sort_values(["date", "game_id"]).tail(250)
    lh, la = model.rates(tail)
    mult = float(np.clip(((tail.h_reg + tail.a_reg).mean()
                          / (lh + la).mean()), 0.90, 1.10))
    # Retune the tie weight on the same recent window, so the share of games
    # finishing level matches what the league is actually doing now.
    tie_target = float((tail.reg_outcome == 1).mean())
    kmult = float(np.clip(model.solve_kappa_mult(lh, la, tie_target),
                          0.6, 1.8))
    print(f"scoring multiplier {mult:.4f}  tie multiplier {kmult:.4f} "
          f"(recent tie rate {tie_target:.3f})")

    games = pd.read_sql(
        "SELECT * FROM games WHERE game_type=2 AND state='FUT' ORDER BY date",
        con)
    today = state["asof"]
    end = (pd.Timestamp(today) + timedelta(days=days)).strftime("%Y-%m-%d")
    up = games[(games.date > today) & (games.date <= end)].copy()
    print(f"upcoming games {today} -> {end}: {len(up)}")
    if up.empty:
        return

    # Assemble features from current state rather than from a played game.
    rows = []
    for g in up.itertuples(index=False):
        row = {"game_id": g.game_id, "season": g.season, "date": g.date,
               "home": g.home, "away": g.away, "game_type": 2,
               "start_utc": g.start_utc}
        ok = True
        for pre, team in (("h", g.home), ("a", g.away)):
            st = state["team"].get(team)
            if st is None:
                ok = False
                break
            for c in F.TEAM_STATS:
                row[f"{pre}_r_{c}"] = st[c]
            s = sf[(sf.game_id == g.game_id) & (sf.team == team)]
            for c in ("rest", "b2b", "games_last4", "travel_km", "tz_shift"):
                row[f"{pre}_{c}"] = float(s[c].iloc[0]) if len(s) else np.nan
            gid_, label, share = probable_starter(gg, team, g.date, reg)
            gs = rg_state.get(gid_) if gid_ else None
            if gs and gs["n"] > 0:
                raw = (gs["xg"] - gs["ga"]) / gs["n"]
                row[f"{pre}_g_gsax"] = raw * (gs["n"] /
                                              (gs["n"] + F.GOALIE_SHRINK_K))
                row[f"{pre}_g_gsax_raw"] = raw
                row[f"{pre}_g_shots_seen"] = gs["n"]
            else:
                row[f"{pre}_g_gsax"] = 0.0
                row[f"{pre}_g_gsax_raw"] = 0.0
                row[f"{pre}_g_shots_seen"] = 0.0
            row[f"{pre}_goalie"] = gid_
            row[f"{pre}_goalie_label"] = label
            row[f"{pre}_goalie_conf"] = share
        if ok:
            rows.append(row)
    up2 = pd.DataFrame(rows)
    pr = model.predict(up2, scoring_mult=np.full(len(up2), mult),
                       kappa_mult=np.full(len(up2), kmult))
    out = pd.concat([up2.reset_index(drop=True), pr.reset_index(drop=True)],
                    axis=1)

    # Elo, for a visible second opinion on each game.
    _, elo_now = elo_ratings(reg)
    out["elo_home"] = out.home.map(elo_now)
    out["elo_away"] = out.away.map(elo_now)

    import names as N
    ids = {int(v) for c in ("h_goalie", "a_goalie") for v in out[c].dropna()}
    nm = N.resolve(ids)
    for c in ("h", "a"):
        out[f"{c}_goalie_name"] = out[f"{c}_goalie"].map(
            lambda v: nm.get(str(int(v)), "TBD") if pd.notna(v) else "TBD")
    out.to_pickle("data/live_preds.pkl")
    print(out[["date", "away", "home", "p_home", "exp_total",
               "p_ot"]].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))
    return out


if __name__ == "__main__":
    main()
