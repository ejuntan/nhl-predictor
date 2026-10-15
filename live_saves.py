"""Price goalie saves for the upcoming slate.

Fits on every completed start, then applies the model to the probable
starter in each scheduled game. The level multiplier from section 17 is
carried over: a model fitted on older seasons runs slightly low against the
league's current shot volume, and the correction tracks that from the
trailing 400 starts.

The starter is a guess until the morning announcement, and section 11
measured that guess at 64% accurate. For saves this matters more than it
does for the win model: the whole quantity is one goalie's workload, so a
wrong name is a wrong price rather than a small perturbation. Each row
carries the guess's confidence so the page can say so.
"""
import pickle
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

import names as N
import saves_model as S
from live import probable_starter
from saves_v2 import fit_size

DB = "data/nhl.sqlite3"
WIN = 400


def main(days=8):
    con = sqlite3.connect(DB, timeout=60)
    d = S.build(con)
    F, off = S.feature_frame(d)
    mu_, sd_ = F.values.mean(0), F.values.std(0) + 1e-9
    Z = (F.values - mu_) / sd_
    w = np.exp(off)
    m = PoissonRegressor(alpha=S.ALPHA, max_iter=8000).fit(
        Z, d.saves.values / w, sample_weight=w)
    fitted = m.predict(Z) * w
    size = fit_size(d.saves.values, fitted)

    # Causal level correction, from the most recent starts only.
    tail = d.sort_values(["date", "game_id"]).tail(WIN)
    ft = fitted[np.isin(d.index, tail.index)]
    mult = float(np.clip(tail.saves.mean() / ft.mean(), 0.95, 1.05))
    print(f"fitted on {len(d)} starts  NB size {size:.0f}  "
          f"level multiplier {mult:.4f}")

    gstate, tstate = S.current_state(con)
    gg = pd.read_pickle("data/goalie_games.pkl")
    games = pd.read_sql(
        "SELECT * FROM games WHERE game_type=2 AND state='FUT' ORDER BY date",
        con)
    asof = pd.read_sql(
        "SELECT MAX(date) d FROM games WHERE state IN ('OFF','FINAL')",
        con).d.iloc[0]
    end = (pd.Timestamp(asof) + timedelta(days=days)).strftime("%Y-%m-%d")
    up = games[(games.date > asof) & (games.date <= end)]
    print(f"upcoming games {asof} -> {end}: {len(up)}")

    rows = []
    for g in up.itertuples(index=False):
        for team, opp, is_home in ((g.home, g.away, 1), (g.away, g.home, 0)):
            gid, label, share = probable_starter(gg, team, g.date, None)
            st = gstate.get(gid)
            own, oppst = tstate.get(team), tstate.get(opp)
            if not (st and own and oppst and st["n"] >= S.MIN_PRIOR):
                continue
            rows.append({
                "game_id": g.game_id, "date": g.date, "team": team,
                "opp": opp, "is_home": is_home, "goalie_id": gid,
                "label": label, "conf": share,
                "e_saves60": st["saves60"], "e_sog60": st["sog60"],
                "e_toi": st["toi"], "own_sf60": own["sf60"],
                "own_sa60": own["sa60"], "opp_sf60": oppst["sf60"],
                "opp_sa60": oppst["sa60"]})
    if not rows:
        print("no priceable starters")
        return None
    u = pd.DataFrame(rows)
    Fu, offu = S.feature_frame(u)
    u["mu"] = m.predict((Fu.values - mu_) / sd_) * np.exp(offu) * mult
    u["size"] = size
    u["exp_toi_min"] = u.e_toi / 60
    for L in S.LINES:
        u[f"p_over_{L}"] = 1 - nbinom.cdf(
            int(np.floor(L)), size, size / (size + u.mu.values))
    nm = N.resolve({int(v) for v in u.goalie_id.dropna()})
    u["goalie"] = u.goalie_id.map(
        lambda v: nm.get(str(int(v)), "TBD") if pd.notna(v) else "TBD")
    u = u.sort_values(["date", "game_id", "mu"], ascending=[True, True, False])
    u.to_pickle("data/live_saves.pkl")
    print(f"priced {len(u)} starters across {u.game_id.nunique()} games")
    print(u[["date", "team", "goalie", "exp_toi_min", "mu",
             "p_over_24.5", "p_over_27.5"]].head(10).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    return u


if __name__ == "__main__":
    main()
