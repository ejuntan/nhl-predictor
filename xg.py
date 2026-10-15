"""Shot-level expected-goals model.

Fit on unblocked shot attempts (Fenwick: shot-on-goal, goal, missed-shot)
from the training seasons only, then applied to every shot. Keeping the fit
off the holdout seasons means the game-level backtest never benefits from an
xG model that saw its outcomes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

FENWICK = ("shot-on-goal", "goal", "missed-shot")
SHOT_TYPES = ["wrist", "snap", "slap", "tip-in", "backhand", "deflected",
              "wrap-around", "bat", "poke", "between-legs", "cradle"]
GOAL_X, GOAL_Y = 89.0, 0.0


def shot_features(d):
    """Geometry, context and sequence features for each shot attempt."""
    f = pd.DataFrame(index=d.index)
    dx = GOAL_X - d.adj_x
    dy = d.adj_y - GOAL_Y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    # Shots from behind the goal line get a negative dx; the angle formula
    # still works but distance should stay positive.
    f["dist"] = dist
    f["log_dist"] = np.log1p(dist)
    ang = np.degrees(np.arctan2(np.abs(dy), np.maximum(dx, 0.1)))
    f["angle"] = ang
    f["angle_sq"] = ang ** 2
    f["dist_x_angle"] = dist * ang / 100.0
    f["behind_net"] = (d.adj_x > GOAL_X).astype(int)
    f["abs_y"] = np.abs(d.adj_y)

    for t in SHOT_TYPES:
        f[f"st_{t}"] = (d.shot_type == t).astype(int)

    sk_sh = d.shooters_skaters.fillna(5)
    sk_def = d.defenders_skaters.fillna(5)
    diff = (sk_sh - sk_def).clip(-2, 2)
    for v in (-2, -1, 1, 2):
        f[f"sd_{v}"] = (diff == v).astype(int)
    f["empty_net"] = (d.goalie_in_net == 0).astype(int)
    f["sk3"] = (sk_sh <= 3).astype(int)      # 3-on-3 overtime
    f["sk4"] = (sk_sh == 4).astype(int)

    prev_fen = d.prev_event.isin(FENWICK)
    dt = d.prev_dt.fillna(999)
    f["rebound"] = (prev_fen & (dt <= 3)).astype(int)
    f["rebound_fast"] = (prev_fen & (dt <= 1)).astype(int)
    f["rush"] = ((dt <= 5) & (d.prev_dist.fillna(0) >= 40)).astype(int)
    f["prev_dt"] = dt.clip(0, 30)
    f["off_faceoff"] = ((d.prev_event == "faceoff") & (dt <= 5)).astype(int)
    return f.astype(float)


def fit_xg(shots, train_seasons, valid_seasons=()):
    fen = shots[shots.event.isin(FENWICK)].copy()
    y = (fen.event == "goal").astype(int).values
    X = shot_features(fen)
    tr = fen.season.isin(train_seasons).values
    sc = StandardScaler().fit(X.values[tr])
    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(sc.transform(X.values[tr]), y[tr])

    out = {"model": model, "scaler": sc, "cols": list(X.columns)}
    p_tr = model.predict_proba(sc.transform(X.values[tr]))[:, 1]
    print(f"xG train  n={tr.sum():>7}  AUC={roc_auc_score(y[tr], p_tr):.4f}  "
          f"logloss={log_loss(y[tr], p_tr):.5f}  "
          f"sum(xG)={p_tr.sum():.0f} vs goals={y[tr].sum()}")
    if len(valid_seasons):
        va = fen.season.isin(valid_seasons).values
        if va.sum():
            p_va = model.predict_proba(sc.transform(X.values[va]))[:, 1]
            print(f"xG valid  n={va.sum():>7}  AUC={roc_auc_score(y[va], p_va):.4f}  "
                  f"logloss={log_loss(y[va], p_va):.5f}  "
                  f"sum(xG)={p_va.sum():.0f} vs goals={y[va].sum()}")
            b = pd.DataFrame({"p": p_va, "y": y[va]})
            b["bin"] = pd.cut(b.p, [0, .02, .05, .08, .12, .2, .35, 1.0])
            print("  calibration (valid):")
            g = b.groupby("bin", observed=True).agg(
                n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
            print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    return out


def apply_xg(xg, shots):
    """Return an xG value for every Fenwick shot (NaN for blocked shots)."""
    out = pd.Series(np.nan, index=shots.index)
    m = shots.event.isin(FENWICK)
    if m.sum():
        X = shot_features(shots[m])[xg["cols"]].values
        out[m] = xg["model"].predict_proba(xg["scaler"].transform(X))[:, 1]
    return out
