"""What does not knowing the starter cost?

The backtest used the goalie who actually started. A live board does not know
that: it guesses, and gets it right about 64% of the time. This rebuilds the
same features with the *guessed* goalie's rating in place of the actual one,
re-runs the identical walk-forward backtest, and compares.

The difference between the two runs is the real price of an unconfirmed
starter -- and the ceiling on what a starter feed would be worth.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import features as F
from backtest import SEASONS, causal_temperature
from model import NHLModel

TEST = ["20252026", "20262027"]


def goalie_rating_series():
    """Each goalie's rolling rating over time, for an as-of lookup."""
    rg = pd.read_pickle("data/roll_goalie.pkl")
    gg = pd.read_pickle("data/goalie_games.pkl")
    s = rg.merge(gg[["game_id", "goalie", "date"]], on=["game_id", "goalie"])
    s = s[["goalie", "date", "g_gsax", "g_gsax_raw", "g_shots_seen"]].copy()
    s["goalie"] = s.goalie.astype("int64")
    s["d"] = pd.to_datetime(s.date)
    return s.sort_values("d").reset_index(drop=True)


def asof_rating(series, want):
    """Rating for each (goalie, date) pair, using the latest prior value."""
    want = want.copy()
    want["d"] = pd.to_datetime(want.date)
    want = want.sort_values("d").reset_index(drop=True)
    out = pd.merge_asof(want, series.rename(columns={"goalie": "g"}),
                        left_on="d", right_on="d",
                        left_by="goalie", right_by="g",
                        allow_exact_matches=False, direction="backward")
    return out


def build_guessed_features():
    """Feature table with the guessed goalie's rating instead of the real one."""
    df = pd.read_pickle("data/features.pkl")
    guess = pd.read_pickle("data/goalie_guess.pkl")
    series = goalie_rating_series()

    want = guess[["game_id", "team", "date", "guess"]].rename(
        columns={"guess": "goalie"})
    # merge_asof needs the by-key dtypes to match exactly.
    want = want[want.goalie.notna()].copy()
    want["goalie"] = want.goalie.astype("int64")
    rated = asof_rating(series, want)
    rated = rated[["game_id", "team", "goalie", "g_gsax", "g_gsax_raw",
                   "g_shots_seen"]]

    out = df.copy()
    for pre, col in (("h", "home"), ("a", "away")):
        r = rated.rename(columns={
            "team": col, "goalie": f"{pre}_goalie_guess",
            "g_gsax": f"{pre}_gsax_guess",
            "g_gsax_raw": f"{pre}_gsax_raw_guess",
            "g_shots_seen": f"{pre}_shots_seen_guess"})
        out = out.merge(r, on=["game_id", col], how="left")
    # Swap the goalie feature the model reads.
    for pre in ("h", "a"):
        out[f"{pre}_g_gsax_actual"] = out[f"{pre}_g_gsax"]
        out[f"{pre}_g_gsax"] = out[f"{pre}_gsax_guess"].fillna(0.0)
        out[f"{pre}_g_shots_seen"] = out[f"{pre}_shots_seen_guess"].fillna(0.0)
    return out


def run(df, label):
    preds = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
        if len(te) < 30 or len(tr) < 500:
            continue
        m = NHLModel().fit(tr)
        W = 250
        tail = tr.sort_values(["date", "game_id"]).tail(W)
        hist = pd.concat([tail, te.sort_values(["date", "game_id"])])
        lh, la = m.rates(hist)
        pt, at = lh + la, (hist.h_reg + hist.a_reg).values
        tie = (hist.reg_outcome == 1).astype(int).values
        mult, km, cur = [], [], 1.0
        bp, ba, bt = list(pt[:len(tail)]), list(at[:len(tail)]), list(tie[:len(tail)])
        blh, bla = list(lh[:len(tail)]), list(la[:len(tail)])
        for j in range(len(tail), len(hist)):
            mult.append(float(np.clip(np.mean(ba[-W:]) / np.mean(bp[-W:]), .9, 1.1)))
            if (j - len(tail)) % 50 == 0:
                cur = float(np.clip(m.solve_kappa_mult(
                    np.array(blh[-W:]), np.array(bla[-W:]),
                    float(np.mean(bt[-W:]))), 0.6, 1.8))
            km.append(cur)
            bp.append(pt[j]); ba.append(at[j]); bt.append(tie[j])
            blh.append(lh[j]); bla.append(la[j])
        te = te.sort_values(["date", "game_id"])
        pr = m.predict(te, scoring_mult=mult, kappa_mult=km)
        preds.append(pd.DataFrame({
            "game_id": te.game_id.values, "season": s,
            "y": te.home_win.values, "p": pr.p_home.values}))
    a = pd.concat(preds, ignore_index=True)
    a["variant"] = label
    return a


def score(a, seasons, label):
    cl = a[a.season.isin(seasons)]
    y, p = cl.y.values, np.clip(cl.p.values, 1e-6, 1 - 1e-6)
    return {"variant": label, "n": len(y),
            "acc": round(float(((p > .5) == (y == 1)).mean()), 4),
            "brier": round(brier_score_loss(y, p), 5),
            "logloss": round(log_loss(y, p), 5)}


if __name__ == "__main__":
    real = pd.read_pickle("data/features.pkl")
    real = real[real.game_type == 2].copy()
    guessed = build_guessed_features()
    guessed = guessed[guessed.game_type == 2].copy()
    print(f"features {len(real)}  guessed-goalie rows filled: "
          f"{guessed.h_gsax_guess.notna().sum()}")

    a_real = run(real, "confirmed starter")
    a_guess = run(guessed, "guessed starter")
    rows = [score(a_real, TEST, "confirmed starter (what the backtest did)"),
            score(a_guess, TEST, "guessed starter (what a live board does)")]
    print("\n" + "=" * 70)
    print("COST OF NOT KNOWING THE STARTER — clean holdout")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))
    d = rows[1]["logloss"] - rows[0]["logloss"]
    print(f"\nknowing the starter is worth {-d:+.5f} log loss"
          f"  ({rows[0]['acc']-rows[1]['acc']:+.4f} accuracy)")

    # paired significance on the same games
    j = a_real[a_real.season.isin(TEST)].merge(
        a_guess[a_guess.season.isin(TEST)], on="game_id", suffixes=("_r", "_g"))
    def ll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    dd = ll(j.y_g.values, j.p_g.values) - ll(j.y_r.values, j.p_r.values)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    print(f"paired: mean {dd.mean():+.5f}  SE {se:.5f}  t = {dd.mean()/se:+.2f}"
          f"  (n={len(dd)})")

    # does the model do worse on games the heuristic would have missed?
    guess = pd.read_pickle("data/goalie_guess.pkl")
    ok = guess.groupby("game_id").right.all().rename("both_right").reset_index()
    k = a_real[a_real.season.isin(TEST)].merge(ok, on="game_id", how="inner")
    print("\nWithin the confirmed-starter run, split by whether a live board"
          "\nwould have named BOTH starters correctly:")
    for v, g in k.groupby("both_right"):
        y, p = g.y.values, np.clip(g.p.values, 1e-6, 1 - 1e-6)
        print(f"  both guessed right = {bool(v)}: n={len(g):4d}  "
              f"acc {((p>.5)==(y==1)).mean():.4f}  "
              f"brier {brier_score_loss(y,p):.5f}  logloss {log_loss(y,p):.5f}")
