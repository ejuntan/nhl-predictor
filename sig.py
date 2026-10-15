"""Is the variant's gain real, or sampling noise?

Paired comparison on per-game log loss, plus a bootstrap over games.
"""
import numpy as np
import pandas as pd

import model as M
from experiment import SEASONS, TEST, VALID, matrix_for


def per_game(df, variant, seasons):
    f, orig = matrix_for(variant)
    M.build_matrix = f
    try:
        out = []
        for i, s in enumerate(SEASONS):
            if s not in seasons or i == 0:
                continue
            tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
            if len(te) < 30 or len(tr) < 500:
                continue
            m = M.NHLModel().fit(tr)
            W = 250
            tail = tr.sort_values(["date", "game_id"]).tail(W)
            hist = pd.concat([tail, te.sort_values(["date", "game_id"])])
            lh, la = m.rates(hist)
            pt, at = lh + la, (hist.h_reg + hist.a_reg).values
            mult, bp, ba = [], list(pt[:len(tail)]), list(at[:len(tail)])
            for j in range(len(tail), len(hist)):
                mult.append(float(np.clip(np.mean(ba[-W:]) / np.mean(bp[-W:]),
                                          0.90, 1.10)))
                bp.append(pt[j]); ba.append(at[j])
            te = te.sort_values(["date", "game_id"])
            pr = m.predict(te, scoring_mult=mult)
            out.append(pd.DataFrame({"game_id": te.game_id.values,
                                     "y": te.home_win.values,
                                     "p": pr.p_home.values}))
        return pd.concat(out, ignore_index=True)
    finally:
        M.build_matrix = orig


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(df, seasons, label):
    a = per_game(df, "base", seasons)
    b = per_game(df, "adj_lean", seasons)
    j = a.merge(b, on="game_id", suffixes=("_a", "_b"))
    assert (j.y_a == j.y_b).all()
    da = ll(j.y_a.values, j.p_a.values)
    db = ll(j.y_b.values, j.p_b.values)
    d = da - db                      # positive = variant is better
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se
    rng = np.random.default_rng(7)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    better = ((db < da).mean())
    print(f"\n{label}  (n={n})")
    print(f"  current model   log loss {da.mean():.5f}")
    print(f"  variant         log loss {db.mean():.5f}")
    print(f"  mean gain       {d.mean():+.5f}  (SE {se:.5f}, t = {t:+.2f})")
    print(f"  95% bootstrap   [{lo:+.5f}, {hi:+.5f}]"
          f"  -> {'significant' if lo > 0 else 'NOT significant'}")
    print(f"  variant better on {better:.1%} of games")
    return d.mean(), lo, hi


if __name__ == "__main__":
    df = pd.read_pickle("data/features_adj.pkl")
    compare(df, VALID, "VALIDATION 2023-24 + 2024-25")
    print("\n" + "=" * 60)
    print("Now the test set, scored once for the variant chosen above.")
    print("=" * 60)
    compare(df, TEST, "TEST 2025-26 + 2026-27")
