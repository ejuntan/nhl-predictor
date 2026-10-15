"""Step 1: old model vs every modified version, on identical games.

Each variant is refit walk-forward -- season s is predicted by a model that
saw only seasons before s -- and every variant is scored on the same game
list, so differences are attributable to the change alone. Significance is a
paired bootstrap over games: the same resampled game list scores both
models, which removes the game-to-game difficulty that dominates the raw
variance of any single model's score.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import model as M
import variants as V
from backtest import causal_temperature

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
CLEAN = ["20252026", "20262027"]
WINDOW = 250
REF = "v2 per-60 + tie retune (shipped)"

GOALIE = ["backup", "share", "k300"]
LINEUP = ["n_out", "role_out", "toi_out", "top_out"]
ADJ = ["adj_att_xgf_5v5", "adj_def_xgf_5v5", "adj_att_cf_5v5",
       "adj_def_cf_5v5", "adj_att_gf_all", "adj_def_gf_all"]

VARIANTS = {
    "v1 counting stats, kappa fixed": dict(frame="raw", tie=False),
    "v1 counting stats + tie retune": dict(frame="raw", tie=True),
    "v2 per-60, kappa fixed": dict(frame="per60", tie=False),
    REF: dict(frame="per60", tie=True),
    "v2 per-60 with imputed ice time": dict(frame="per60i", tie=True),
    "v2 per-60, no zero-filled ratings": dict(frame="per60c", tie=True),
    "v2 + causal recalibration": dict(frame="per60", tie=True, recal=True),
    "v2 + confirmed-goalie encodings": dict(frame="goalie", tie=True,
                                            extra=GOALIE),
    "v2 + lineup absences": dict(frame="lineup", tie=True, extra=LINEUP),
    "v2 + opponent-adjusted ratings": dict(frame="adj", tie=True, extra=ADJ),
}


def patched(extra):
    """build_matrix plus one variant's extra columns and their home gaps."""
    orig = M.build_matrix

    def f(d):
        x = orig(d)
        for k in extra:
            hc, ac = f"h_{k}", f"a_{k}"
            if hc not in d.columns:
                raise KeyError(hc)
            x[hc] = d[hc].astype(float).fillna(0.0)
            x[ac] = d[ac].astype(float).fillna(0.0)
            x[f"{k}_gap"] = x[hc] - x[ac]
        return x.astype(float).fillna(0.0)
    return f, orig


def walk_forward(df, tie=True, extra=(), recal=False):
    """One variant, refit season by season on prior seasons only."""
    f, orig = patched(extra)
    M.build_matrix = f
    try:
        out = []
        for i, s in enumerate(SEASONS):
            if i == 0:
                continue
            tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
            if len(te) < 30 or len(tr) < 500:
                continue
            m = M.NHLModel().fit(tr)

            tail = tr.sort_values(["date", "game_id"]).tail(WINDOW)
            te = te.sort_values(["date", "game_id"])
            hist = pd.concat([tail, te])
            lh, la = m.rates(hist)
            pred_tot, act_tot = lh + la, (hist.h_reg + hist.a_reg).values
            n0 = len(tail)

            mult, bp, ba = [], list(pred_tot[:n0]), list(act_tot[:n0])
            for j in range(n0, len(hist)):
                mult.append(float(np.clip(
                    np.mean(ba[-WINDOW:]) / np.mean(bp[-WINDOW:]), .90, 1.10)))
                bp.append(pred_tot[j])
                ba.append(act_tot[j])

            km = None
            if tie:
                ta = (hist.reg_outcome == 1).astype(int).values
                bt, blh, bla = (list(ta[:n0]), list(lh[:n0]), list(la[:n0]))
                km, cur = [], 1.0
                for j in range(n0, len(hist)):
                    if (j - n0) % 50 == 0:
                        cur = float(np.clip(m.solve_kappa_mult(
                            np.array(blh[-WINDOW:]), np.array(bla[-WINDOW:]),
                            float(np.mean(bt[-WINDOW:]))), 0.6, 1.8))
                    km.append(cur)
                    bt.append(ta[j])
                    blh.append(lh[j])
                    bla.append(la[j])

            pr = m.predict(te, scoring_mult=mult, kappa_mult=km)
            p = pr.p_home.values
            if recal:
                p = causal_temperature(p, te.home_win.values)
            out.append(pd.DataFrame({
                "game_id": te.game_id.values, "season": te.season.values,
                "date": te.date.values, "home": te.home.values,
                "away": te.away.values, "y": te.home_win.values, "p": p,
                "p_ot": pr.p_ot.values, "exp_total": pr.exp_total.values}))
        return pd.concat(out, ignore_index=True)
    finally:
        M.build_matrix = orig


def metrics(y, p):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": len(y), "acc": ((p > .5) == (y == 1)).mean(),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def _ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(y, pa, pb, b=4000, seed=0):
    """Variant a minus reference b, resampling games (not predictions)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    pa = np.clip(np.asarray(pa, float), 1e-6, 1 - 1e-6)
    pb = np.clip(np.asarray(pb, float), 1e-6, 1 - 1e-6)
    dll = _ll(y, pa) - _ll(y, pb)
    dbr = (pa - y) ** 2 - (pb - y) ** 2
    dac = (((pa > .5) == (y == 1)).astype(float)
           - ((pb > .5) == (y == 1)).astype(float))
    n = len(y)
    idx = rng.integers(0, n, size=(b, n))
    res = {}
    for nm, v in (("logloss", dll), ("brier", dbr), ("acc", dac)):
        draws = v[idx].mean(axis=1)
        win = (draws > 0) if nm == "acc" else (draws < 0)
        better = win.mean() + 0.5 * (draws == 0).mean()
        res[nm] = {"delta": v.mean(), "lo": np.quantile(draws, .025),
                   "hi": np.quantile(draws, .975), "p_better": better}
    return res


def disagreement(y, pa, pb):
    pick_a, pick_b = np.asarray(pa) > .5, np.asarray(pb) > .5
    d = pick_a != pick_b
    if d.sum() == 0:
        return {"n": 0, "a_acc": np.nan, "b_acc": np.nan}
    yy = np.asarray(y, int)[d]
    return {"n": int(d.sum()),
            "a_acc": (pick_a[d] == (yy == 1)).mean(),
            "b_acc": (pick_b[d] == (yy == 1)).mean()}


def run():
    frames = V.load()
    preds = {}
    for name, cfg in VARIANTS.items():
        df = frames[cfg["frame"]]
        df = df[df.game_type == 2].copy()
        preds[name] = walk_forward(df, tie=cfg.get("tie", True),
                                   extra=cfg.get("extra", ()),
                                   recal=cfg.get("recal", False))
        print(f"done: {name}  ({len(preds[name])} predictions)", flush=True)

    # Identical games only: the intersection across every variant, clean
    # holdout seasons only.
    pd.to_pickle(preds, "data/compare_preds_all.pkl")
    common = None
    for p in preds.values():
        ids = set(p[p.season.isin(CLEAN)].game_id)
        common = ids if common is None else (common & ids)
    common = sorted(common)
    print(f"\ncommon clean-holdout games: {len(common)}")
    for k in preds:
        preds[k] = preds[k][preds[k].game_id.isin(common)] \
            .sort_values("game_id").reset_index(drop=True)
    pd.to_pickle(preds, "data/compare_preds.pkl")
    return preds


def report(preds):
    ref = preds[REF]
    y = ref.y.values

    print("\n" + "=" * 78)
    print(f"POOLED CLEAN HOLDOUT  n={len(y)}  "
          f"({', '.join(sorted(ref.season.unique()))})")
    print("=" * 78)
    rows = []
    for k, p in preds.items():
        r = metrics(y, p.p.values)
        r["variant"] = k
        rows.append(r)
    t = pd.DataFrame(rows)[["variant", "n", "acc", "brier", "logloss"]]
    base = t[t.variant == REF].iloc[0]
    t["d_brier"] = t.brier - base.brier
    t["d_logloss"] = t.logloss - base.logloss
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    print("\n" + "=" * 78)
    print(f"PAIRED BOOTSTRAP vs {REF!r}  (4000 resamples over games)")
    print("negative delta = variant better on brier/logloss; "
          "positive = better on acc")
    print("=" * 78)
    for k, p in preds.items():
        if k == REF:
            continue
        r = paired_bootstrap(y, p.p.values, ref.p.values)
        d = disagreement(y, p.p.values, ref.p.values)
        print(f"\n{k}")
        for nm in ("logloss", "brier", "acc"):
            v = r[nm]
            sig = "*" if (v["lo"] > 0) == (v["hi"] > 0) else " "
            print(f"  {nm:<8} {v['delta']:+.5f}  "
                  f"95% CI [{v['lo']:+.5f}, {v['hi']:+.5f}] {sig}  "
                  f"P(better) {v['p_better']:.3f}")
        if d["n"]:
            print(f"  flips the pick on {d['n']} of {len(y)} games "
                  f"({d['n']/len(y):.1%}); on those: variant "
                  f"{d['a_acc']:.3f} vs shipped {d['b_acc']:.3f}")
        else:
            print("  never changes the predicted winner")

    print("\n" + "=" * 78)
    print("BY SEASON (accuracy / brier / log loss)")
    print("=" * 78)
    rows = []
    for k, p in preds.items():
        for s, g in p.groupby("season"):
            r = metrics(g.y.values, g.p.values)
            rows.append({"variant": k, "season": s, **r})
    bys = pd.DataFrame(rows).pivot(index="variant", columns="season",
                                   values=["acc", "brier", "logloss"])
    print(bys.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(pd.read_pickle("data/compare_preds.pkl"))
    else:
        report(run())
