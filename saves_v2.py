"""Finish the saves model: fix the dispersion, then price it.

Section 10 measured a real edge here -- the model beats a goalie's own
recent average by 6.12% of error at t=6.9, eight times its margin on skater
shots, because a goalie's past save count reflects the opponents he happened
to face rather than anything about him. It was shelved as unusable because
the probabilities were miscalibrated in the upper bands.

Re-diagnosed, the fault is narrower than the note claimed. The calibration
slope is 1.122 with a +0.170 intercept: the model is under-confident and
biased low, not compressed and wrong. Both symptoms point at one cause.

`fit_nb` estimates dispersion from training residuals, where mu is fitted
in-sample and therefore too accurate. Out of sample the spread is wider,
because the realised variance is Var(y|mu) plus the error in mu itself. A
dispersion fitted in-sample is too tight, the tails come out too thin, and
every "over" at a high line is priced low -- exactly the observed pattern.

The fix is to fit dispersion where the model will actually be used: on the
previous season's out-of-sample predictions. Nothing is fitted on the clean
holdout. Two alternatives are measured beside it for honesty: a
mean-dependent dispersion, and a Platt layer fitted on the validation
seasons.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.linear_model import LogisticRegression, PoissonRegressor

import saves_model as S

SEASONS = S.SEASONS
VALID = ["20232024", "20242025"]
CLEAN = ["20252026", "20262027"]
LINES = S.LINES


def nb_ll(y, mu, r):
    p = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(p) + y * np.log1p(-p)).sum()


def fit_size(y, mu):
    grid = np.concatenate([np.arange(3, 60, 1.0), np.arange(60, 400, 10.0)])
    return max(grid, key=lambda r: nb_ll(y, mu, r))


def fit_size_curve(y, mu):
    """Dispersion allowed to depend on the predicted level: r = a * mu^b."""
    best, bll = (fit_size(y, mu), 0.0), -np.inf
    for a in np.arange(0.2, 6.01, 0.2):
        for b in np.arange(-1.0, 1.51, 0.1):
            r = np.clip(a * mu ** b, 1.0, 500.0)
            ll = nb_ll(y, mu, r)
            if ll > bll:
                best, bll = (a, b), ll
    return best


def over(mu, size, line):
    k = int(np.floor(line))
    size = np.asarray(size, float)
    return 1 - nbinom.cdf(k, size, size / (size + mu))


def walk(d):
    """Walk-forward, carrying last season's out-of-sample residuals forward."""
    out, prev = [], None
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = d[d.season.isin(SEASONS[:i])], d[d.season == s]
        if len(te) < 100 or len(tr) < 500:
            continue
        Ftr, otr = S.feature_frame(tr)
        Fte, ote = S.feature_frame(te)
        mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu_) / sd_, (Fte.values - mu_) / sd_
        w = np.exp(otr)
        m = PoissonRegressor(alpha=S.ALPHA, max_iter=8000).fit(
            Ztr, tr.saves.values / w, sample_weight=w)
        mu_tr = m.predict(Ztr) * w
        mu_te = m.predict(Zte) * np.exp(ote)

        in_sample = fit_size(tr.saves.values, mu_tr)
        # Out-of-sample dispersion: last season's held-out predictions, which
        # is the only honest picture of how wrong mu actually gets.
        if prev is None:
            oos, curve = in_sample, (in_sample, 0.0)
        else:
            oos = fit_size(prev["y"], prev["mu"])
            curve = fit_size_curve(prev["y"], prev["mu"])
        o = te[["game_id", "goalie_id", "season", "date", "team", "opp",
                "saves", "e_saves60", "e_toi"]].copy()
        o["mu"] = mu_te
        o["size_in"] = in_sample
        o["size_oos"] = oos
        o["size_curve"] = np.clip(curve[0] * mu_te ** curve[1], 1.0, 500.0)
        o["base"] = te.e_saves60.values * (te.e_toi.values / 3600)

        # League shot volume drifts between seasons, and mu is fitted on
        # older ones. The win model already corrects for exactly this with a
        # trailing multiplier; saves had no equivalent. Causal: a start is
        # only ever adjusted by starts already played.
        te_s = te.sort_values(["date", "game_id"])
        idx = te_s.index
        mu_s = pd.Series(mu_te, index=te.index).loc[idx].values
        act_s = te_s.saves.values
        WIN = 400
        buf_a = list(prev["y"][-WIN:]) if prev else []
        buf_p = list(prev["mu"][-WIN:]) if prev else []
        mult = []
        for j in range(len(te_s)):
            r = (np.mean(buf_a[-WIN:]) / np.mean(buf_p[-WIN:])
                 if len(buf_a) >= 100 else 1.0)
            mult.append(float(np.clip(r, 0.95, 1.05)))
            buf_a.append(act_s[j])
            buf_p.append(mu_s[j])
        o["mult"] = pd.Series(mult, index=idx).loc[te.index].values
        o["mu_adj"] = o.mu * o.mult
        out.append(o)
        print(f"  fit<={SEASONS[i-1]} -> {s}: {len(te)} starts  "
              f"size in-sample {in_sample:.0f}  out-of-sample {oos:.0f}  "
              f"curve {curve[0]:.1f}*mu^{curve[1]:+.1f}", flush=True)
        prev = {"y": te.saves.values, "mu": mu_te}
    return pd.concat(out, ignore_index=True)


def stack(d, size_col, p=None):
    """One row per (start, line) with the model's probability of the over."""
    rows = []
    for L in LINES:
        q = (over(d.mu.values, d[size_col].values, L) if p is None
             else p[L])
        rows.append(pd.DataFrame({"line": L, "p": q,
                                  "y": (d.saves > L).astype(int).values,
                                  "season": d.season.values}))
    return pd.concat(rows, ignore_index=True)


def calib(a, label):
    a = a.copy()
    a["bin"] = pd.cut(a.p, [0, .1, .2, .3, .4, .5, .6, .7, 1.0])
    g = a.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    g["gap"] = g.act - g.pred
    g["moe"] = 1.96 * np.sqrt(g.act * (1 - g.act) / g.n)
    g["off"] = np.where(g.gap.abs() > g.moe, "<-- outside", "")
    z = np.log(np.clip(a.p, 1e-6, 1 - 1e-6) / (1 - np.clip(a.p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(
        z.values.reshape(-1, 1), a.y.values)
    inside = int((g.gap.abs() <= g.moe).sum())
    print(f"\n--- {label} ---")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"slope {lr.coef_[0][0]:+.3f}  intercept {lr.intercept_[0]:+.3f}  "
          f"bands inside the interval {inside}/{len(g)}  "
          f"brier {((a.p - a.y) ** 2).mean():.5f}")
    return lr


def main():
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    d = S.build(con)
    pr = walk(d)
    pr.to_pickle("data/saves_preds_v2.pkl")
    cl = pr[pr.season.isin(CLEAN)]
    va = pr[pr.season.isin(VALID)]
    print("\n" + "=" * 74)
    print(f"CALIBRATION ON THE CLEAN HOLDOUT  ({len(cl)} starts, "
          f"{len(cl) * len(LINES)} prices)")
    print("=" * 74)
    calib(stack(cl, "size_in"), "dispersion fitted in-sample (shipped)")
    calib(stack(cl, "size_oos"), "dispersion fitted out-of-sample")
    calib(stack(cl, "size_curve"), "dispersion as a function of the mean")

    cl_adj = cl.copy()
    cl_adj["mu"] = cl_adj.mu_adj
    calib(stack(cl_adj, "size_oos"),
          f"causal level correction (mean multiplier "
          f"{cl.mult.mean():.4f})")

    # Platt, fitted on validation only.
    av = stack(va, "size_in")
    zv = np.log(np.clip(av.p, 1e-6, 1 - 1e-6) / (1 - np.clip(av.p, 1e-6, 1 - 1e-6)))
    pl = LogisticRegression(C=1e6, max_iter=2000).fit(
        zv.values.reshape(-1, 1), av.y.values)
    ac = stack(cl, "size_in")
    zc = np.log(np.clip(ac.p, 1e-6, 1 - 1e-6) / (1 - np.clip(ac.p, 1e-6, 1 - 1e-6)))
    ac2 = ac.copy()
    ac2["p"] = pl.predict_proba(zc.values.reshape(-1, 1))[:, 1]
    calib(ac2, f"Platt fitted on validation "
               f"(slope {pl.coef_[0][0]:+.3f}, intercept {pl.intercept_[0]:+.3f})")

    print("\n" + "=" * 74)
    print("SKILL AGAINST THE HAND BASELINE (unchanged by any of this)")
    print("=" * 74)
    y, mu, bs = cl.saves.values, cl.mu.values, cl.base.values
    lg = np.full(len(y), pr[pr.season.isin(SEASONS[:3])].saves.mean())
    for nm, p_ in (("model", mu), ("model, level corrected", cl.mu_adj.values),
                   ("his own recent average", bs),
                   ("league mean", lg)):
        print(f"  {nm:24} MAE {np.abs(y - p_).mean():.4f}  "
              f"R2 {1 - ((y - p_) ** 2).sum() / ((y - lg) ** 2).sum():+.4f}")


if __name__ == "__main__":
    main()
