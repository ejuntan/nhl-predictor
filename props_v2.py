"""Shots model, second version: corrected mean and mean-dependent dispersion.

Two faults were measured in the first version, both in the upper range:

  mean  -- the top decile of predicted shots came in at 2.879 against 2.752
           predicted, a significant miss. Regularisation pulls high-volume
           shooters toward the pack.
  spread -- that same decile implies a negative-binomial size near 11.8
           where the single fitted value was 16.5, so the right tail was
           priced too thin.

Both corrections are fit inside each walk-forward fold, on training data
only, so nothing here sees the season it is scored on.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

import props_model as P

SEASONS = P.SEASONS
VALID = ["20232024", "20242025"]
TEST = ["20252026", "20262027"]
LINES = P.LINES
SIZE_LO, SIZE_HI = 3.0, 120.0


def fit_mean_correction(y, mu):
    """Two-parameter Poisson GLM of outcome on log(predicted mean).

    A slope above one stretches the top of the range upward, which is where
    the first version lost the mean.
    """
    z = np.log(np.clip(mu, 1e-3, None)).reshape(-1, 1)
    m = PoissonRegressor(alpha=1e-8, max_iter=5000).fit(z, y)
    return m


def apply_mean_correction(m, mu):
    z = np.log(np.clip(mu, 1e-3, None)).reshape(-1, 1)
    return m.predict(z)


def fit_size_curve(y, mu, bins=12):
    """Negative-binomial size as a function of the mean.

    Within bins of the predicted mean, the observed variance implies a size
    of mu^2 / (var - mu). Fitting log(size) on log(mu) across those bins
    gives a smooth curve instead of one global value.
    """
    d = pd.DataFrame({"y": y, "mu": mu})
    d["b"] = pd.qcut(d.mu, bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(
        n=("y", "size"), m=("mu", "mean"), v=("y", "var")).reset_index()
    g = g[(g.v > g.m) & (g.n >= 200)]
    if len(g) < 4:
        return None
    g["size"] = (g.m ** 2 / (g.v - g.m)).clip(SIZE_LO, SIZE_HI)
    X = np.c_[np.ones(len(g)), np.log(g.m.values)]
    w = np.sqrt(g.n.values)
    coef, *_ = np.linalg.lstsq(X * w[:, None], np.log(g["size"].values) * w,
                               rcond=None)
    return coef


def size_for(coef, mu, fallback):
    if coef is None:
        return np.full(len(mu), fallback)
    s = np.exp(coef[0] + coef[1] * np.log(np.clip(mu, 1e-3, None)))
    return np.clip(s, SIZE_LO, SIZE_HI)


def run(pg, correct_mean=True, vary_size=True):
    out = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = pg[pg.season.isin(SEASONS[:i])], pg[pg.season == s]
        if len(te) < 200 or len(tr) < 2000:
            continue
        Ftr, otr = P.feature_frame(tr)
        Fte, ote = P.feature_frame(te)
        mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu_) / sd_, (Fte.values - mu_) / sd_
        w = np.exp(otr)
        m = PoissonRegressor(alpha=P.ALPHA, max_iter=8000).fit(
            Ztr, tr.sog.values / w, sample_weight=w)
        raw_tr = m.predict(Ztr) * w
        raw_te = m.predict(Zte) * np.exp(ote)

        if correct_mean:
            mc = fit_mean_correction(tr.sog.values, raw_tr)
            adj_tr = apply_mean_correction(mc, raw_tr)
            adj_te = apply_mean_correction(mc, raw_te)
        else:
            adj_tr, adj_te = raw_tr, raw_te

        flat = P.fit_nb_dispersion(tr.sog.values, adj_tr)
        coef = fit_size_curve(tr.sog.values, adj_tr) if vary_size else None

        o = te[["game_id", "player_id", "season", "date", "team", "opp",
                "sog", "e_sog", "toi_reg"]].copy()
        o["mu"] = adj_te
        o["mu_raw"] = raw_te
        o["size"] = size_for(coef, adj_te, flat)
        o["size_flat"] = flat
        out.append(o)
    return pd.concat(out, ignore_index=True)


def over(mu, size, line):
    k = int(np.floor(line))
    return 1 - nbinom.cdf(k, size, size / (size + mu))


def report(a, label, seasons):
    from sklearn.metrics import brier_score_loss
    cl = a[a.season.isin(seasons)]
    y, mu, size = cl.sog.values, cl.mu.values, cl["size"].values
    base = cl.e_sog.values
    r2 = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    row = {"variant": label, "R2": round(r2, 4)}
    for L in LINES:
        hit = (y > L).astype(int)
        pm = np.clip(over(mu, size, L), 1e-6, 1 - 1e-6)
        pb = np.clip(over(base, cl.size_flat.values, L), 1e-6, 1 - 1e-6)
        pl = np.full(len(hit), hit.mean())
        row[f"o{L}_naive%"] = round(100 * (1 - brier_score_loss(hit, pm)
                                           / brier_score_loss(hit, pl)), 2)
        row[f"o{L}_own%"] = round(100 * (1 - brier_score_loss(hit, pm)
                                         / brier_score_loss(hit, pb)), 2)
    return row


def calib(a, seasons, line=2.5):
    cl = a[a.season.isin(seasons)]
    p = over(cl.mu.values, cl["size"].values, line)
    d = pd.DataFrame({"p": p, "y": (cl.sog.values > line).astype(int)})
    d["b"] = pd.cut(d.p, [0, .15, .25, .35, .45, .55, .7, 1.0])
    g = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                          obs=("y", "mean"))
    g["moe95"] = 1.96 * np.sqrt(g.obs * (1 - g.obs) / g.n)
    g["ok"] = (g.pred - g.obs).abs() <= g.moe95
    return g


# ---------------------------------------------------------------------------
# Direct line models
#
# Deriving P(over) from a fitted count distribution assumes the count really
# is negative binomial. It is not: a low-minutes skater is a mixture of "sat
# most of the night" and "played a normal shift", which inflates variance
# while concentrating mass at zero. A negative binomial has to put that extra
# variance in the right tail instead, which is exactly the observed fault --
# too much tail at the bottom of the range, too little at the top.
#
# Fitting each line as its own classifier drops the distributional assumption.
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression


def run_direct(pg, C=1.0, use_mu=True):
    """One logistic model per line, on the same features as the count model."""
    out = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = pg[pg.season.isin(SEASONS[:i])], pg[pg.season == s]
        if len(te) < 200 or len(tr) < 2000:
            continue
        Ftr, otr = P.feature_frame(tr)
        Fte, ote = P.feature_frame(te)

        # The count model's own prediction is a useful single summary of the
        # features, so it is offered to the classifier as one more column.
        if use_mu:
            mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
            w = np.exp(otr)
            pm = PoissonRegressor(alpha=P.ALPHA, max_iter=8000).fit(
                (Ftr.values - mu_) / sd_, tr.sog.values / w, sample_weight=w)
            mtr = pm.predict((Ftr.values - mu_) / sd_) * w
            mte = pm.predict((Fte.values - mu_) / sd_) * np.exp(ote)
            Ftr = Ftr.assign(l_mu=np.log(np.clip(mtr, 1e-3, None)))
            Fte = Fte.assign(l_mu=np.log(np.clip(mte, 1e-3, None)))
        else:
            mtr = mte = None

        Ftr = Ftr.assign(l_exposure=otr)
        Fte = Fte.assign(l_exposure=ote)
        mu2, sd2 = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu2) / sd2, (Fte.values - mu2) / sd2

        o = te[["game_id", "player_id", "season", "date", "team", "opp",
                "sog", "e_sog", "toi_reg"]].copy()
        if mte is not None:
            o["mu"] = mte
        for L in LINES:
            ytr = (tr.sog.values > L).astype(int)
            clf = LogisticRegression(C=C, max_iter=4000).fit(Ztr, ytr)
            o[f"d{L}"] = clf.predict_proba(Zte)[:, 1]
        out.append(o)
    return pd.concat(out, ignore_index=True)


def report_direct(a, label, seasons, flat_size=16.5):
    from sklearn.metrics import brier_score_loss, log_loss
    cl = a[a.season.isin(seasons)]
    y = cl.sog.values
    row = {"variant": label}
    for L in LINES:
        hit = (y > L).astype(int)
        pm = np.clip(cl[f"d{L}"].values, 1e-6, 1 - 1e-6)
        pb = np.clip(over(cl.e_sog.values, flat_size, L), 1e-6, 1 - 1e-6)
        pl = np.full(len(hit), hit.mean())
        row[f"o{L}_naive%"] = round(100 * (1 - brier_score_loss(hit, pm)
                                           / brier_score_loss(hit, pl)), 2)
        row[f"o{L}_own%"] = round(100 * (1 - brier_score_loss(hit, pm)
                                         / brier_score_loss(hit, pb)), 2)
        row[f"o{L}_ll"] = round(log_loss(hit, pm), 5)
    return row


def calib_direct(a, seasons, line=2.5):
    cl = a[a.season.isin(seasons)]
    d = pd.DataFrame({"p": cl[f"d{line}"].values,
                      "y": (cl.sog.values > line).astype(int)})
    d["b"] = pd.cut(d.p, [0, .15, .25, .35, .45, .55, .7, 1.0])
    g = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                          obs=("y", "mean"))
    g["moe95"] = 1.96 * np.sqrt(g.obs * (1 - g.obs) / g.n)
    g["ok"] = (g.pred - g.obs).abs() <= g.moe95
    return g


def run_sharpened(pg, C=1.0):
    """Direct line models with a temperature fit on a held-out season.

    A maximum-likelihood logistic fit is calibrated on the data it saw, so a
    temperature fit there would come out at 1 and do nothing. The last
    training season is held back instead: the lines are fit on everything
    before it, the temperature is fit on it, and both are then applied to the
    test season. Nothing touches the season being scored.
    """
    out = []
    for i, s in enumerate(SEASONS):
        if i < 2:
            continue                      # needs a fit set and a calib set
        fit_seasons = SEASONS[:i - 1]
        calib_season = SEASONS[i - 1]
        tr = pg[pg.season.isin(fit_seasons)]
        ca = pg[pg.season == calib_season]
        te = pg[pg.season == s]
        if len(te) < 200 or len(tr) < 2000 or len(ca) < 2000:
            continue

        def design(d):
            F, off = P.feature_frame(d)
            return F.assign(l_exposure=off)

        Ftr, Fca, Fte = design(tr), design(ca), design(te)
        mu2, sd2 = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr = (Ftr.values - mu2) / sd2
        Zca = (Fca.values - mu2) / sd2
        Zte = (Fte.values - mu2) / sd2

        o = te[["game_id", "player_id", "season", "date", "team", "opp",
                "sog", "e_sog", "toi_reg"]].copy()
        for L in LINES:
            clf = LogisticRegression(C=C, max_iter=4000).fit(
                Ztr, (tr.sog.values > L).astype(int))
            zc = clf.decision_function(Zca)
            yc = (ca.sog.values > L).astype(int)
            best_t, best_ll = 1.0, np.inf
            for t in np.arange(0.55, 1.81, 0.025):
                q = 1 / (1 + np.exp(-zc / t))
                q = np.clip(q, 1e-9, 1 - 1e-9)
                ll = -np.mean(yc * np.log(q) + (1 - yc) * np.log(1 - q))
                if ll < best_ll:
                    best_t, best_ll = t, ll
            zt = clf.decision_function(Zte)
            o[f"d{L}"] = 1 / (1 + np.exp(-zt / best_t))
            o[f"t{L}"] = best_t
        out.append(o)
    return pd.concat(out, ignore_index=True)
