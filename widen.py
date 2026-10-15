"""The one actionable finding from step 2: overconfidence on goalie changes.

Step 2 found the model claims 59.7% and delivers 56.0% when an unusual
goalie starts -- t=-2.16, the same sign in three of four seasons, and the
market handles those games better by 0.004 log loss. That is a *variance*
fault, not a mean fault: the earlier test (section 11) swapped the confirmed
starter's rating in and changed nothing, because knowing which goalie plays
does not tell you how he plays. Not knowing should widen the distribution
instead of moving it.

Three corrections are fitted on 2023-24 and 2024-25 and applied unchanged to
the clean holdout:

  group temperature   logit(p) / T, with a separate T where the starter is unusual
  interacted logistic c0 + c1.logit(p) + c2.unusual + c3.unusual.logit(p)
  continuous          temperature as a function of how unusual the starter is

The indicator uses the goalie who *actually* started, which is not known
until the announcement. This therefore measures the ceiling of a starter
feed used this way, not a deployable gain.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from compare import paired_bootstrap

FIT = ["20232024", "20242025"]
TEST = ["20252026", "20262027"]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def unusual(d):
    """1 where either side's starter holds under 35% of recent starts."""
    return ((d.h_backup.fillna(0) > 0) | (d.a_backup.fillna(0) > 0)).astype(float).values


def oddity(d):
    """Continuous: how far the less-established starter is below a split."""
    s = np.minimum(d.h_share.fillna(.5), d.a_share.fillna(.5)).values
    return np.clip(0.5 - s, 0, 0.5) / 0.5


def fit_group_temp(z, y, u):
    def nll(th):
        t = np.exp(th[0] + th[1] * u)
        q = np.clip(sig(z / t), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead")
    return r.x


def apply_group_temp(th, z, u):
    return sig(z / np.exp(th[0] + th[1] * u))


def metrics(y, p, name):
    p = np.clip(p, EPS, 1 - EPS)
    return {"variant": name, "n": len(y),
            "acc": ((p > .5) == (y == 1)).mean(),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def main():
    d = pd.read_pickle("data/ref_preds_all.pkl")
    tr = d[d.season.isin(FIT)].copy()
    te = d[d.season.isin(TEST)].copy()
    ztr, zte = logit(tr.p.values), logit(te.p.values)
    utr, ute = unusual(tr), unusual(te)
    otr, ote = oddity(tr), oddity(te)
    ytr, yte = tr.y.values, te.y.values
    print(f"fit on {len(tr)} games ({utr.mean():.1%} with an unusual "
          f"starter), test on {len(te)} ({ute.mean():.1%})")

    th = fit_group_temp(ztr, ytr, utr)
    print(f"\ngroup temperature: T = {np.exp(th[0]):.3f} normally, "
          f"{np.exp(th[0] + th[1]):.3f} when the starter is unusual")
    p_gt = apply_group_temp(th, zte, ute)

    X = np.c_[ztr, utr, utr * ztr]
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X, ytr)
    print("interacted logistic coefficients "
          f"logit={lr.coef_[0][0]:+.3f} unusual={lr.coef_[0][1]:+.3f} "
          f"interaction={lr.coef_[0][2]:+.3f}")
    p_lr = lr.predict_proba(np.c_[zte, ute, ute * zte])[:, 1]

    th2 = fit_group_temp(ztr, ytr, otr)
    print(f"continuous: T = {np.exp(th2[0]):.3f} at a settled starter, "
          f"{np.exp(th2[0] + th2[1] * 1.0):.3f} at the extreme")
    p_ct = apply_group_temp(th2, zte, ote)

    rows = [metrics(yte, te.p.values, "shipped"),
            metrics(yte, p_gt, "group temperature"),
            metrics(yte, p_lr, "interacted logistic"),
            metrics(yte, p_ct, "continuous temperature")]
    t = pd.DataFrame(rows)
    base = t.iloc[0]
    t["d_logloss"] = t.logloss - base.logloss
    print("\nCLEAN HOLDOUT")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\nPaired bootstrap vs shipped:")
    for nm, p in (("group temperature", p_gt), ("interacted logistic", p_lr),
                  ("continuous temperature", p_ct)):
        r = paired_bootstrap(yte, p, te.p.values)
        v = r["logloss"]
        print(f"  {nm:<24} {v['delta']:+.5f}  "
              f"CI [{v['lo']:+.5f}, {v['hi']:+.5f}]  "
              f"P(better) {v['p_better']:.3f}")

    m = ute > 0
    print(f"\nOn the {int(m.sum())} holdout games with an unusual starter:")
    for nm, p in (("shipped", te.p.values), ("group temperature", p_gt),
                  ("interacted logistic", p_lr),
                  ("continuous temperature", p_ct)):
        r = metrics(yte[m], np.asarray(p)[m], nm)
        conf = np.maximum(np.asarray(p)[m], 1 - np.asarray(p)[m])
        print(f"  {nm:<24} acc {r['acc']:.4f}  claimed {conf.mean():.4f}  "
              f"gap {r['acc'] - conf.mean():+.4f}  ll {r['logloss']:.5f}")


if __name__ == "__main__":
    main()
