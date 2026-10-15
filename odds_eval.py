"""A genuine comparison against sportsbook prices.

ESPN records a DraftKings line per completed game. That is the last line
ESPN held, at or near the close -- not verifiably the exact closing number,
so this is labelled "recorded line" rather than "closing line" throughout.

Three things are measured:
  1. Accuracy: log loss, Brier and calibration for the model against the
     market's no-vig probabilities, on exactly the same games.
  2. Agreement: where the two disagree, and who is right when they do.
  3. Profitability: betting the model into the actual posted prices, which
     is where the transaction cost lives.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

TEST = ["20252026", "20262027"]


def american_to_dec(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / np.abs(a))


def load():
    nhl = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    od = sqlite3.connect("data/odds.sqlite3", timeout=60)
    ev = pd.read_sql("SELECT * FROM espn_events", od)
    odds = pd.read_sql("SELECT * FROM odds", od)
    e = ev.merge(odds, on="event_id", how="inner")
    e = e[e.home_ml.notna() & e.away_ml.notna()]

    bt = pd.read_pickle("data/backtest_preds.pkl")
    bt = bt[bt.season.isin(TEST)]
    # ESPN's calendar date can differ by a day for late starts, so try both.
    j = bt.merge(e, on=["date", "home", "away"], how="inner")
    missed = bt[~bt.game_id.isin(j.game_id)]
    if len(missed):
        e2 = e.copy()
        e2["date"] = (pd.to_datetime(e2.date) - pd.Timedelta(days=1)
                      ).dt.strftime("%Y-%m-%d")
        j2 = missed.merge(e2, on=["date", "home", "away"], how="inner")
        j = pd.concat([j, j2], ignore_index=True)
    return bt, j


def devig_proportional(dh, da):
    ih, ia = 1 / dh, 1 / da
    s = ih + ia
    return ih / s, ia / s, s - 1


def devig_logarithmic(dh, da):
    """Odds-ratio (logarithmic) de-vig, per game, as a robustness check.

    Finds the exponent k with implied_home^k + implied_away^k = 1. Unlike the
    proportional method this does not assume the vig is spread evenly across
    both sides, which matters on lopsided prices.
    """
    ih, ia = 1 / np.asarray(dh, float), 1 / np.asarray(da, float)
    lo = np.full(ih.shape, 0.5)
    hi = np.full(ih.shape, 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        too_big = ih ** k + ia ** k > 1
        lo = np.where(too_big, k, lo)
        hi = np.where(too_big, hi, k)
    k = (lo + hi) / 2
    return ih ** k, ia ** k


def main():
    bt, j = load()
    print(f"holdout games: {len(bt)}   matched to a recorded line: {len(j)}"
          f"  ({len(j)/len(bt):.1%})")
    print(f"providers: {j.provider.value_counts().to_dict()}")

    dh = american_to_dec(j.home_ml.values)
    da = american_to_dec(j.away_ml.values)
    mh, ma, over = devig_proportional(dh, da)
    lh, la = devig_logarithmic(dh, da)
    j = j.assign(dec_h=dh, dec_a=da, mkt_h=mh, mkt_a=ma, overround=over,
                 mkt_h_log=lh)
    y = j.home_win.values
    print(f"median overround (two-way hold): {np.median(over):.4f}")
    print(f"market home-win rate implied {mh.mean():.4f}, actual {y.mean():.4f}")

    print("\n" + "=" * 72)
    print("1. ACCURACY on the same games")
    print("=" * 72)
    rows = []
    for lab, p in (("model", j.p_home.values),
                   ("model + recalibration", j.p_home_cal.values),
                   ("market, no-vig (proportional)", mh),
                   ("market, no-vig (logarithmic)", lh),
                   ("market, raw implied (with vig)", 1 / dh),
                   ("always home", np.full(len(y), y.mean()))):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        rows.append({"source": lab, "n": len(y),
                     "acc": round(float(((pc > .5) == (y == 1)).mean()), 4),
                     "brier": round(brier_score_loss(y, pc), 5),
                     "logloss": round(log_loss(y, pc), 5)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    mdl = t[t.source == "model"].iloc[0]
    mkt = t[t.source == "market, no-vig (proportional)"].iloc[0]
    print(f"\nmarket is better by {mdl.logloss - mkt.logloss:+.5f} log loss, "
          f"{mdl.brier - mkt.brier:+.5f} Brier")

    # paired test on the same games
    def ll(yy, pp):
        pp = np.clip(pp, 1e-6, 1 - 1e-6)
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    d = ll(y, j.p_home.values) - ll(y, mh)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"paired: model minus market = {d.mean():+.5f}  SE {se:.5f}  "
          f"t = {d.mean()/se:+.2f}  (positive means the market wins)")

    print("\n" + "=" * 72)
    print("2. CALIBRATION, model against market")
    print("=" * 72)
    bins = [0, .35, .45, .5, .55, .6, .65, .7, 1.0]
    for lab, p in (("model", j.p_home.values), ("market", mh)):
        d2 = pd.DataFrame({"p": p, "y": y})
        d2["b"] = pd.cut(d2.p, bins)
        g = d2.groupby("b", observed=True).agg(
            n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
        g["moe"] = 1.96 * np.sqrt(g.obs * (1 - g.obs) / g.n)
        g["ok"] = (g.pred - g.obs).abs() <= g.moe
        print(f"\n  {lab}: {int(g.ok.sum())}/{len(g)} bands inside 95%")
        print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 72)
    print("3. WHERE THEY DISAGREE")
    print("=" * 72)
    j["disagree"] = (j.p_home - mh)
    for lo, hi in ((0, .03), (.03, .06), (.06, .10), (.10, 1.0)):
        m = j[j.disagree.abs().between(lo, hi)]
        if len(m) < 25:
            continue
        ym = m.home_win.values
        mk = m.mkt_h.values
        md = m.p_home.values
        # when they pick different sides, who is right?
        split = m[(md > .5) != (mk > .5)]
        print(f"  gap {lo:.2f}-{hi:.2f}: n={len(m):4d}  "
              f"model brier {brier_score_loss(ym, md):.4f}  "
              f"market brier {brier_score_loss(ym, mk):.4f}"
              + (f"   opposite sides n={len(split)}, model right "
                 f"{(((split.p_home>.5)==(split.home_win==1)).mean()):.3f}"
                 if len(split) >= 15 else ""))

    print("\n" + "=" * 72)
    print("4. PROFITABILITY at the actual posted prices")
    print("=" * 72)
    print("  threshold = how much the model's probability must exceed the")
    print("  price's raw implied probability before a bet is placed.\n")
    res = []
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        picks = []
        for side, p, dec, won in (("home", j.p_home.values, dh, y),
                                  ("away", j.p_away.values, da, 1 - y)):
            edge = p - 1 / dec
            m = edge > thr
            if m.sum():
                stake = m.sum()
                ret = np.where(won[m] == 1, dec[m] - 1, -1.0).sum()
                picks.append((stake, ret))
        n = sum(s for s, _ in picks)
        pl = sum(r for _, r in picks)
        res.append({"threshold": f"{thr:.0%}", "bets": int(n),
                    "units_won": round(float(pl), 2),
                    "roi_%": round(100 * pl / n, 2) if n else np.nan})
    print(pd.DataFrame(res).to_string(index=False))
    print("\n  A fair-coin bettor loses the hold, about "
          f"{100*np.median(over)/2:.1f}% per bet, on average.")
    j.to_pickle("data/odds_joined.pkl")


if __name__ == "__main__":
    main()


def profitability_significance(j, thresholds=(0.0, 0.01, 0.02, 0.03, 0.05, 0.08)):
    """ROI with a confidence interval, and the monotonicity check.

    A positive ROI on a few hundred bets means little on its own: the
    per-bet standard deviation is close to one unit, so the standard error of
    the mean is roughly 1/sqrt(n) -- about 4.5% on 500 bets.

    The sharper test is the shape. A model with real edge should earn *more*
    per bet as the threshold rises, because it is selecting its stronger
    opinions. If ROI falls as the threshold rises, the large apparent edges
    are the model's own errors rather than market mistakes.
    """
    dh = american_to_dec(j.home_ml.values)
    da = american_to_dec(j.away_ml.values)
    y = j.home_win.values
    rng = np.random.default_rng(11)
    rows = []
    for thr in thresholds:
        pl = []
        for p, dec, won in ((j.p_home.values, dh, y),
                            (j.p_away.values, da, 1 - y)):
            m = (p - 1 / dec) > thr
            if m.sum():
                pl.append(np.where(won[m] == 1, dec[m] - 1, -1.0))
        if not pl:
            continue
        r = np.concatenate(pl)
        n = len(r)
        se = r.std(ddof=1) / np.sqrt(n)
        boot = np.array([r[rng.integers(0, n, n)].mean() for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"threshold": f"{thr:.0%}", "bets": n,
                     "roi_%": round(100 * r.mean(), 2),
                     "se_%": round(100 * se, 2),
                     "t": round(r.mean() / se, 2),
                     "ci95_%": f"[{100*lo:+.2f}, {100*hi:+.2f}]",
                     "beats_zero": "yes" if lo > 0 else "no"})
    return pd.DataFrame(rows)


def side_disagreement(j):
    """When the model and the market pick different sides, who is right?"""
    dh = american_to_dec(j.home_ml.values)
    da = american_to_dec(j.away_ml.values)
    mh, _, _ = devig_proportional(dh, da)
    model_home = j.p_home.values > 0.5
    mkt_home = mh > 0.5
    opp = model_home != mkt_home
    y = j.home_win.values
    n = int(opp.sum())
    right = int((model_home[opp] == (y[opp] == 1)).sum())
    se = np.sqrt(0.25 / n)
    return {"games_on_opposite_sides": n, "model_right": right,
            "model_win_rate": round(right / n, 4),
            "se": round(float(se), 4),
            "z_vs_coinflip": round((right / n - 0.5) / se, 2)}
