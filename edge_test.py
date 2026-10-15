"""Was there ever a pocket of games worth betting?

Section 13 found the model beats the market on strong favourites (-0.0091
log loss on 183 games) and loses badly when it backs the market's underdog
(+0.0115 on 229). Log loss is not money, so this prices both: bet at the
actual posted American price, pay the actual vig, and see what comes back.

Splitting by whether the model and the market agree on the side, and by how
confident the model is. 1,414 games with prices.
"""
import numpy as np
import pandas as pd
from scipy import stats

from odds_eval import american_to_dec, devig_logarithmic


def roi_ci(profit, b=20000, seed=0):
    """Bootstrap CI on mean profit per $1 staked."""
    rng = np.random.default_rng(seed)
    n = len(profit)
    if n < 10:
        return np.nan, np.nan
    draws = profit[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return np.quantile(draws, .025), np.quantile(draws, .975)


def main():
    j = pd.read_pickle("data/odds_joined.pkl").copy()
    dh = american_to_dec(j.home_ml.values)
    da = american_to_dec(j.away_ml.values)
    mh, ma = devig_logarithmic(dh, da)
    j["mkt_h"], j["dec_h"], j["dec_a"] = mh, dh, da

    # The model's own side, and what it would cost to back it.
    back_home = j.p_home.values > .5
    p_model = np.where(back_home, j.p_home.values, 1 - j.p_home.values)
    p_mkt = np.where(back_home, mh, ma)
    dec = np.where(back_home, dh, da)
    won = np.where(back_home, j.home_win.values == 1, j.home_win.values == 0)
    profit = np.where(won, dec - 1.0, -1.0)
    agree = (p_mkt > .5)                       # market favours the same side
    edge = p_model - 1.0 / dec                 # edge against the vigged price

    j = j.assign(p_model=p_model, p_mkt=p_mkt, dec=dec, won=won,
                 profit=profit, agree=agree, edge=edge)

    print("=" * 84)
    print("BACKING THE MODEL'S SIDE AT THE POSTED PRICE, by situation")
    print("n, win rate, what the model claimed, ROI per $1, 95% CI")
    print("=" * 84)
    rows = []
    bands = [(.5, .55), (.55, .60), (.60, .65), (.65, 1.01)]
    for lab, m in (("model agrees with the market", j.agree),
                   ("model backs the market's dog", ~j.agree)):
        for lo, hi in bands:
            s = j[m & (j.p_model >= lo) & (j.p_model < hi)]
            if len(s) < 25:
                continue
            ci = roi_ci(s.profit.values)
            rows.append({"situation": lab, "conf": f"{lo:.2f}-{hi:.2f}",
                         "n": len(s), "win%": s.won.mean(),
                         "claimed": s.p_model.mean(), "ROI": s.profit.mean(),
                         "lo": ci[0], "hi": ci[1]})
        s = j[m]
        ci = roi_ci(s.profit.values)
        rows.append({"situation": lab, "conf": "all", "n": len(s),
                     "win%": s.won.mean(), "claimed": s.p_model.mean(),
                     "ROI": s.profit.mean(), "lo": ci[0], "hi": ci[1]})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, formatters={
        "win%": "{:.3f}".format, "claimed": "{:.3f}".format,
        "ROI": "{:+.4f}".format, "lo": "{:+.4f}".format, "hi": "{:+.4f}".format}))

    print("\n" + "=" * 84)
    print("ONLY BETTING WHEN THE MODEL SHOWS AN EDGE OVER THE POSTED PRICE")
    print("=" * 84)
    rows = []
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        s = j[j.edge >= thr]
        if len(s) < 10:
            continue
        ci = roi_ci(s.profit.values)
        t_, p_ = stats.ttest_1samp(s.profit.values, 0)
        rows.append({"min edge": f"{thr:.0%}", "bets": len(s),
                     "win%": s.won.mean(), "ROI": s.profit.mean(),
                     "lo": ci[0], "hi": ci[1], "t": t_, "p": p_})
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "win%": "{:.3f}".format, "ROI": "{:+.4f}".format,
        "lo": "{:+.4f}".format, "hi": "{:+.4f}".format,
        "t": "{:+.2f}".format, "p": "{:.3f}".format}))

    print("\n" + "=" * 84)
    print("THE SAME, RESTRICTED TO STRONG FAVOURITES THE MARKET AGREES WITH")
    print("(the one pocket where the model beat the market on log loss)")
    print("=" * 84)
    fav = j[j.agree & (j.p_model >= .60)]
    rows = []
    for thr in (0.0, 0.01, 0.02, 0.03):
        s = fav[fav.edge >= thr]
        if len(s) < 10:
            continue
        ci = roi_ci(s.profit.values)
        rows.append({"min edge": f"{thr:.0%}", "bets": len(s),
                     "win%": s.won.mean(), "claimed": s.p_model.mean(),
                     "ROI": s.profit.mean(), "lo": ci[0], "hi": ci[1]})
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "win%": "{:.3f}".format, "claimed": "{:.3f}".format,
        "ROI": "{:+.4f}".format, "lo": "{:+.4f}".format, "hi": "{:+.4f}".format}))

    print("\n" + "=" * 84)
    print("TOTALS, for completeness: backing the model's side of the posted line")
    print("=" * 84)
    k = j[j.total.notna() & j.over_odds.notna() & j.under_odds.notna()].copy()
    # The model's probability of going over the book's actual line.
    lam = k.lam_h.values + k.lam_a.values
    line = k.total.values
    # Read P(over) off the model's own total distribution via its score grid
    # proxy: Poisson on the expected total is close enough for a line check.
    from scipy.stats import poisson
    p_over = 1 - poisson.cdf(np.floor(line), lam)
    do = american_to_dec(k.over_odds.values)
    du = american_to_dec(k.under_odds.values)
    take_over = p_over > .5
    pm = np.where(take_over, p_over, 1 - p_over)
    dd = np.where(take_over, do, du)
    hit = np.where(take_over, k.total_goals.values > line,
                   k.total_goals.values < line)
    push = k.total_goals.values == line
    pf = np.where(push, 0.0, np.where(hit, dd - 1.0, -1.0))
    eg = pm - 1.0 / dd
    rows = []
    for thr in (0.0, 0.02, 0.04, 0.06):
        m = eg >= thr
        if m.sum() < 10:
            continue
        ci = roi_ci(pf[m])
        rows.append({"min edge": f"{thr:.0%}", "bets": int(m.sum()),
                     "hit%": hit[m & ~push].mean(), "ROI": pf[m].mean(),
                     "lo": ci[0], "hi": ci[1]})
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "hit%": "{:.3f}".format, "ROI": "{:+.4f}".format,
        "lo": "{:+.4f}".format, "hi": "{:+.4f}".format}))


if __name__ == "__main__":
    main()
