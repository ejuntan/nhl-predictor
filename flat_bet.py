"""Flat one unit on every prediction. What actually comes back.

Real posted prices exist here only for the moneyline -- the ESPN feed
carries moneyline, total and spread, and no prop prices at all. So this
splits into two honest halves:

  measured    the moneyline, bet at the actual price the book posted on
              each of 1,414 games
  conditional the props, where no price was ever observed, so the result is
              stated as a function of the book's hold rather than invented

For the props the question that matters is not "what would I have won" but
"how much vig can this edge carry". That is answerable without price data:
the model's probability and the realised rate are both known, so the
break-even price is known, and the only unknown is what a book would have
charged above it.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from odds_eval import american_to_dec, devig_logarithmic

CLEAN = ["20252026", "20262027"]


def bankroll(profit, label):
    n = len(profit)
    tot = profit.sum()
    roi = profit.mean()
    sd = profit.std(ddof=1)
    se = sd / np.sqrt(n)
    print(f"  {label:34} {n:6d} bets  {tot:+8.1f}u  "
          f"ROI {roi:+.2%}  95% CI [{roi-1.96*se:+.2%}, {roi+1.96*se:+.2%}]")
    return tot


def moneyline():
    j = pd.read_pickle("data/odds_joined.pkl").copy()
    dh = american_to_dec(j.home_ml.values)
    da = american_to_dec(j.away_ml.values)
    mh, ma = devig_logarithmic(dh, da)
    back_home = j.p_home.values > .5
    dec = np.where(back_home, dh, da)
    won = np.where(back_home, j.home_win.values == 1, j.home_win.values == 0)
    profit = np.where(won, dec - 1.0, -1.0)
    pm = np.where(back_home, j.p_home.values, 1 - j.p_home.values)
    pk = np.where(back_home, mh, ma)
    edge = pm - 1 / dec

    print("=" * 86)
    print("MONEYLINE -- real posted prices, 1,414 games")
    print("=" * 86)
    bankroll(profit, "every game")
    bankroll(profit[edge > 0], "only where the model shows an edge")
    bankroll(profit[(pk > .5) & (pm < .60)], "the situation measured to lose")
    bankroll(profit[(pk > .5) & (pm >= .60)], "strong favourites, book agrees")
    bankroll(profit[pk < .495], "backing the book's underdog")
    hold = (1 / dh + 1 / da - 1).mean()
    print(f"\n  the book's median hold on these games: {hold:.2%}")
    print(f"  a bettor with no edge at all loses that, slowly")


def prop_breakeven(p_model, y, label, holds=(0.0, 0.02, 0.04, 0.06, 0.08)):
    """What a market with this much skill returns at a given book hold.

    A book that priced every line at the league base rate is the softest it
    could plausibly be; one that priced it at the model's own number is the
    hardest. The truth is between, so both bounds are reported.
    """
    p_model = np.clip(np.asarray(p_model, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    base = y.mean()
    take = p_model > base            # bet the side the model likes
    print(f"\n  {label}  (n={len(y)}, base rate {base:.3f})")
    print(f"    {'book hold':>10} {'vs a lazy book':>18} "
          f"{'vs a sharp book':>18}")
    for h in holds:
        # Lazy: the line is priced at the base rate, plus the hold.
        d_lazy = np.where(take, 1 / (base * (1 + h)), 1 / ((1 - base) * (1 + h)))
        hit = np.where(take, y == 1, y == 0)
        lazy = np.where(hit, d_lazy - 1, -1.0).mean()
        # Sharp: the line is priced at the truth, plus the hold. The truth
        # is the realised rate in this model's own probability band.
        bins = pd.qcut(pd.Series(p_model), 10, duplicates="drop")
        truth = pd.Series(y).groupby(bins, observed=True).transform("mean").values
        d_sharp = np.where(take, 1 / (truth * (1 + h)),
                           1 / ((1 - truth) * (1 + h)))
        sharp = np.where(hit, d_sharp - 1, -1.0).mean()
        print(f"    {h:>9.0%} {lazy:>17.2%} {sharp:>18.2%}")


def props():
    print("\n" + "=" * 86)
    print("PROPS -- no price was ever observed, so this is a function of the hold")
    print("=" * 86)
    pr = pd.read_pickle("data/props_preds.pkl")
    cl = pr[pr.season.isin(CLEAN)]
    for L in (1.5, 2.5):
        prop_breakeven(cl[f"p_over_{L}"], (cl.sog > L).astype(int),
                       f"shots over {L}")
    sv = pd.read_pickle("data/saves_preds_v2.pkl")
    sc = sv[sv.season.isin(CLEAN)]
    p = 1 - nbinom.cdf(27, sc.size_oos,
                       sc.size_oos / (sc.size_oos + sc.mu * sc.mult))
    prop_breakeven(p, (sc.saves > 27.5).astype(int), "saves over 27.5")


if __name__ == "__main__":
    moneyline()
    props()
