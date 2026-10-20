"""Who wins the overtime? Measuring the component nobody had tested.

A win probability here is regulation plus overtime:

    p_home = p_reg_home + p_reg_tie * p_ot_home

so a side behind on expected regulation goals can still come out ahead if
the model likes it in overtime. That happens on about 9% of games, and it
is only defensible if `p_ot_home` -- a logistic fitted on the games that
actually went to overtime -- knows something.

Section 18 measured `p_ot`, the chance a game REACHES overtime. This is the
other one: who wins once it is there. It had never been tested.
"""
import numpy as np
import pandas as pd

HOLDOUT = ("20252026", "20262027")


def main():
    b = pd.read_pickle("data/backtest_preds.pkl")
    tr = b[~b.season.isin(HOLDOUT)]
    ho = b[b.season.isin(HOLDOUT)]
    ot = ho[ho.went_ot == 1]
    y, p = ot.home_win.values, ot.p_ot_home.values
    base = y.mean()

    print("=" * 72)
    print(f"THE OVERTIME WINNER, ON {len(ot):,} HOLDOUT OVERTIMES "
          f"({len(ot) / len(ho):.1%} of games)")
    print("=" * 72)
    br = ((p - y) ** 2).mean()
    br0 = ((base - y) ** 2).mean()
    print(f"  home actually won      {base:>7.1%}")
    print(f"  model said on average  {p.mean():>7.1%}")
    print(f"  accuracy               {((p > .5) == (y == 1)).mean():>7.1%}")
    print(f"  brier {br:.4f} against the base rate's {br0:.4f} "
          f"= {1 - br / br0:+.2%} skill")

    rng = np.random.default_rng(7)
    n = len(ot)
    idx = rng.integers(0, n, (4000, n))
    sk = 1 - (((p[idx] - y[idx]) ** 2).mean(1)
              / ((y[idx].mean(1)[:, None] - y[idx]) ** 2).mean(1))
    print(f"  95% interval on that skill "
          f"[{np.percentile(sk, 2.5):+.2%}, {np.percentile(sk, 97.5):+.2%}]")
    print(f"\n  and it is not a timid component: p_ot_home runs "
          f"{ho.p_ot_home.min():.2f} to {ho.p_ot_home.max():.2f}, "
          f"5-95% {ho.p_ot_home.quantile(.05):.2f}-"
          f"{ho.p_ot_home.quantile(.95):.2f}")

    # What replacing it would do to the win probability itself. The constant
    # is the training seasons' rate, so nothing is fitted on the holdout.
    c = tr.loc[tr.went_ot == 1, "home_win"].mean()
    yy = ho.home_win.values
    print("\n" + "=" * 72)
    print(f"REPLACING IT WITH THE TRAINING RATE ({c:.3f}), "
          f"ON ALL {len(ho):,} HOLDOUT GAMES")
    print("=" * 72)

    def show(pp, label):
        q = np.clip(pp, 1e-9, 1 - 1e-9)
        ll = -(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean()
        print(f"  {label:<32} logloss {ll:.5f}  "
              f"brier {((q - yy) ** 2).mean():.5f}  "
              f"acc {((q > .5) == (yy == 1)).mean():.4f}")
        return q

    a = show(ho.p_home.values, "as shipped")
    bb = show(ho.p_reg_home.values + ho.p_reg_tie.values * c, "constant instead")

    def lls(q):
        return -(yy * np.log(q) + (1 - yy) * np.log(1 - q))

    d = lls(bb) - lls(a)
    idx = rng.integers(0, len(ho), (4000, len(ho)))
    db = d[idx].mean(1)
    print(f"\n  constant minus shipped: {d.mean():+.5f} "
          f"[{np.percentile(db, 2.5):+.5f}, {np.percentile(db, 97.5):+.5f}]")
    m = (a > .5) != (bb > .5)
    print(f"  the pick changes on {m.sum()} of {len(ho):,} games ({m.mean():.1%}); "
          f"shipped right {(((a > .5) == (yy == 1))[m]).mean():.1%}, "
          f"constant right {(((bb > .5) == (yy == 1))[m]).mean():.1%}")
    print("\n  So: the component is established to be worse than a constant,")
    print("  but swapping it is not an established improvement to the win")
    print("  probability -- overtime decides too few games for the effect to")
    print("  clear its own interval, and accuracy moves the other way.")


if __name__ == "__main__":
    main()
