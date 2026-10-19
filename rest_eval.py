"""Does a back-to-back actually move the result, and does the model price it?

Three separate questions, which a single "yes" would blur:
  1. Does playing the second night of a back-to-back lower a team's chance
     of winning, in the data?
  2. The model already carries rest, b2b, games-in-four and travel as
     features. Is the weight it puts on them the right weight -- i.e. is it
     calibrated inside these situations, or does it under- or over-rate a
     tired team?
  3. MODEL.md has long claimed back-to-backs are a spot where this model
     beats the market. That slice was chosen after seeing the holdout, so
     it needs re-testing against real posted prices.

Holdout is 2025-26 + 2026-27, which the xG fit never saw.
"""
import numpy as np
import pandas as pd

HOLDOUT = ("20252026", "20262027")   # season is stored as a string


def wilson(k, n, z=1.96):
    """Binomial interval that behaves at small n, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load():
    f = pd.read_pickle("data/features.pkl")
    f = f[["game_id", "season", "date", "home", "away", "home_win",
           "h_rest", "a_rest", "h_b2b", "a_b2b", "h_travel_km",
           "a_travel_km"]].copy()
    for c in ("h_b2b", "a_b2b"):
        f[c] = f[c].fillna(0).astype(int)
    for c in ("h_rest", "a_rest"):
        f[c] = f[c].fillna(3).clip(0, 10)
    return f


def line(label, k, n, base=None):
    if n == 0:
        print(f"  {label:<34} {'--':>7}")
        return
    p = k / n
    lo, hi = wilson(k, n)
    tail = ""
    if base is not None:
        tail = f"  {p - base:+6.1%} vs the rest"
    print(f"  {label:<34} {n:>6,}  {p:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]{tail}")


def raw_effect(f, tag):
    print(f"\n{'=' * 78}\nRAW: HOW OFTEN A TEAM ON THE SECOND NIGHT WINS  ({tag})\n{'=' * 78}")
    print(f"  {'situation':<34} {'n':>6}  {'win%':>6}  {'95%':>13}")
    # Home side
    m = f.h_b2b == 1
    base = f.loc[~m, "home_win"].mean()
    line("home on a back-to-back", f.loc[m, "home_win"].sum(), m.sum(), base)
    line("home rested", f.loc[~m, "home_win"].sum(), (~m).sum())
    # Away side: away wins when home_win == 0
    m = f.a_b2b == 1
    aw = 1 - f.home_win
    base = aw[~m].mean()
    line("away on a back-to-back", aw[m].sum(), m.sum(), base)
    line("away rested", aw[~m].sum(), (~m).sum())
    # The clean contrast: exactly one side is tired.
    only_h = (f.h_b2b == 1) & (f.a_b2b == 0)
    only_a = (f.a_b2b == 1) & (f.h_b2b == 0)
    both = (f.h_b2b == 1) & (f.a_b2b == 1)
    neither = (f.h_b2b == 0) & (f.a_b2b == 0)
    print()
    line("only the home side is tired", f.loc[only_h, "home_win"].sum(),
         only_h.sum())
    line("only the away side is tired", f.loc[only_a, "home_win"].sum(),
         only_a.sum())
    line("both tired", f.loc[both, "home_win"].sum(), both.sum())
    line("neither tired", f.loc[neither, "home_win"].sum(), neither.sum())
    print("  (these four are the HOME win rate, so the rested side is the"
          "\n   away team in the second row)")


def rest_gap(f, tag):
    print(f"\n{'=' * 78}\nBY REST DIFFERENCE, HOME MINUS AWAY  ({tag})\n{'=' * 78}")
    print(f"  {'gap':<34} {'n':>6}  {'home win%':>6}  {'95%':>13}")
    g = (f.h_rest - f.a_rest).clip(-3, 3)
    for v in sorted(g.unique()):
        m = g == v
        if m.sum() < 30:
            continue
        lab = f"home has {v:+d} days more rest" if v else "equal rest"
        line(lab, f.loc[m, "home_win"].sum(), m.sum())


def main():
    f = load()
    print(f"all seasons: {len(f):,} games   "
          f"holdout: {(f.season.isin(HOLDOUT)).sum():,}")
    raw_effect(f, "all seasons")
    raw_effect(f[f.season.isin(HOLDOUT)], "clean holdout only")
    rest_gap(f, "all seasons")

    # One logistic on the schedule alone: how many points of win
    # probability is a back-to-back actually worth, holding nothing else?
    print(f"\n{'=' * 78}\nTHE EFFECT AS ONE NUMBER\n{'=' * 78}")
    from sklearn.linear_model import LogisticRegression
    X = pd.DataFrame({
        "b2b_gap": f.h_b2b - f.a_b2b,
        "rest_gap": (f.h_rest - f.a_rest).clip(-3, 3),
    })
    y = f.home_win.values
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X.values, y)
    base = m.intercept_[0]
    for i, c in enumerate(X.columns):
        b = m.coef_[0][i]
        # what one unit does to the home side's probability at the mean
        p0 = 1 / (1 + np.exp(-base))
        p1 = 1 / (1 + np.exp(-(base + b)))
        print(f"  {c:<12} coefficient {b:+.4f}  "
              f"= {p1 - p0:+.1%} on the home win probability per unit")
    print(f"  home-ice baseline with neither tired and equal rest: "
          f"{1 / (1 + np.exp(-base)):.1%}")


if __name__ == "__main__":
    main()


def model_view():
    """Does the model's own number already contain the effect?

    A feature being in the matrix is not the same as the model getting the
    size right. The test is calibration inside each situation: take every
    holdout game where one side was on a back-to-back and compare what the
    model said would happen with what did.
    """
    f = load()
    b = pd.read_pickle("data/backtest_preds.pkl")[
        ["game_id", "season", "p_home", "home_win"]]
    d = b.merge(f.drop(columns=["season", "home_win"]), on="game_id")
    d = d[d.season.isin(HOLDOUT)]
    print(f"\n{'=' * 78}\nDOES THE MODEL ALREADY PRICE IT? (clean holdout, "
          f"n={len(d):,})\n{'=' * 78}")
    print(f"  {'situation':<30} {'n':>5} {'model says':>11} {'actual':>8} "
          f"{'error':>8} {'95% band':>16}")
    cases = [
        ("only the home side is tired", (d.h_b2b == 1) & (d.a_b2b == 0)),
        ("only the away side is tired", (d.a_b2b == 1) & (d.h_b2b == 0)),
        ("both tired", (d.h_b2b == 1) & (d.a_b2b == 1)),
        ("neither tired", (d.h_b2b == 0) & (d.a_b2b == 0)),
        ("either side tired", (d.h_b2b == 1) | (d.a_b2b == 1)),
    ]
    for label, m in cases:
        q = d[m]
        if len(q) < 20:
            print(f"  {label:<30} {len(q):>5}  too few")
            continue
        said, act = q.p_home.mean(), q.home_win.mean()
        lo, hi = wilson(q.home_win.sum(), len(q))
        print(f"  {label:<30} {len(q):>5} {said:>11.1%} {act:>8.1%} "
              f"{act - said:>+8.1%}  [{lo:>5.1%},{hi:>5.1%}]")
    # Accuracy and Brier inside the tired games, against the base rate, so
    # "the model is good on back-to-backs" can be checked rather than said.
    print()
    for label, m in cases[:2] + [cases[4]]:
        q = d[m]
        if len(q) < 20:
            continue
        pick = (q.p_home > 0.5).astype(int)
        acc = (pick == q.home_win).mean()
        br = ((q.p_home - q.home_win) ** 2).mean()
        base = q.home_win.mean()
        br0 = ((base - q.home_win) ** 2).mean()
        print(f"  {label:<30} n={len(q):>4}  accuracy {acc:>5.1%}  "
              f"brier {br:.4f}  skill vs that group's own base rate "
              f"{1 - br / br0:+6.2%}")


def vs_market():
    """The standing claim: back-to-backs are a spot where this beats the
    market. The slice was picked after seeing the holdout, so it is a lead,
    not a result -- this is the re-test against real posted prices."""
    o = pd.read_pickle("data/odds_joined.pkl")
    f = load()[["game_id", "h_b2b", "a_b2b", "h_rest", "a_rest"]]
    d = o.merge(f, on="game_id")
    print(f"\n{'=' * 78}\nAGAINST REAL POSTED PRICES (n={len(d):,})\n{'=' * 78}")
    print(f"  {'situation':<28} {'n':>5} {'model LL':>9} {'market LL':>10} "
          f"{'model acc':>10} {'flat ROI':>9}")
    y = d.home_win.values

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d["ll_model"], d["ll_mkt"] = ll(d.p_home.values), ll(d.mkt_h.values)
    cases = [
        ("either side tired", (d.h_b2b == 1) | (d.a_b2b == 1)),
        ("only the away side tired", (d.a_b2b == 1) & (d.h_b2b == 0)),
        ("only the home side tired", (d.h_b2b == 1) & (d.a_b2b == 0)),
        ("neither tired", (d.h_b2b == 0) & (d.a_b2b == 0)),
    ]
    for label, m in cases:
        q = d[m]
        if len(q) < 20:
            print(f"  {label:<28} {len(q):>5}  too few")
            continue
        pick_home = q.p_home > 0.5
        acc = (pick_home == (q.home_win == 1)).mean()
        # one unit on the model's side at the posted price
        dec = np.where(pick_home, q.dec_h, q.dec_a)
        won = np.where(pick_home, q.home_win == 1, q.home_win == 0)
        roi = (np.where(won, dec - 1, -1)).mean()
        print(f"  {label:<28} {len(q):>5} {q.ll_model.mean():>9.4f} "
              f"{q.ll_mkt.mean():>10.4f} {acc:>10.1%} {roi:>+9.1%}")
    print("\n  (flat ROI is one unit on the model's side at the posted price;"
          "\n   the whole-holdout figure is -7.3%)")


model_view()
vs_market()


def is_it_real():
    """Both of the interesting-looking results above need an interval.

    The model beating the market's log loss on tired games, and a positive
    ROI on away back-to-backs, are each a slice chosen because it looked
    good. A paired bootstrap over games is the honest test: resample whole
    games, recompute the difference, and see whether zero is inside.
    """
    rng = np.random.default_rng(20261019)
    o = pd.read_pickle("data/odds_joined.pkl")
    f = load()[["game_id", "h_b2b", "a_b2b"]]
    d = o.merge(f, on="game_id")
    y = d.home_win.values

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d["dll"] = ll(d.p_home.values) - ll(d.mkt_h.values)   # negative = model better
    pick_home = (d.p_home > 0.5).values
    dec = np.where(pick_home, d.dec_h.values, d.dec_a.values)
    won = np.where(pick_home, y == 1, y == 0)
    d["unit"] = np.where(won, dec - 1, -1.0)

    cases = [
        ("either side tired", ((d.h_b2b == 1) | (d.a_b2b == 1)).values),
        ("only the away side tired", ((d.a_b2b == 1) & (d.h_b2b == 0)).values),
        ("neither tired", ((d.h_b2b == 0) & (d.a_b2b == 0)).values),
        ("every holdout game", np.ones(len(d), bool)),
    ]
    print(f"\n{'=' * 78}\nIS EITHER OF THOSE REAL? (4,000 resamples over games)"
          f"\n{'=' * 78}")
    print(f"  {'situation':<26} {'n':>5} "
          f"{'model-market log loss':>24} {'flat ROI':>22}")
    for label, m in cases:
        dll, unit = d.dll.values[m], d.unit.values[m]
        n = len(dll)
        if n < 20:
            continue
        idx = rng.integers(0, n, size=(4000, n))
        bd = dll[idx].mean(1)
        bu = unit[idx].mean(1)
        print(f"  {label:<26} {n:>5} "
              f"{dll.mean():>+8.4f} [{np.percentile(bd, 2.5):+.4f},"
              f"{np.percentile(bd, 97.5):+.4f}] "
              f"{unit.mean():>+7.1%} [{np.percentile(bu, 2.5):+.1%},"
              f"{np.percentile(bu, 97.5):+.1%}]")
    print("\n  negative log-loss difference = the model is better than the "
          "market\n  an interval that crosses zero is not a result")


is_it_real()
