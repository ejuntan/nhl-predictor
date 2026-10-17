"""Which player props are worth putting on a shortlist, and does the rule work?

The page can rank props, but only if "best" means something measured. There
are no posted prop prices in the payload, so edge against a book cannot be
computed. What CAN be computed is the model's one established prop claim:
it beats a player's own recent average. This asks whether that advantage is
concentrated where the model departs furthest from that average -- which is
the only defensible way to pick fifteen props out of fifteen hundred.

Holdout only (2025-26 + 2026-27): the xG fit never saw these seasons.
"""
import numpy as np
import pandas as pd

HOLDOUT = ("20252026", "20262027")   # season is stored as a string
LINES = [("1.5", 1.5), ("2.5", 2.5), ("3.5", 3.5)]


def load():
    d = pd.read_pickle("data/props_preds.pkl")
    d = d[d.season.isin(HOLDOUT)].copy()
    rows = []
    for tag, ln in LINES:
        c = d[["date", "player_id", "team", "opp", "sog"]].copy()
        c["line"] = ln
        c["p"] = d["p_over_" + tag].values
        c["base"] = d["base_over_" + tag].values
        c["y"] = (d.sog.values > ln).astype(int)
        rows.append(c)
    a = pd.concat(rows, ignore_index=True)
    # The model's side relative to the naive read, and how far it departs.
    a["gap"] = a.p - a.base
    a["side_over"] = (a.gap > 0).astype(int)
    # Probability each one assigns to the side the model is taking.
    a["p_side"] = np.where(a.side_over == 1, a.p, 1 - a.p)
    a["b_side"] = np.where(a.side_over == 1, a.base, 1 - a.base)
    a["hit"] = np.where(a.side_over == 1, a.y, 1 - a.y)
    return a


def brier_skill(p, y, ref):
    return 1 - np.mean((p - y) ** 2) / np.mean((ref - y) ** 2)


def breakeven(p):
    """American price at which a bet at probability p is exactly even."""
    if not (0 < p < 1):
        return None
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def main():
    a = load()
    print(f"holdout prop-lines: {len(a):,} over {a.date.nunique()} days\n")

    print("=" * 74)
    print("BY HOW FAR THE MODEL DEPARTS FROM THE PLAYER'S OWN AVERAGE")
    print("=" * 74)
    print("the model's side = the side it takes against that average\n")
    edges = [0, .02, .05, .08, .12, 1]
    print(f"{'gap':>12} {'n':>7} {'model says':>11} {'baseline':>9} "
          f"{'actual':>8} {'calib err':>10} {'skill vs base':>14}")
    g = a.gap.abs()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = a[(g >= lo) & (g < hi)]
        if len(m) < 50:
            continue
        sk = brier_skill(m.p.values, m.y.values, m.base.values)
        print(f"{lo:>6.0%}-{hi:<5.0%} {len(m):>7,} {m.p_side.mean():>10.1%} "
              f"{m.b_side.mean():>9.1%} {m.hit.mean():>8.1%} "
              f"{m.hit.mean() - m.p_side.mean():>+10.1%} {sk:>+13.2%}")

    print("\n" + "=" * 74)
    print("THE SHORTLIST RULE ITSELF: top N by gap, each day")
    print("=" * 74)
    for n in (5, 10, 15, 25, 50):
        picks = (a.sort_values("gap", key=lambda s: s.abs(), ascending=False)
                  .groupby("date").head(n))
        be = breakeven(picks.hit.mean())
        print(f"  top {n:>2}/day  n={len(picks):>6,}  model says "
              f"{picks.p_side.mean():>5.1%}  actual {picks.hit.mean():>5.1%}  "
              f"(+-{1.96 * np.sqrt(picks.hit.mean() * (1 - picks.hit.mean())
                                   / len(picks)):.1%})  "
              f"breaks even at {be:+d}")

    print("\n" + "=" * 74)
    print("SAME RULE, BUT RANKED BY RAW CONFIDENCE INSTEAD (|p - 50%|)")
    print("=" * 74)
    b = a.copy()
    b["side_over"] = (b.p > .5).astype(int)
    b["p_side"] = np.where(b.side_over == 1, b.p, 1 - b.p)
    b["hit"] = np.where(b.side_over == 1, b.y, 1 - b.y)
    b["conf"] = (b.p - .5).abs()
    for n in (15, 50):
        picks = (b.sort_values("conf", ascending=False)
                  .groupby("date").head(n))
        print(f"  top {n:>2}/day  n={len(picks):>6,}  model says "
              f"{picks.p_side.mean():>5.1%}  actual {picks.hit.mean():>5.1%}  "
              f"breaks even at {breakeven(picks.hit.mean()):+d}")

    print("\n" + "=" * 74)
    print("BY LINE, top 15/day by gap")
    print("=" * 74)
    picks = (a.sort_values("gap", key=lambda s: s.abs(), ascending=False)
              .groupby("date").head(15))
    for ln in (1.5, 2.5, 3.5):
        m = picks[picks.line == ln]
        if not len(m):
            continue
        print(f"  over {ln}  n={len(m):>5,}  model {m.p_side.mean():>5.1%}  "
              f"actual {m.hit.mean():>5.1%}  "
              f"over-side {m.side_over.mean():>5.1%} of the time")


if __name__ == "__main__":
    main()


def refine():
    """The gap rule alone takes sides the model itself rates under 50%.
    A prop you would actually put money on has to clear both bars."""
    a = load()
    print("\n" + "=" * 74)
    print("GAP ALONE TAKES SIDES THE MODEL DOES NOT EVEN FAVOUR")
    print("=" * 74)
    picks = (a.sort_values("gap", key=lambda s: s.abs(), ascending=False)
              .groupby("date").head(15))
    u = picks[picks.p_side < .5]
    print(f"  of the top 15/day, {len(u):,} of {len(picks):,} "
          f"({len(u)/len(picks):.0%}) are sides the model rates under 50%; "
          f"they came in {u.hit.mean():.1%} of the time")

    print("\n" + "=" * 74)
    print("ADDING A FLOOR ON THE MODEL'S OWN PROBABILITY")
    print("=" * 74)
    print(f"{'floor':>7} {'n':>7} {'model':>7} {'actual':>7} {'err':>7} "
          f"{'needs':>7} {'95% band':>16}")
    for floor in (.50, .55, .60, .65, .70):
        q = a[a.p_side >= floor]
        p = (q.sort_values("gap", key=lambda s: s.abs(), ascending=False)
              .groupby("date").head(15))
        if len(p) < 100:
            continue
        se = np.sqrt(p.hit.mean() * (1 - p.hit.mean()) / len(p))
        lo, hi = p.hit.mean() - 1.96 * se, p.hit.mean() + 1.96 * se
        print(f"{floor:>7.0%} {len(p):>7,} {p.p_side.mean():>7.1%} "
              f"{p.hit.mean():>7.1%} {p.hit.mean()-p.p_side.mean():>+7.1%} "
              f"{breakeven(p.hit.mean()):>+7d} "
              f"{breakeven(hi):>+6d} to {breakeven(lo):>+6d}")

    print("\n" + "=" * 74)
    print("CHOSEN RULE: p_side >= 55%, then the 15 biggest gaps, by line")
    print("=" * 74)
    q = a[a.p_side >= .55]
    p = (q.sort_values("gap", key=lambda s: s.abs(), ascending=False)
          .groupby("date").head(15))
    for ln in (1.5, 2.5, 3.5):
        m = p[p.line == ln]
        if not len(m):
            continue
        print(f"  over {ln}  n={len(m):>5,}  model {m.p_side.mean():>5.1%}  "
              f"actual {m.hit.mean():>5.1%}  needs {breakeven(m.hit.mean()):>+5d}"
              f"   over-side {m.side_over.mean():>5.1%}")
    print(f"\n  days with a full 15: "
          f"{(p.groupby('date').size() >= 15).sum()} of {p.date.nunique()}")
    print(f"  median per day: {p.groupby('date').size().median():.0f}")

    print("\n" + "=" * 74)
    print("DOES RANK WITHIN THE FIFTEEN MEAN ANYTHING?")
    print("=" * 74)
    p = p.copy()
    p["ag"] = p.gap.abs()
    p["rk"] = p.groupby("date")["ag"].rank(ascending=False)
    for lo, hi in ((1, 5), (6, 10), (11, 15)):
        m = p[(p.rk >= lo) & (p.rk <= hi)]
        print(f"  ranks {lo:>2}-{hi:<2}  n={len(m):>5,}  "
              f"model {m.p_side.mean():>5.1%}  actual {m.hit.mean():>5.1%}")


refine()


def correct():
    """The shortlist is over-confident by about two points. Is that a level
    shift that can simply be taken off, or does it vary with the number?"""
    a = load()
    q = a[a.p_side >= .55]
    p = (q.sort_values("gap", key=lambda s: s.abs(), ascending=False)
          .groupby("date").head(15)).copy()
    print("\n" + "=" * 74)
    print("CALIBRATION OF THE SHORTLIST, BY THE NUMBER THE MODEL QUOTES")
    print("=" * 74)
    print(f"{'model says':>13} {'n':>6} {'actual':>8} {'err':>7} {'needs':>7}")
    for lo, hi in ((.55, .62), (.62, .70), (.70, .78), (.78, .86), (.86, 1.01)):
        m = p[(p.p_side >= lo) & (p.p_side < hi)]
        if len(m) < 80:
            continue
        print(f"{lo:>6.0%}-{hi:<6.0%} {len(m):>6,} {m.hit.mean():>8.1%} "
              f"{m.hit.mean()-m.p_side.mean():>+7.1%} "
              f"{breakeven(m.hit.mean()):>+7d}")

    # One free parameter, fitted on the first half and checked on the second.
    days = sorted(p.date.unique())
    cut = days[len(days) // 2]
    tr, te = p[p.date < cut], p[p.date >= cut]
    k = te  # placeholder so the name exists if the fit degenerates
    shift = tr.hit.mean() - tr.p_side.mean()
    print(f"\n  shift fitted on {len(tr):,} earlier picks: {shift:+.3f}")
    print(f"  applied to {len(te):,} later picks: model "
          f"{te.p_side.mean():.1%}{shift:+.1%} = "
          f"{te.p_side.mean()+shift:.1%}, actual {te.hit.mean():.1%} "
          f"(err {te.hit.mean()-te.p_side.mean()-shift:+.1%}, "
          f"uncorrected {te.hit.mean()-te.p_side.mean():+.1%})")
    del k


correct()


def dedupe():
    """A player can qualify at two lines at once. Three unders on one man is
    one bet in three coupons, so test keeping only his biggest gap."""
    a = load()
    q = a[a.p_side >= .55].copy()
    q["ag"] = q.gap.abs()
    both = (q.sort_values("ag", ascending=False).groupby("date").head(15))
    one = (q.sort_values("ag", ascending=False)
            .groupby(["date", "player_id"]).head(1)
            .sort_values("ag", ascending=False).groupby("date").head(15))
    print("\n" + "=" * 74)
    print("ONE LINE PER PLAYER, OR LET A PLAYER REPEAT?")
    print("=" * 74)
    for nm, p in (("repeats allowed", both), ("one per player", one)):
        dup = len(p) - p.groupby(["date", "player_id"]).ngroups
        se = np.sqrt(p.hit.mean() * (1 - p.hit.mean()) / len(p))
        print(f"  {nm:<17} n={len(p):>5,}  model {p.p_side.mean():>5.1%}  "
              f"actual {p.hit.mean():>5.1%} (+-{1.96*se:.1%})  "
              f"needs {breakeven(p.hit.mean()):>+5d}  "
              f"duplicate players {dup:,}")
    print(f"\n  distinct players in a 15 (repeats allowed): median "
          f"{both.groupby('date').player_id.nunique().median():.0f}")


dedupe()
