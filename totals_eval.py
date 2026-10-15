"""Are the game-total estimates actually any good?

The page prints an expected total on every card, and the backtest reports
calibration at three fixed lines. Neither says whether the number is
*accurate*, because calibration only asks whether a stated probability
comes true at the stated rate -- a forecast that always says "6.2 goals"
can be perfectly calibrated and carry no information at all.

So this asks three harder questions on the clean holdout:

  1. Does the expected total beat simply guessing the league average?
  2. Does it beat the number the sportsbook posted for the same game?
  3. When it disagrees with that number, is it right?

The third is the one that matters, and it is the same test the moneyline
failed. The MLB model in this account failed it too: its total MAE was
3.607 against the market's 3.527.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson

from odds_eval import american_to_dec, devig_logarithmic

CLEAN = ["20252026", "20262027"]


def skill(y, pred, base):
    """Mean absolute error, root mean squared error, and R2 vs a baseline."""
    mae = np.abs(y - pred).mean()
    rmse = np.sqrt(((y - pred) ** 2).mean())
    ss = 1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum()
    return mae, rmse, ss


def main():
    bt = pd.read_pickle("data/backtest_preds.pkl")
    cl = bt[bt.season.isin(CLEAN)].copy()
    y = cl.total_goals.values.astype(float)
    pred = cl.exp_total.values
    mean = np.full(len(y), y.mean())

    print("=" * 72)
    print(f"POINT ACCURACY ON THE CLEAN HOLDOUT  (n={len(y)} games)")
    print("=" * 72)
    print(f"actual totals      mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}")
    print(f"model's estimate   mean {pred.mean():.3f}  sd {pred.std(ddof=1):.3f}")
    print(f"  the model's estimates span {pred.min():.2f} to {pred.max():.2f}; "
          f"real games span {y.min():.0f} to {y.max():.0f}")
    m = skill(y, pred, mean)
    b = skill(y, mean, mean)
    print(f"\n{'':22} {'MAE':>7} {'RMSE':>7} {'R2 vs mean':>11}")
    print(f"{'model':22} {m[0]:7.3f} {m[1]:7.3f} {m[2]:11.4f}")
    print(f"{'always the average':22} {b[0]:7.3f} {b[1]:7.3f} {b[2]:11.4f}")
    t, p = stats.ttest_rel(np.abs(y - pred), np.abs(y - mean))
    print(f"\npaired t on absolute error, model vs the average: "
          f"t={t:+.2f}  p={p:.3f}")

    # 2. Against the sportsbook's posted total.
    od = pd.read_pickle("data/odds_joined.pkl")[
        ["game_id", "total", "over_odds", "under_odds"]]
    j = cl.merge(od, on="game_id", how="inner")
    j = j[j.total.notna()].copy()
    yy = j.total_goals.values.astype(float)
    mk = j.total.values.astype(float)
    mo = j.exp_total.values
    mm = np.full(len(yy), yy.mean())
    print("\n" + "=" * 72)
    print(f"AGAINST THE POSTED TOTAL  (n={len(j)} games with a line)")
    print("=" * 72)
    a, bk = skill(yy, mo, mm), skill(yy, mk, mm)
    print(f"{'':22} {'MAE':>7} {'RMSE':>7} {'R2 vs mean':>11}")
    print(f"{'model':22} {a[0]:7.3f} {a[1]:7.3f} {a[2]:11.4f}")
    print(f"{'posted line':22} {bk[0]:7.3f} {bk[1]:7.3f} {bk[2]:11.4f}")
    t2, p2 = stats.ttest_rel(np.abs(yy - mo), np.abs(yy - mk))
    print(f"\npaired t on absolute error, model vs the line: "
          f"t={t2:+.2f}  p={p2:.3f}  "
          f"({'model closer' if a[0] < bk[0] else 'line closer'})")
    print(f"correlation between the two estimates: "
          f"{np.corrcoef(mo, mk)[0, 1]:.3f}")
    print(f"model minus line: mean {np.mean(mo - mk):+.3f}  "
          f"sd {np.std(mo - mk, ddof=1):.3f}")

    # 3. Does disagreement carry information?
    print("\n" + "=" * 72)
    print("WHEN THE MODEL DISAGREES WITH THE LINE, IS IT RIGHT?")
    print("A push is excluded; the line is the book's own number.")
    print("=" * 72)
    d = mo - mk
    live = yy != mk
    rows = []
    for lo, hi, lab in ((0.00, 0.25, "under 0.25"), (0.25, 0.50, "0.25-0.50"),
                        (0.50, 10.0, "over 0.50")):
        for side, nm in ((d > 0, "model says over"), (d < 0, "model says under")):
            m_ = side & (np.abs(d) >= lo) & (np.abs(d) < hi) & live
            if m_.sum() < 20:
                continue
            went_over = (yy[m_] > mk[m_]).mean()
            right = went_over if nm == "model says over" else 1 - went_over
            se = np.sqrt(right * (1 - right) / m_.sum())
            rows.append({"disagreement": lab, "direction": nm,
                         "n": int(m_.sum()), "model right": right,
                         "95% lo": right - 1.96 * se,
                         "95% hi": right + 1.96 * se})
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "model right": "{:.3f}".format, "95% lo": "{:.3f}".format,
        "95% hi": "{:.3f}".format}))

    all_dis = live & (np.abs(d) > 0)
    over_called = d > 0
    right_all = np.where(over_called, yy > mk, yy < mk)[all_dis]
    se = np.sqrt(right_all.mean() * (1 - right_all.mean()) / all_dis.sum())
    print(f"\nevery non-push game: model right {right_all.mean():.4f} "
          f"on {all_dis.sum()} games  "
          f"95% CI [{right_all.mean()-1.96*se:.3f}, "
          f"{right_all.mean()+1.96*se:.3f}]  (break-even ~0.524 at -110)")

    # 4. Calibration at the book's own line, using the model's real
    # probability for that line rather than a Poisson approximation of it.
    # The stored distribution carries 4.5, 5.5 and 6.5, which is most of
    # the board; other lines are left out rather than approximated.
    print("\n" + "=" * 72)
    print("IS THE MODEL'S P(OVER) CALIBRATED AT THE BOOK'S OWN LINE?")
    print("Only games whose posted line is one the model prices directly.")
    print("=" * 72)
    cols = {4.5: "p_over_45", 5.5: "p_over_55", 6.5: "p_over_65"}
    k = j[j.total.isin(cols)].copy()
    k["p"] = [r[cols[t]] for t, r in zip(k.total, k.to_dict("records"))]
    k["over"] = (k.total_goals > k.total).astype(float)
    k = k[k.total_goals != k.total]
    print(f"{'line':>6} {'n':>5} {'predicted':>10} {'actual':>8} {'gap':>8}")
    for t_, grp in k.groupby("total"):
        se = np.sqrt(grp.over.mean() * (1 - grp.over.mean()) / len(grp))
        flag = "" if abs(grp.p.mean() - grp.over.mean()) < 1.96 * se else "  <- outside the MOE"
        print(f"{t_:>6} {len(grp):>5} {grp.p.mean():>10.4f} "
              f"{grp.over.mean():>8.4f} {grp.p.mean() - grp.over.mean():>+8.4f}{flag}")
    dd = k.copy()
    dd["bin"] = pd.cut(dd.p, [0, .4, .45, .5, .55, .6, 1.0])
    g = dd.groupby("bin", observed=True).agg(
        n=("over", "size"), predicted=("p", "mean"), actual=("over", "mean"))
    g["gap"] = g.actual - g.predicted
    print("\nby probability band:")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 72)
    print("AT THE PAGE'S OWN FIXED LINES")
    print("=" * 72)
    for line, col in ((4.5, "p_over_45"), (5.5, "p_over_55"),
                      (6.5, "p_over_65")):
        act = (cl.total_goals > line).mean()
        pr = cl[col].mean()
        n = len(cl)
        se = np.sqrt(act * (1 - act) / n)
        flag = "" if abs(pr - act) < 1.96 * se else "   <- outside the MOE"
        print(f"  over {line}: predicted {pr:.4f}  actual {act:.4f}  "
              f"gap {pr-act:+.4f}  (+/-{1.96*se:.4f}){flag}")


if __name__ == "__main__":
    main()
