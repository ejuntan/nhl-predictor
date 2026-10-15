"""Do referees move the total? Measure the spread before modelling it.

If crews genuinely differ, the spread of their long-run averages has to
exceed what pure sampling noise would produce from the same number of
games. That test costs nothing and settles whether a feature is worth
building.

Then, if there is anything there, the crew's causal history goes into the
direct totals model as extra features and is scored on the clean holdout.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from model import build_matrix

DB = "data/nhl.sqlite3"
SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
CLEAN = ["20252026", "20262027"]
HL = 40.0        # games of referee history, half-life


def load():
    con = sqlite3.connect(DB, timeout=60)
    o = pd.read_sql("SELECT game_id, name FROM officials WHERE role='referee'",
                    con)
    g = pd.read_sql("""SELECT game_id, date, season, home_score, away_score
                       FROM games WHERE game_type=2 AND state IN ('OFF','FINAL')""",
                    con)
    g["total"] = g.home_score + g.away_score
    return o.merge(g, on="game_id").sort_values(["date", "game_id"])


def spread(d):
    print("=" * 74)
    print("1. DO REFEREES DIFFER MORE THAN CHANCE WOULD PRODUCE?")
    print("=" * 74)
    by = d.groupby("name").total.agg(["size", "mean"])
    by = by[by["size"] >= 100]
    obs = by["mean"].std(ddof=1)
    # Expected spread of sample means if every crew were identical.
    within = d.total.std(ddof=1)
    exp = np.sqrt((within ** 2 * (1 / by["size"])).mean())
    print(f"  {len(by)} referees with 100+ games; "
          f"game totals sd {within:.3f}")
    print(f"  observed spread of their averages : {obs:.4f}")
    print(f"  spread chance alone would produce : {exp:.4f}")
    print(f"  excess attributable to the referee: "
          f"{max(obs ** 2 - exp ** 2, 0) ** .5:.4f} goals")
    f, p = stats.f_oneway(*[g.total.values for _, g in d.groupby("name")
                            if len(g) >= 100])
    print(f"  one-way ANOVA across crews: F={f:.3f}  p={p:.3f}")
    print("\n  highest and lowest by average total:")
    s = by.sort_values("mean")
    for nm, r in pd.concat([s.head(3), s.tail(3)]).iterrows():
        print(f"    {nm:22} {int(r['size']):4d} games   {r['mean']:.3f}")
    return obs, exp


def ref_features(d):
    """Causal: each referee's trailing average total, before this game."""
    alpha = 1 - 0.5 ** (1 / HL)
    state, rows = {}, []
    for r in d.itertuples(index=False):
        s = state.get(r.name)
        rows.append({"game_id": r.game_id, "name": r.name,
                     "ref_hist": s["v"] if s else np.nan,
                     "ref_n": s["n"] if s else 0})
        state[r.name] = ({"v": r.total, "n": 1} if s is None else
                         {"v": (1 - alpha) * s["v"] + alpha * r.total,
                          "n": s["n"] + 1})
    f = pd.DataFrame(rows)
    f = f[f.ref_n >= 20]
    return f.groupby("game_id").agg(crew_hist=("ref_hist", "mean"),
                                    crew_n=("ref_n", "min")).reset_index()


def predictive(crew):
    print("\n" + "=" * 74)
    print("2. DOES THE CREW'S HISTORY PREDICT THE TOTAL?")
    print("=" * 74)
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].merge(crew, on="game_id", how="inner")
    df = df.dropna(subset=["crew_hist"])
    print(f"  games with a crew history: {len(df)}")

    out = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
        if len(te) < 30 or len(tr) < 500:
            continue
        for nm, extra in (("without the crew", []), ("with the crew", ["crew_hist"])):
            Xtr = build_matrix(tr).copy()
            Xte = build_matrix(te).copy()
            for c in extra:
                Xtr[c] = tr[c].values
                Xte[c] = te[c].values
            mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-9
            r = Ridge(alpha=50.0).fit((Xtr.values - mu) / sd,
                                      tr.total_goals.values)
            out.append(pd.DataFrame({
                "season": s, "variant": nm, "game_id": te.game_id.values,
                "y": te.total_goals.values,
                "p": r.predict((Xte.values - mu) / sd)}))
    o = pd.concat(out, ignore_index=True)
    cl = o[o.season.isin(CLEAN)]
    base = cl[cl.variant == "without the crew"].set_index("game_id")
    with_ = cl[cl.variant == "with the crew"].set_index("game_id")
    y = base.y.values
    mean = np.full(len(y), y.mean())
    print(f"\n  clean holdout, n={len(y)}")
    print(f"  {'':22} {'MAE':>8} {'R2':>10}")
    for nm, p in (("without the crew", base.p.values),
                  ("with the crew", with_.p.values),
                  ("the league average", mean)):
        mae = np.abs(y - p).mean()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - mean) ** 2).sum()
        print(f"  {nm:22} {mae:8.4f} {r2:+10.5f}")
    t, pv = stats.ttest_rel(np.abs(y - with_.p.values),
                            np.abs(y - base.p.values))
    print(f"\n  paired t, with vs without: t={t:+.2f}  p={pv:.3f}")

    c = np.corrcoef(with_.join(base[[]], how="left").index.map(
        lambda _: 0) if False else base.p.values, y)[0, 1]
    print(f"  correlation of the crew's own history with the total: "
          f"{np.corrcoef(df.crew_hist, df.total_goals)[0, 1]:+.4f}")


if __name__ == "__main__":
    d = load()
    spread(d)
    predictive(ref_features(d))
