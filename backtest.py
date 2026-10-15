"""Walk-forward backtest.

For each test season the model is refit on every prior season only. The xG
model underneath was fit on 2022-23..2024-25, so 2025-26 and 2026-27 are the
only seasons where *nothing* in the pipeline has seen the outcomes -- those
are the headline numbers. Earlier seasons are reported for completeness and
flagged.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.linear_model import LogisticRegression

from model import NHLModel, build_matrix, elo_ratings, log5_points_pct

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
CLEAN = {"20252026", "20262027"}   # untouched by the xG fit


def causal_temperature(p_raw, y, warmup=150, shrink_k=200.0):
    """Expanding-window temperature scaling, using only prior games.

    The model is fit on seasons whose team-strength spread may differ from
    the season being predicted; when the league compresses, raw probabilities
    come out too sharp. One temperature parameter, refit as the season goes
    and shrunk toward no-op while the sample is small, corrects that without
    any peeking: game N is only ever adjusted using games 1..N-1.
    """
    p_raw = np.clip(np.asarray(p_raw, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    z = np.log(p_raw / (1 - p_raw))
    out = np.empty_like(p_raw)
    for i in range(len(p_raw)):
        if i < warmup:
            out[i] = p_raw[i]
            continue
        zz, yy = z[:i], y[:i]
        best_t, best_ll = 1.0, np.inf
        for t in np.arange(0.6, 2.61, 0.05):
            q = 1 / (1 + np.exp(-zz / t))
            ll = -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))
            if ll < best_ll:
                best_t, best_ll = t, ll
        t_use = 1.0 + (i / (i + shrink_k)) * (best_t - 1.0)
        out[i] = 1 / (1 + np.exp(-z[i] / t_use))
    return out


def metrics(y, p, name):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {"model": name, "n": len(y), "acc": ((p > .5) == (y == 1)).mean(),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def calib_table(y, p, bins=(0, .35, .45, .5, .55, .6, .65, .7, 1.0)):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.cut(d.p, bins)
    return d.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))


def run(df):
    df = df[df.game_type == 2].copy()          # regular season only
    elo, _ = elo_ratings(df)
    lg5 = log5_points_pct(df)
    df = df.merge(elo, on="game_id").merge(lg5, on="game_id")

    preds = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr = df[df.season.isin(SEASONS[:i])]
        te = df[df.season == s]
        if len(te) < 30 or len(tr) < 500:
            continue
        m = NHLModel().fit(tr)

        # League scoring level drifts between seasons. Track it causally:
        # for each test game, compare actual to predicted regulation goals
        # over the most recent WINDOW completed games only.
        WINDOW = 250
        tail = tr.sort_values(["date", "game_id"]).tail(WINDOW)
        hist = pd.concat([tail, te.sort_values(["date", "game_id"])])
        lh_all, la_all = m.rates(hist)
        pred_tot = lh_all + la_all
        act_tot = (hist.h_reg + hist.a_reg).values
        mult, buf_p, buf_a = [], list(pred_tot[:len(tail)]), list(act_tot[:len(tail)])
        for j in range(len(tail), len(hist)):
            r = (np.mean(buf_a[-WINDOW:]) / np.mean(buf_p[-WINDOW:])
                 if buf_p else 1.0)
            mult.append(float(np.clip(r, 0.90, 1.10)))
            buf_p.append(pred_tot[j])
            buf_a.append(act_tot[j])
        # The league's rate of games finishing level drifts too, and kappa
        # is fit on the training seasons. Retune it every 50 games from the
        # trailing window, using only games already played.
        tie_act = (hist.reg_outcome == 1).astype(int).values
        bt = list(tie_act[:len(tail)])
        blh, bla = list(lh_all[:len(tail)]), list(la_all[:len(tail)])
        km, cur = [], 1.0
        for j in range(len(tail), len(hist)):
            if (j - len(tail)) % 50 == 0:
                cur = float(np.clip(m.solve_kappa_mult(
                    np.array(blh[-WINDOW:]), np.array(bla[-WINDOW:]),
                    float(np.mean(bt[-WINDOW:]))), 0.6, 1.8))
            km.append(cur)
            bt.append(tie_act[j])
            blh.append(lh_all[j])
            bla.append(la_all[j])

        te = te.sort_values(["date", "game_id"])
        pr = m.predict(te, scoring_mult=mult, kappa_mult=km)

        # Direct classifier on the same features, as an honest alternative
        # to reading the moneyline off the score distribution.
        Ztr = (build_matrix(tr).values - m.mu) / m.sd
        Zte = (build_matrix(te).values - m.mu) / m.sd
        clf = LogisticRegression(C=0.05, max_iter=4000).fit(
            Ztr, tr.home_win.values)

        out = te[["game_id", "season", "date", "home", "away", "home_win",
                  "total_goals", "went_ot", "h_reg", "a_reg", "home_score",
                  "away_score", "elo_p_home", "log5_p_home"]].copy()
        out["clf_p_home"] = clf.predict_proba(Zte)[:, 1]
        out["scoring_mult"] = mult
        for c in pr.columns:
            out[c] = pr[c].values
        out["p_home_cal"] = causal_temperature(out.p_home.values,
                                               out.home_win.values)
        out["n_train"] = len(tr)
        out["tau"] = m.tau
        out["kappa"] = m.kappa
        preds.append(out)
        print(f"fit<={SEASONS[i-1]} ({len(tr)} games) -> test {s} "
              f"({len(te)} games)  tau={m.tau:+.3f}  kappa={m.kappa:.3f}  "
              f"scoring_mult {np.mean(mult):.3f}", flush=True)
    return pd.concat(preds, ignore_index=True)


def report(p):
    print("\n" + "=" * 72)
    print("MONEYLINE (win probability), by season")
    print("=" * 72)
    rows = []
    for s, g in p.groupby("season"):
        tag = "CLEAN" if s in CLEAN else "xg-seen"
        for name, col in (("model", "p_home"),
                          ("model+recal", "p_home_cal"),
                          ("direct-logit", "clf_p_home"),
                          ("elo", "elo_p_home"),
                          ("log5(points%)", "log5_p_home")):
            r = metrics(g.home_win, g[col], name)
            r["season"], r["holdout"] = s, tag
            rows.append(r)
        r = metrics(g.home_win, np.full(len(g), g.home_win.mean()), "always-home")
        r["season"], r["holdout"] = s, tag
        rows.append(r)
    t = pd.DataFrame(rows)[["season", "holdout", "model", "n", "acc",
                            "brier", "logloss"]]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cl = p[p.season.isin(CLEAN)]
    print("\n" + "=" * 72)
    print(f"CLEAN HOLDOUT POOLED  (n={len(cl)} games, "
          f"{', '.join(sorted(cl.season.unique()))})")
    print("=" * 72)
    rows = [metrics(cl.home_win, cl.p_home, "model"),
            metrics(cl.home_win, cl.p_home_cal, "model+recal"),
            metrics(cl.home_win, cl.clf_p_home, "direct-logit"),
            metrics(cl.home_win, (cl.p_home + cl.clf_p_home) / 2, "blend"),
            metrics(cl.home_win, cl.elo_p_home, "elo"),
            metrics(cl.home_win, cl.log5_p_home, "log5(points%)"),
            metrics(cl.home_win, np.full(len(cl), 0.536), "always-home")]
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))

    print("\nCalibration, raw:")
    print(calib_table(cl.home_win.values, cl.p_home.values).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\nCalibration, after causal recalibration:")
    print(calib_table(cl.home_win.values, cl.p_home_cal.values).to_string(
        float_format=lambda v: f"{v:.4f}"))

    print("\nAccuracy by confidence band:")
    conf = np.maximum(cl.p_home, 1 - cl.p_home)
    hit = (cl.p_home > .5) == (cl.home_win == 1)
    d = pd.DataFrame({"conf": conf, "hit": hit})
    print(d.groupby(pd.cut(d.conf, [.5, .55, .6, .65, .7, .8, 1.0]),
                    observed=True).agg(n=("hit", "size"),
                                       acc=("hit", "mean")).to_string(
        float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 72)
    print("DERIVED MARKETS on the clean holdout")
    print("=" * 72)
    ot_act = cl.went_ot.mean()
    print(f"OT/SO rate     predicted {cl.p_ot.mean():.4f}  actual {ot_act:.4f}")
    print(f"total goals    predicted {cl.exp_total.mean():.3f}  "
          f"actual {cl.total_goals.mean():.3f}")
    for line, col in ((4.5, "p_over_45"), (5.5, "p_over_55"), (6.5, "p_over_65")):
        y = (cl.total_goals > line).astype(int)
        r = metrics(y, cl[col], f"over {line}")
        print(f"  over {line}: pred {cl[col].mean():.4f} actual {y.mean():.4f}"
              f"  brier {r['brier']:.4f}  logloss {r['logloss']:.4f}")
    y15 = ((cl.home_score - cl.away_score) >= 2).astype(int)
    r = metrics(y15, cl.p_home_m15, "home -1.5")
    print(f"  home -1.5: pred {cl.p_home_m15.mean():.4f} "
          f"actual {y15.mean():.4f}  brier {r['brier']:.4f}")

    print("\nRegulation three-way (the structurally honest target):")
    reg = np.select([cl.h_reg > cl.a_reg, cl.h_reg < cl.a_reg], [2, 0], 1)
    P = np.c_[cl.p_reg_away, cl.p_reg_tie, cl.p_reg_home]
    P = P / P.sum(1, keepdims=True)
    print(f"  3-way log loss {log_loss(reg, P, labels=[0,1,2]):.4f}  "
          f"(uniform = {np.log(3):.4f})")
    for i, nm in ((2, "home reg win"), (1, "reg tie"), (0, "away reg win")):
        print(f"  {nm:<13} pred {P[:, i].mean():.4f}  actual {(reg == i).mean():.4f}")
    return cl


if __name__ == "__main__":
    df = pd.read_pickle("data/features.pkl")
    p = run(df)
    p.to_pickle("data/backtest_preds.pkl")
    report(p)
