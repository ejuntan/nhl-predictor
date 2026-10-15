"""Game-level model: regulation goal rates -> full score distribution -> prices.

Structure follows the shape of hockey rather than fitting win/loss directly:

  1. Two Poisson rates for regulation goals (home, away).
  2. A joint regulation score matrix with a Dixon-Coles low-score correction.
  3. Regulation three-way probabilities; ties resolved by a small OT/SO model.

Everything downstream -- moneyline, puck line, totals -- is read off the same
score matrix, so the prices are mutually consistent by construction.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

MAXG = 12          # score matrix truncation; P(>12 goals in regulation) ~ 0


def _net(df, pre, stat):
    return df[f"{pre}_r_{stat}f_5v5"] - df[f"{pre}_r_{stat}a_5v5"]


def build_matrix(df):
    """Feature matrix shared by every component of the model."""
    x = pd.DataFrame(index=df.index)
    for pre in ("h", "a"):
        x[f"{pre}_xg_net5"] = _net(df, pre, "xg")
        x[f"{pre}_c_net5"] = _net(df, pre, "c") / 10.0
        x[f"{pre}_xgf5"] = df[f"{pre}_r_xgf_5v5"]
        x[f"{pre}_xga5"] = df[f"{pre}_r_xga_5v5"]
        x[f"{pre}_gf_all"] = df[f"{pre}_r_gf_all"]
        x[f"{pre}_ga_all"] = df[f"{pre}_r_ga_all"]
        x[f"{pre}_xgf_all"] = df[f"{pre}_r_xgf_all"]
        x[f"{pre}_xga_all"] = df[f"{pre}_r_xga_all"]
        x[f"{pre}_pp"] = df[f"{pre}_r_xgf_pp"]
        x[f"{pre}_pk"] = df[f"{pre}_r_xga_pk"]
        x[f"{pre}_pp_conv"] = df[f"{pre}_r_gf_pp"]
        x[f"{pre}_pk_conv"] = df[f"{pre}_r_ga_pk"]
        # Finishing / goaltending beyond the shot quality already counted.
        x[f"{pre}_finish"] = (df[f"{pre}_r_gf_5v5"]
                              - df[f"{pre}_r_xgf_5v5"])
        x[f"{pre}_suppress"] = (df[f"{pre}_r_xga_5v5"]
                                - df[f"{pre}_r_ga_5v5"])
        x[f"{pre}_gsax"] = df[f"{pre}_g_gsax"].fillna(0.0) * 100.0
        x[f"{pre}_rest"] = df[f"{pre}_rest"].fillna(3).clip(0, 6)
        x[f"{pre}_b2b"] = df[f"{pre}_b2b"].fillna(0)
        x[f"{pre}_g4"] = df[f"{pre}_games_last4"].fillna(0)
        x[f"{pre}_travel"] = df[f"{pre}_travel_km"].fillna(0) / 1000.0
    x["tz_abs"] = df["a_tz_shift"].fillna(0).abs()
    x["rest_gap"] = x.h_rest - x.a_rest
    x["b2b_gap"] = x.h_b2b - x.a_b2b
    # Matchup interactions: my offence against your defence.
    x["h_off_vs_a_def"] = x.h_xgf5 - x.a_xga5
    x["a_off_vs_h_def"] = x.a_xgf5 - x.h_xga5
    x["gsax_gap"] = x.h_gsax - x.a_gsax
    x["xg_gap"] = x.h_xg_net5 - x.a_xg_net5
    return x.astype(float).fillna(0.0)


class NHLModel:
    def __init__(self, alpha=1.0, C=0.08):
        self.alpha = alpha      # Poisson GLM ridge penalty
        self.C = C              # OT/SO logistic penalty

    def fit(self, df):
        X = build_matrix(df).values
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd

        self.ph = PoissonRegressor(alpha=self.alpha, max_iter=5000).fit(
            Z, df.h_reg.values)
        self.pa = PoissonRegressor(alpha=self.alpha, max_iter=5000).fit(
            Z, df.a_reg.values)

        # Dixon-Coles low-score dependence, fit by grid search on the
        # training likelihood. Hockey scores are mildly dependent (score
        # effects), which independent Poissons get wrong exactly at 0-0/1-1.
        lh = self.ph.predict(Z)
        la = self.pa.predict(Z)
        self.tau = self._fit_tau(lh, la, df.h_reg.values, df.a_reg.values)

        # Independent Poissons badly under-produce regulation ties: real
        # hockey has score effects (the trailing team pulls its goalie, play
        # tightens when level) that pull games back to level far more often
        # than two independent rates imply. One multiplicative parameter on
        # the diagonal, moment-matched to the observed regulation-tie rate,
        # fixes the resulting over-confidence near 50/50.
        self.kappa = 1.0
        target = (df.reg_outcome == 1).mean()
        lo, hi = 0.5, 4.0
        for _ in range(40):
            self.kappa = (lo + hi) / 2
            got = np.mean([self._cells(a, b)[1] for a, b in zip(lh, la)])
            if got < target:
                lo = self.kappa
            else:
                hi = self.kappa
        self.reg_tie_target = target

        # Who wins once regulation ends level.
        ot = df[df.reg_outcome == 1]
        if len(ot) > 60:
            Zo = (build_matrix(ot).values - self.mu) / self.sd
            self.ot = LogisticRegression(C=self.C, max_iter=3000).fit(
                Zo, ot.home_win.values)
            self.ot_base = ot.home_win.mean()
        else:
            self.ot, self.ot_base = None, 0.5
        return self

    @staticmethod
    def _dc_adjust(P, lh, la, tau):
        """Dixon-Coles correction on the four lowest-score cells."""
        Q = P.copy()
        Q[0, 0] *= 1 - lh * la * tau
        Q[0, 1] *= 1 + lh * tau
        Q[1, 0] *= 1 + la * tau
        Q[1, 1] *= 1 - tau
        return np.clip(Q, 1e-12, None)

    @staticmethod
    def _fit_tau(lh, la, yh, ya):
        base = np.log(np.clip(poisson.pmf(yh, lh) * poisson.pmf(ya, la),
                              1e-12, None))
        low = (yh <= 1) & (ya <= 1)
        best, best_ll = 0.0, -np.inf
        for tau in np.arange(-0.45, 0.4501, 0.005):
            f = np.ones_like(lh)
            f[low & (yh == 0) & (ya == 0)] = 1 - (lh * la * tau)[
                low & (yh == 0) & (ya == 0)]
            f[low & (yh == 0) & (ya == 1)] = 1 + (lh * tau)[
                low & (yh == 0) & (ya == 1)]
            f[low & (yh == 1) & (ya == 0)] = 1 + (la * tau)[
                low & (yh == 1) & (ya == 0)]
            f[low & (yh == 1) & (ya == 1)] = 1 - tau
            ll = (base + np.log(np.clip(f, 1e-9, None))).sum()
            if ll > best_ll:
                best, best_ll = tau, ll
        return best

    def _cells(self, lh, la):
        """Regulation (home win, tie, away win) for one pair of rates."""
        k = np.arange(MAXG + 1)
        P = self._dc_adjust(np.outer(poisson.pmf(k, lh), poisson.pmf(k, la)),
                            lh, la, self.tau)
        idx_h, idx_a = np.meshgrid(k, k, indexing="ij")
        P[idx_h == idx_a] *= getattr(self, "kappa", 1.0)
        P /= P.sum()
        return (P[idx_h > idx_a].sum(), P[idx_h == idx_a].sum(),
                P[idx_h < idx_a].sum())

    def matrix(self, lh, la, kappa_mult=1.0):
        """Normalised regulation score matrix for one game."""
        k = np.arange(MAXG + 1)
        P = self._dc_adjust(np.outer(poisson.pmf(k, lh), poisson.pmf(k, la)),
                            lh, la, self.tau)
        idx_h, idx_a = np.meshgrid(k, k, indexing="ij")
        P[idx_h == idx_a] *= getattr(self, "kappa", 1.0) * kappa_mult
        return P / P.sum(), idx_h, idx_a

    def tie_rate(self, lh, la, kappa_mult=1.0):
        """Mean regulation-tie probability over a set of rates."""
        out = np.empty(len(lh))
        for i in range(len(lh)):
            P, ih, ia = self.matrix(lh[i], la[i], kappa_mult)
            out[i] = P[ih == ia].sum()
        return float(out.mean())

    def solve_kappa_mult(self, lh, la, target, lo=0.4, hi=3.0, iters=24):
        """Multiplier on kappa that makes the predicted tie rate hit target.

        The league's rate of games finishing level drifts between seasons
        just as scoring does, and kappa is fit on the training seasons. This
        lets a caller retune it from recent games only.
        """
        for _ in range(iters):
            mid = (lo + hi) / 2
            if self.tie_rate(lh, la, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def rates(self, df):
        Z = (build_matrix(df).values - self.mu) / self.sd
        return self.ph.predict(Z), self.pa.predict(Z)

    def predict(self, df, scoring_mult=None, kappa_mult=None):
        """Full set of probabilities, one row per game.

        `scoring_mult` rescales both goal rates to track the league's current
        scoring level, which drifts between seasons.
        """
        Z = (build_matrix(df).values - self.mu) / self.sd
        lh, la = self.ph.predict(Z), self.pa.predict(Z)
        if scoring_mult is not None:
            m = np.asarray(scoring_mult, dtype=float)
            lh, la = lh * m, la * m
        if self.ot is not None:
            p_ot_home = self.ot.predict_proba(Z)[:, 1]
        else:
            p_ot_home = np.full(len(df), self.ot_base)

        km = (np.ones(len(df)) if kappa_mult is None
              else np.asarray(kappa_mult, dtype=float))
        out = []
        for i in range(len(df)):
            P, idx_h, idx_a = self.matrix(lh[i], la[i], km[i])
            reg_h = P[idx_h > idx_a].sum()
            reg_t = P[idx_h == idx_a].sum()
            reg_a = P[idx_h < idx_a].sum()
            pw = reg_h + reg_t * p_ot_home[i]

            # Final total = regulation total, plus one goal whenever the
            # game was level after regulation (the OT winner or the single
            # shootout goal credited to the winner).
            final = np.zeros(2 * MAXG + 3)
            tied = idx_h == idx_a
            np.add.at(final, (idx_h[~tied] + idx_a[~tied]).ravel(),
                      P[~tied].ravel())
            np.add.at(final, (idx_h[tied] + idx_a[tied]).ravel() + 1,
                      P[tied].ravel())
            cum = np.cumsum(final)
            out.append({
                "lam_h": lh[i], "lam_a": la[i],
                "p_home": pw, "p_away": 1 - pw,
                "p_reg_home": reg_h, "p_reg_tie": reg_t, "p_reg_away": reg_a,
                "p_ot_home": p_ot_home[i], "p_ot": reg_t,
                "kappa_mult": km[i],
                "p_home_m15": P[idx_h - idx_a >= 2].sum(),
                "p_away_m15": P[idx_a - idx_h >= 2].sum(),
                "exp_total": float((np.arange(len(final)) * final).sum()),
                "p_over_45": 1 - cum[4], "p_over_55": 1 - cum[5],
                "p_over_65": 1 - cum[6],
            })
        return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- baselines

def elo_ratings(df, k=6.0, home_adv=35.0, carry=0.70, base=1500.0):
    """Goal-differential Elo. Returns pre-game home win prob per game."""
    r, season_of, preds = {}, {}, []
    for g in df.sort_values(["date", "game_id"]).itertuples(index=False):
        for t in (g.home, g.away):
            if t not in r:
                r[t] = base
            elif season_of.get(t) != g.season:
                r[t] = base + carry * (r[t] - base)
            season_of[t] = g.season
        dh = r[g.home] + home_adv - r[g.away]
        p = 1 / (1 + 10 ** (-dh / 400))
        preds.append({"game_id": g.game_id, "elo_p_home": p,
                      "elo_diff": dh})
        margin = g.home_score - g.away_score
        actual = 1.0 if margin > 0 else 0.0
        mult = np.log1p(abs(margin)) * (2.2 / (0.001 * abs(dh) + 2.2))
        delta = k * mult * (actual - p)
        r[g.home] += delta
        r[g.away] -= delta
    return pd.DataFrame(preds), r


def log5_points_pct(df):
    """The naive standings-based baseline: Log5 on points percentage."""
    pts, gp, season_of, out = {}, {}, {}, []
    for g in df.sort_values(["date", "game_id"]).itertuples(index=False):
        for t in (g.home, g.away):
            if t not in pts or season_of.get(t) != g.season:
                pts[t], gp[t] = 0.0, 0
            season_of[t] = g.season
        pa = pts[g.home] / (2 * gp[g.home]) if gp[g.home] >= 5 else 0.5
        pb = pts[g.away] / (2 * gp[g.away]) if gp[g.away] >= 5 else 0.5
        den = pa + pb - 2 * pa * pb
        p = (pa - pa * pb) / den if den > 1e-9 else 0.5
        out.append({"game_id": g.game_id, "log5_p_home": np.clip(p, .05, .95)})
        hw = g.home_score > g.away_score
        ot = g.last_period_type in ("OT", "SO")
        pts[g.home] += 2 if hw else (1 if ot else 0)
        pts[g.away] += 2 if not hw else (1 if ot else 0)
        gp[g.home] += 1
        gp[g.away] += 1
    return pd.DataFrame(out)
