"""Compact JSON payload for the betting tool page."""
import json
import sqlite3

import numpy as np
import pandas as pd

import names as N

# Per-market reliability, taken from the measured holdout in MODEL.md.
# These drive the tool's own warnings rather than sitting in a footnote.
#
# Totals and the overtime price are deliberately absent. Both are right on
# average and neither carries information -- the expected total explains
# 0.33% of the variance in real totals (MODEL.md section 15) and the
# overtime price scores slightly worse than quoting the league rate
# (section 18). A market that cannot be priced honestly is not listed here,
# and the page therefore cannot offer a price box for it.
MARKETS = {
    "moneyline": {"label": "Moneyline", "grade": "reference",
                  "note": "Model log loss 0.682 against the actual posted "
                          "prices on all 1,414 holdout games, the market's "
                          "0.679 -- statistically level. But backing the "
                          "model's side at those prices returned -8.1% per "
                          "dollar. Shown for context, not to bet into."},
    # Totals are calibrated on average and carry almost no information.
    # The expected total explains 0.33% of the variance in real totals, and
    # the calibration slope at a book's posted line is 0.28 against the
    # moneyline's 0.78 -- a 56% from this model is really about 52%. Graded
    # reference so a large apparent edge can never read as a play.
    "puckline": {"label": "Home -1.5", "grade": "usable",
                 "note": "The sharpest game-level market here: it removes "
                         "2.79% of the base rate's error against the "
                         "moneyline's 1.81%, and all seven probability bands "
                         "land inside their intervals. It used to sit three "
                         "points low; a trailing-window level correction "
                         "now closes that."},
    # Calibrated on average and close to uninformative, the same shape as
    # game totals: a Brier skill of -0.0019 means it is slightly worse than
    # simply quoting the league's overtime rate, and a slope of 0.34 means
    # the number moves without meaning it.
    "sog_15": {"label": "Over 1.5 shots", "grade": "primary",
               "note": "Best market available: 13.3% of baseline error removed, "
                       "and beats a player's own average at t=8.3."},
    "sog_25": {"label": "Over 2.5 shots", "grade": "primary",
               "note": "13.2% of baseline error removed, t=5.6 vs a player's own average."},
    # Correction to an earlier claim: the 13.1% figure is the error cut on
    # the save COUNT, not on the over/under. On the quantity actually bet,
    # these remove 4.6-5.7% of the base rate's error, under half what the
    # shots markets manage, and the slopes near 1.3 say the probabilities
    # are still too timid. Graded usable rather than primary.
    "saves_245": {"label": "Over 24.5 saves", "grade": "usable",
                  "note": "Removes 5.7% of the base rate's error, under half "
                          "what shots on goal manage. Its value is the "
                          "baseline it beats: a goalie's own recent average "
                          "is worse than guessing the league mean, because "
                          "past save counts reflect the opponents he faced."},
    "saves_275": {"label": "Over 27.5 saves", "grade": "usable",
                  "note": "Removes 5.6% of the base rate's error. The "
                          "calibration slope of 1.31 means the quoted "
                          "probabilities are, if anything, too cautious."},
    "saves_305": {"label": "Over 30.5 saves", "grade": "usable",
                  "note": "Removes 4.6%. The thin end of the distribution: "
                          "fewer starts reach it, so the estimate is "
                          "noisier, though every band lands inside its "
                          "interval."},
    # Re-measured: skill +0.1025 and a slope of 1.026 over 47,345
    # skater-games, with 5 of 6 bands inside their intervals. The earlier
    # "caution" grade predated the temperature correction.
    "sog_35": {"label": "Over 3.5 shots", "grade": "usable",
               "note": "Removes 10.3% of the base rate's error against "
                       "13.3% at the lower lines. Slightly weaker and in "
                       "the tail, but properly calibrated."},
}


def breakeven(p):
    """The American price at which a bet at probability p is exactly even."""
    if not 0 < p < 1:
        return None
    return int(-round(100 * p / (1 - p)) if p >= 0.5
               else round(100 * (1 - p) / p))


def top_props(p):
    """Fifteen props a day, by the rule props_top.py validated.

    The model departs from a player's own recent average; the holdout says
    that departure is where its skill is, and that the departure is almost
    always downward because a recent average over-states a shooter (it runs
    +0.022 shots high on holdout against the model's -0.001). So the list
    comes out mostly unders, and that is the finding rather than a fault.
    """
    from scipy.stats import nbinom
    # The caller has already renamed the dotted columns, as itertuples
    # requires; use those names.
    LINES = [(1.5, "p15"), (2.5, "p25"), (3.5, "p35")]
    rows = []
    for ln, col in LINES:
        k = int(ln)
        size = p.nb_size.values
        # The same negative binomial, fed the player's own recent average:
        # the hand baseline the model has to beat to be worth anything.
        base = 1 - nbinom.cdf(k, size, size / (size + p.e_sog.values))
        pm = p[col].values
        over = pm > base
        rows.append(pd.DataFrame({
            "date": p.date.values, "pid": p.player_id.values,
            "n": p.player.values, "t": p.team.values, "o": p.opp.values,
            "hm": p.is_home.values.astype(int), "ln": ln,
            "over": over.astype(int),
            "p": np.where(over, pm, 1 - pm),
            "b": np.where(over, base, 1 - base),
            "xs": p.exp_sog.values, "es": p.e_sog.values,
            "gap": np.abs(pm - base),
        }))
    a = pd.concat(rows, ignore_index=True)
    # Both bars: the model must disagree with the recent average, and must
    # rate its own side better than a coin flip by a clear margin.
    a = a[a.p >= 0.55]
    a = (a.sort_values("gap", ascending=False)
          .groupby(["date", "pid"]).head(1)          # one line per player
          .sort_values("gap", ascending=False)
          .groupby("date").head(15))
    out = {}
    for d, grp in a.groupby("date"):
        picks = []
        for r in grp.sort_values("gap", ascending=False).itertuples():
            # Price the rounded probability, not the full-precision one, so
            # a reader checking the break-even against the percentage the
            # page shows him gets the same answer.
            pr = round(float(r.p), 4)
            picks.append({
                "n": r.n, "t": r.t, "o": r.o, "hm": int(r.hm),
                "ln": float(r.ln), "ov": int(r.over),
                "p": pr, "b": round(float(r.b), 4),
                "xs": round(float(r.xs), 2), "es": round(float(r.es), 2),
                "be": breakeven(pr),
            })
        out[str(d)] = picks
    return out


def past_payload(nm):
    """Played games with the prediction the model actually made beforehand.

    These come from the walk-forward backtest, so each one was produced by a
    model fit only on earlier seasons. Nothing here is hindsight.
    """
    bt = pd.read_pickle("data/backtest_preds.pkl")
    cur = bt[bt.season == "20262027"].copy()
    # Where a prediction was logged before puck drop, that is the one that
    # counts. The backtest is a different model instance -- fitted only
    # through last season, where the live model is fitted through yesterday
    # -- and the two can disagree on the side. Showing the backtest's call
    # on a game the page had already called differently is a record of a
    # prediction nobody saw.
    try:
        log = pd.read_csv("data/frozen_log.csv")[["game_id", "p_home"]]
        log = log.rename(columns={"p_home": "p_logged"}).drop_duplicates(
            "game_id")
        cur = cur.merge(log, on="game_id", how="left")
        n_log = int(cur.p_logged.notna().sum())
        cur["src"] = np.where(cur.p_logged.notna(), "logged", "backtest")
        cur["p_home"] = cur.p_logged.fillna(cur.p_home)
        cur["p_away"] = 1 - cur.p_home
    except FileNotFoundError:
        cur["src"], n_log = "backtest", 0
    print(f"  past games using the logged prediction: {n_log} of {len(cur)}")
    games = []
    for r in cur.sort_values(["date", "game_id"]).itertuples(index=False):
        pick_home = r.p_home > 0.5
        games.append({
            "d": r.date, "a": r.away, "h": r.home,
            "ph": round(float(r.p_home), 4), "pa": round(float(r.p_away), 4),
            "lh": round(float(r.lam_h), 2), "la": round(float(r.lam_a), 2),
            "tot": round(float(r.exp_total), 2),
            "ot": round(float(r.p_ot), 4),
            "o55": round(float(r.p_over_55), 4),
            "o65": round(float(r.p_over_65), 4),
            "pl": round(float(r.p_home_m15), 4),
            # what actually happened
            "hsc": int(r.home_score), "asc": int(r.away_score),
            "wentot": int(r.went_ot), "atot": int(r.total_goals),
            "hit": int(bool(pick_home) == bool(r.home_win == 1)),
            "src": r.src,
        })

    pr = pd.read_pickle("data/props_preds.pkl")
    pc = pr[pr.season == "20262027"].copy()
    pc = pc.rename(columns={"p_over_1.5": "p15", "p_over_2.5": "p25",
                            "p_over_3.5": "p35"})
    props = []
    for r in pc.sort_values(["date", "mu"], ascending=[True, False]).itertuples(index=False):
        props.append({
            "d": r.date, "n": nm.get(str(int(r.player_id)), "#" + str(r.player_id)),
            "t": r.team, "o": r.opp,
            "toi": round(float(r.toi_reg) / 60, 1),
            "xs": round(float(r.mu), 2),
            "p15": round(float(r.p15), 4),
            "p25": round(float(r.p25), 4),
            "p35": round(float(r.p35), 4),
            "asog": int(r.sog),
        })

    hits = sum(g["hit"] for g in games)
    return {"games": games, "props": props,
            "record": {"w": hits, "l": len(games) - hits,
                       "acc": round(hits / len(games), 4) if games else 0}}


# Measured on 1,414 games with real posted prices (edge_test.py, bucket
# figures recomputed with the pick'em boundary at 0.495). `roi` is the return
# per dollar from backing the model's side in that situation; `ci` is the
# half-width of the 95% bootstrap interval; `vp_roi` the same restricted to
# the games where the model also showed an edge over the posted price --
# which is what this page calls a value pick.
BUCKETS = {
    "dog": {"n": 212, "win": 0.476, "roi": 0.0043, "ci": 0.142,
            "vp_n": 202, "vp_roi": 0.0154,
            "why": "the model is backing the side the market has as the "
                   "underdog; it won 47.6% of 212 such games, but underdog "
                   "prices pay more, so the return was +0.4% with an "
                   "interval of plus or minus 14 points -- not established "
                   "either way"},
    "thin": {"n": 758, "win": 0.509, "roi": -0.1277, "ci": 0.061,
             "vp_n": 171, "vp_roi": -0.1369,
             "why": "the model's confidence is under 60% and the market "
                    "agrees on the side; 758 such games returned -12.8% per "
                    "dollar with the whole interval below zero, and the 171 "
                    "of them that showed an edge returned -13.7%. This is "
                    "the one situation established as a loser"},
    "fav": {"n": 444, "win": 0.642, "roi": -0.0179, "ci": 0.068,
            "vp_n": 204, "vp_roi": 0.0273,
            "why": "a strong favourite the market also favours; 444 such "
                   "games returned -1.8%, and the 204 that showed an edge "
                   "returned +2.7% with an interval of plus or minus 11 "
                   "points -- the only positive sign anywhere, and still "
                   "indistinguishable from zero"},
}

# Every game where the model showed an edge over the posted price, which is
# the filter the MLB page's value finder uses.
# ---------------------------------------------------------------------------
# The prop shortlist.
#
# There are no posted prop prices in this payload, so "best" cannot mean edge
# against a book. What it can mean is the model's one established prop claim:
# it beats a player's own recent average. props_top.py measures where that
# advantage lives, on the 143,334 holdout prop-lines the xG fit never saw.
#
# What it found, and why the rule is shaped the way it is:
#   * Skill over the baseline rises with the size of the disagreement:
#     +0.06% of its error removed under a 2-point gap, +6.3% over 12 points.
#   * Ranking by the model's raw confidence instead is useless for betting --
#     it returns 99% props that need -7529 to break even.
#   * The gap alone takes sides the model itself rates under 50%; 37% of such
#     a list, and those came in 38.7% of the time. Hence the 55% floor.
#   * Rank within the fifteen carries no information (67.0 / 66.3 / 68.2% for
#     ranks 1-5, 6-10, 11-15), so the page must not number them.
#   * The 2-point over-confidence is not a level shift: fitted on the first
#     half of the holdout and applied to the second it made calibration
#     worse (+2.1% against -0.9% uncorrected). So nothing is corrected and
#     the model's own number is shown, with the measured shortfall stated.
#   * One line per player rather than letting a man repeat: same calibration,
#     and the group needs -180 rather than -204, so the prices are reachable.
TOP_PROPS = {
    "n": 2669, "days": 178, "model": 0.660, "actual": 0.643, "be": -180,
    "lo": -196, "hi": -166,
    # Split so the page can set the measured half in bold without having to
    # cut a sentence apart at a marker.
    "head": "Over 178 holdout days this shortlist's side came in 64.3% of "
            "the time against the 66.0% the model claimed",
    "tail": "\u2014 about two points optimistic, and that shortfall moves "
            "around too much to correct. Treat the model's number as a "
            "ceiling and the break-even price as the bar a book has to clear.",
}

VALUE_PICKS = {"n": 577, "win": 0.532, "roi": -0.026,
               "lo": -0.1021, "hi": 0.0503}


def line_payload():
    """Posted prices for the upcoming slate, if they have been fetched.

    Prices are a snapshot: `live_edge.py` reads them live, and from the
    moment this payload is written they are historical. The page says so.
    """
    try:
        d = pd.read_pickle("data/live_edge.pkl")
    except FileNotFoundError:
        return {}, None
    out = {}
    for r in d.itertuples(index=False):
        key = r.date + "|" + r.away + r.home
        # A pick'em is not the market naming an underdog, so the dog
        # bucket needs the market to actually favour the other side. Those
        # games land in "thin", which is where a coin-flip belongs anyway.
        bucket = ("dog" if r.p_mkt < 0.495
                  else ("fav" if r.p_model >= 0.60 else "thin"))
        out[key] = {
            "book": r.book,
            "hml": None if pd.isna(r.home_ml) else int(r.home_ml),
            "aml": None if pd.isna(r.away_ml) else int(r.away_ml),
            "tot": None if pd.isna(r.total) else float(r.total),
            "oo": None if pd.isna(r.over_odds) else int(r.over_odds),
            "uo": None if pd.isna(r.under_odds) else int(r.under_odds),
            "mh": round(float(r.p_mkt if r.side == r.home
                              else 1 - r.p_mkt), 4),
            "side": r.side,
            "pm": round(float(r.p_model), 4),
            "pk": round(float(r.p_mkt), 4),
            "gap": round(float(r.gap), 4),
            "edge": round(float(r.edge), 4),
            "b": bucket,
        }
    return out, str(d.attrs.get("fetched_at", ""))


def in_progress_payload():
    """Games under way: neither finished nor upcoming, and previously shown
    nowhere at all.

    The pre-game call comes from the frozen log where there is one, because
    that is the prediction actually committed to before puck drop. The
    walk-forward backtest is the fallback for games played before the log
    began on 2026-10-14; it is a different model instance and can differ on
    the side, so the two are never mixed silently -- each row says which it
    came from.
    """
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    g = pd.read_sql("""SELECT game_id, date, away, home, state,
                              away_score, home_score
                       FROM games WHERE state IN ('LIVE','CRIT','PRE')
                       ORDER BY date, game_id""", con)
    if g.empty:
        return []
    try:
        log = pd.read_csv("data/frozen_log.csv")[["game_id", "p_home"]]
        log["src"] = "logged"
    except FileNotFoundError:
        log = pd.DataFrame(columns=["game_id", "p_home", "src"])
    bt = pd.read_pickle("data/backtest_preds.pkl")[["game_id", "p_home"]]
    bt["src"] = "backtest"
    pre = pd.concat([log, bt]).drop_duplicates("game_id", keep="first")
    g = g.merge(pre, on="game_id", how="left")

    out = []
    for r in g.itertuples(index=False):
        if pd.isna(r.p_home):
            continue
        out.append({
            "d": r.date, "a": r.away, "h": r.home,
            "ph": round(float(r.p_home), 4),
            "pa": round(1 - float(r.p_home), 4),
            "src": r.src,
            "hsc": int(r.home_score) if pd.notna(r.home_score) else 0,
            "asc": int(r.away_score) if pd.notna(r.away_score) else 0,
            "st": r.state})
    return out


def main():
    live = pd.read_pickle("data/live_preds.pkl")
    props = pd.read_pickle("data/live_props.pkl")
    nm = N.load()

    games = []
    for r in live.sort_values(["date", "game_id"]).itertuples(index=False):
        games.append({
            "d": r.date, "a": r.away, "h": r.home,
            "ph": round(float(r.p_home), 4), "pa": round(float(r.p_away), 4),
            "lh": round(float(r.lam_h), 2), "la": round(float(r.lam_a), 2),
            "tot": round(float(r.exp_total), 2),
            "ot": round(float(r.p_ot), 4),
            "o45": round(float(r.p_over_45), 4),
            "o55": round(float(r.p_over_55), 4),
            "o65": round(float(r.p_over_65), 4),
            "pl": round(float(r.p_home_m15), 4),
            "hg": nm.get(str(int(r.h_goalie)), "TBD") if pd.notna(r.h_goalie) else "TBD",
            "ag": nm.get(str(int(r.a_goalie)), "TBD") if pd.notna(r.a_goalie) else "TBD",
            "hb": int(str(r.h_goalie_label).startswith("backup")),
            "ab": int(str(r.a_goalie_label).startswith("backup")),
            "hr": int(r.h_rest), "ar": int(r.a_rest),
        })

    p = props[props.exp_sog >= 1.0].copy()
    # itertuples mangles the dotted column names, so rename first.
    p = p.rename(columns={"p_over_1.5": "p15", "p_over_2.5": "p25",
                          "p_over_3.5": "p35"})
    plist = []
    for r in p.sort_values(["date", "exp_sog"], ascending=[True, False]).itertuples(index=False):
        plist.append({
            "d": r.date, "n": r.player, "t": r.team, "o": r.opp,
            "hm": int(r.is_home),
            "toi": round(float(r.exp_toi_min), 1),
            "xs": round(float(r.exp_sog), 2),
            "p15": round(float(r.p15), 4),
            "p25": round(float(r.p25), 4),
            "p35": round(float(r.p35), 4),
            "sa": round(float(r.opp_sa60), 1),
        })

    # Goalie saves. itertuples mangles the dotted line names, so rename
    # first, exactly as the skater props do.
    try:
        sv = pd.read_pickle("data/live_saves.pkl").rename(columns={
            "p_over_24.5": "p245", "p_over_27.5": "p275",
            "p_over_30.5": "p305"})
    except FileNotFoundError:
        sv = pd.DataFrame()
    saves = []
    for r in sv.itertuples(index=False):
        saves.append({
            "d": r.date, "t": r.team, "o": r.opp, "hm": int(r.is_home),
            "g": r.goalie,
            # A guessed starter is a guessed price, and more so here than
            # anywhere else on the page: the whole quantity is one goalie's
            # workload, so a wrong name is a wrong number rather than a
            # small perturbation.
            "bk": int(str(r.label).startswith("backup")),
            "toi": round(float(r.exp_toi_min), 1),
            "xs": round(float(r.mu), 2),
            "p245": round(float(r.p245), 4),
            "p275": round(float(r.p275), 4),
            "p305": round(float(r.p305), 4),
        })

    tops = top_props(p)

    past = past_payload(nm)
    lines, line_at = line_payload()
    payload = {
        "asof": str(live.date.min()),
        "generated": str(pd.Timestamp.now().date()),
        "games": games,
        "props": plist,
        "saves": saves,
        "inplay": in_progress_payload(),
        "past": past["games"],
        "pastProps": past["props"],
        "record": past["record"],
        "markets": MARKETS,
        "lines": lines,
        "lineAt": line_at,
        "buckets": BUCKETS,
        "valuePicks": VALUE_PICKS,
        "topProps": tops,
        "topPropsEvidence": TOP_PROPS,
        "evidence": {
            "ml_logloss": 0.68187, "ml_brier": 0.24452, "ml_acc": 0.5460,
            "ml_games": 1414, "market_logloss": 0.6788,
            "ml_roi": -0.0805, "ml_roi_lo": -0.1266, "ml_roi_hi": -0.0344,
            "sog_n": 47179, "sog_cut_naive": 13.344, "sog_cut_own": 0.787,
            "sog_t": 8.35, "ml_cut_naive": 2.39,
        },
    }
    js = json.dumps(payload, separators=(",", ":"))
    open("out/tool_data.json", "w").write(js)
    print(f"upcoming games {len(games)}  props {len(plist)}  "
          f"saves {len(saves)}  posted lines {len(lines)}  "
          f"in progress {len(payload['inplay'])}")
    print(f"prop shortlist: {sum(len(v) for v in tops.values())} picks "
          f"over {len(tops)} days")
    print(f"past games {len(past['games'])}  past props {len(past['props'])}"
          f"  record {past['record']['w']}-{past['record']['l']}"
          f" ({past['record']['acc']:.1%})")
    print(f"payload {len(js)/1024:.1f} KB")


if __name__ == "__main__":
    main()
