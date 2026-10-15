"""Compact JSON payload for the betting tool page."""
import json

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
    "saves_245": {"label": "Over 24.5 saves", "grade": "primary",
                  "note": "Removes 13.1% of the error a league-average guess "
                          "makes, and a goalie's own recent average is worse "
                          "than that guess, so the usual handicap is "
                          "actively misleading here."},
    "saves_275": {"label": "Over 27.5 saves", "grade": "primary",
                  "note": "Same model. Calibrated in 7 of 8 probability "
                          "bands after the level correction."},
    "saves_305": {"label": "Over 30.5 saves", "grade": "usable",
                  "note": "The thin end of the distribution; fewer starts "
                          "reach it, so the estimate is noisier."},
    "sog_35": {"label": "Over 3.5 shots", "grade": "caution",
               "note": "Calibrated except at the top: the two highest bands came in "
                       "about 4 points above what was predicted."},
}


def past_payload(nm):
    """Played games with the prediction the model actually made beforehand.

    These come from the walk-forward backtest, so each one was produced by a
    model fit only on earlier seasons. Nothing here is hindsight.
    """
    bt = pd.read_pickle("data/backtest_preds.pkl")
    cur = bt[bt.season == "20262027"].copy()
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

    past = past_payload(nm)
    lines, line_at = line_payload()
    payload = {
        "asof": str(live.date.min()),
        "generated": str(pd.Timestamp.now().date()),
        "games": games,
        "props": plist,
        "saves": saves,
        "past": past["games"],
        "pastProps": past["props"],
        "record": past["record"],
        "markets": MARKETS,
        "lines": lines,
        "lineAt": line_at,
        "buckets": BUCKETS,
        "valuePicks": VALUE_PICKS,
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
          f"saves {len(saves)}  posted lines {len(lines)}")
    print(f"past games {len(past['games'])}  past props {len(past['props'])}"
          f"  record {past['record']['w']}-{past['record']['l']}"
          f" ({past['record']['acc']:.1%})")
    print(f"payload {len(js)/1024:.1f} KB")


if __name__ == "__main__":
    main()
