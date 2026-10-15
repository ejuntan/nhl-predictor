"""Tonight's slate against tonight's prices, labelled by what was measured.

`edge_test.py` priced every bucket on 1,414 games with real posted odds. The
labels here are those measurements, not opinions:

  conf < 0.60, market agrees    758 games, -12.8% ROI, CI below zero
  model backs the market's dog  212 games, 47.6% wins but +0.4% ROI, CI +/-14
  conf >= 0.60, market agrees   444 games, -1.8% ROI, CI +/-7
  this filter's own value picks 577 games, -2.6% ROI, CI -10.2% to +5.0%

Only the first is established. The other two are washes, so a game landing
in them is unmeasurable rather than bad.

Odds are fetched live and are not cached: a line moves, so a number here is
only true at the moment it printed.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fetch_odds import OD, SB, get, norm
from odds_eval import american_to_dec, devig_logarithmic


def scoreboard(days):
    rows = []
    for d in days:
        js = get(SB.format(d.strftime("%Y%m%d")))
        for e in (js or {}).get("events", []) or []:
            comp = (e.get("competitions") or [{}])[0]
            h = a = None
            for c in comp.get("competitors", []) or []:
                ab = norm((c.get("team") or {}).get("abbreviation"))
                if c.get("homeAway") == "home":
                    h = ab
                else:
                    a = ab
            if h and a:
                rows.append({"event_id": int(e["id"]),
                             "date": d.strftime("%Y-%m-%d"),
                             "home": h, "away": a})
    return pd.DataFrame(rows)


def odds_for(event_ids):
    rows = []
    for eid in event_ids:
        js = get(OD.format(eid))
        items = (js or {}).get("items") or []
        if not items:
            continue
        it = next((x for x in items if "draft" in
                   ((x.get("provider") or {}).get("name", "")).lower()), items[0])
        ho, ao = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        rows.append({"event_id": eid,
                     "book": (it.get("provider") or {}).get("name"),
                     "home_ml": ho.get("moneyLine"),
                     "away_ml": ao.get("moneyLine"),
                     "total": it.get("overUnder"),
                     "over_odds": it.get("overOdds"),
                     "under_odds": it.get("underOdds")})
    return pd.DataFrame(rows)


def label(agree, conf):
    if not agree:
        return "+0.4% +/-14  backing the market's dog (n=212)"
    if conf < 0.60:
        return "-12.8% +/-6  under 60% and the book agrees (n=758)"
    return "-1.8% +/-7  strong favourite, book agrees (n=444)"


def main(days=4):
    live = pd.read_pickle("data/live_preds.pkl")
    d0 = pd.Timestamp(live.date.min()).date()
    want = [d0 + timedelta(days=i) for i in range(days)]
    sb = scoreboard(want)
    if sb.empty:
        print("no ESPN events for those dates")
        return
    od = odds_for(sb.event_id.tolist())
    m = sb.merge(od, on="event_id").merge(
        live[["date", "home", "away", "p_home", "exp_total", "h_goalie_name",
              "a_goalie_name", "h_goalie_label", "a_goalie_label"]],
        on=["date", "home", "away"], how="inner")
    m = m[m.home_ml.notna() & m.away_ml.notna()]
    if m.empty:
        print("no games matched a live price")
        return

    dh = american_to_dec(m.home_ml.values.astype(float))
    da = american_to_dec(m.away_ml.values.astype(float))
    mh, ma = devig_logarithmic(dh, da)
    ph = m.p_home.values
    back_home = ph > .5
    m["side"] = np.where(back_home, m.home, m.away)
    m["p_model"] = np.where(back_home, ph, 1 - ph)
    m["p_mkt"] = np.where(back_home, mh, ma)
    m["price"] = np.where(back_home, m.home_ml, m.away_ml).astype(int)
    m["dec"] = np.where(back_home, dh, da)
    m["agree"] = m.p_mkt > .5
    m["gap"] = m.p_model - m.p_mkt
    m["edge"] = m.p_model - 1 / m.dec
    m["verdict"] = [label(a, c) for a, c in zip(m.agree, m.p_model)]
    m["ev100"] = m.edge * 100 * m.dec

    m = m.sort_values(["date", "gap"], ascending=[True, False])
    pd.set_option("display.width", 200)
    print(f"{len(m)} games priced, book: {m.book.iloc[0]}  "
          f"(odds live as of this run)\n")
    for d, g in m.groupby("date"):
        print(f"--- {d} ---")
        print(g[["away", "home", "side", "price", "p_model", "p_mkt", "gap",
                 "edge", "ev100", "verdict"]].to_string(
            index=False, formatters={
                "p_model": "{:.3f}".format, "p_mkt": "{:.3f}".format,
                "gap": "{:+.3f}".format, "edge": "{:+.3f}".format,
                "ev100": "{:+.2f}".format}))
        print()

    ok = m[m.agree & (m.p_model >= .60) & (m.edge > 0)]
    print("=" * 70)
    if ok.empty:
        print("No strong favourite prices above the model's number today.")
    else:
        print("Strong favourites priced above the model's number "
              "(+2.7% on 204 past bets, CI +/-11):")
        print(ok[["date", "away", "home", "side", "price", "p_model",
                  "p_mkt", "edge"]].to_string(index=False, formatters={
                      "p_model": "{:.3f}".format, "p_mkt": "{:.3f}".format,
                      "edge": "{:+.3f}".format}))
    print("=" * 70)
    est = m[m.agree & (m.p_model < .60)]
    print(f"{len(est)} of {len(m)} games are in the one situation "
          f"established as a loser (-12.8%, interval below zero). The model "
          f"shows an edge over the posted price in "
          f"{int((m.edge > 0).sum())} games overall; that filter has "
          f"returned -2.6% with an interval straddling zero.")
    m.attrs["fetched_at"] = pd.Timestamp.now(tz="UTC").strftime(
        "%Y-%m-%d %H:%M UTC")
    m.to_pickle("data/live_edge.pkl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
