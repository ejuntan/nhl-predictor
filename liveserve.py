"""Refresh the sportsbook numbers on the server, between pushes.

The MLB site rebuilds itself on every request because it pulls from the MLB
Stats API and keeps no history. This one cannot: the model is fitted on five
seasons of play-by-play in a 144 MB database that will never be in git.

But the model's own prices do not move during a day. They are computed from
ratings that only change when games finish, which is what the nightly push
is for. The number that moves all day is the sportsbook's, and that one
needs nothing local at all -- only the game list already in the payload and
a public endpoint.

So this refreshes the market side on a timer: prices, the no-vig
probabilities, the edge against each posted price, and which measured
bucket each game falls into. Pure standard library plus `requests`, so the
web service keeps its ten-second build and does not drag in pandas,
scikit-learn or a 45 MB frame to do arithmetic on twenty numbers.

Any failure leaves the payload exactly as the build shipped it. A page
showing this morning's prices is a far better outcome than a broken one.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "out", "tool_data.json")
TTL = int(os.environ.get("NHL_REFRESH_TTL", "900"))        # 15 minutes
SB = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={}"
OD = ("https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/"
      "events/{0}/competitions/{0}/odds")
UA = "nhl-predictor/1.0 (personal research)"
ESPN_TO_NHL = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
               "UTAH": "UTA"}

_lock = threading.Lock()
_s = {"payload": None, "at": 0.0, "source": "build", "error": None,
      "lines": 0}


def _get(url, tries=2):
    for a in range(tries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA})
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if a == tries - 1:
                return None
    return None


def american_to_dec(a):
    a = float(a)
    if a == 0:
        return None
    return 1 + (a / 100 if a > 0 else 100 / -a)


def devig_log(dh, da):
    """Logarithmic (odds-ratio) de-vig: p_i proportional to (1/d_i)**k.

    Bisection on k until the two fair probabilities sum to one. The
    proportional method just scales both by the overround, which skews
    long prices; this one is the standard correction for that.
    """
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        h, a = (1 / dh) ** k, (1 / da) ** k
        if h + a > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    h, a = (1 / dh) ** k, (1 / da) ** k
    return h / (h + a), a / (h + a)


def bucket(p_model, p_mkt):
    """The situations measured in MODEL.md section 14, same boundaries."""
    if p_mkt < 0.495:
        return "dog"
    return "fav" if p_model >= 0.60 else "thin"


def fetch_lines(games):
    """Current prices for whichever scheduled games the books have posted."""
    want = {}
    for g in games:
        want.setdefault(g["d"], []).append(g)
    out, seen = {}, 0
    for day, gs in sorted(want.items())[:4]:
        js = _get(SB.format(day.replace("-", "")))
        if not js:
            continue
        for ev in js.get("events", []) or []:
            comp = (ev.get("competitions") or [{}])[0]
            home = away = None
            for c in comp.get("competitors", []) or []:
                ab = ((c.get("team") or {}).get("abbreviation") or "").upper()
                ab = ESPN_TO_NHL.get(ab, ab)
                if c.get("homeAway") == "home":
                    home = ab
                else:
                    away = ab
            g = next((x for x in gs if x["h"] == home and x["a"] == away), None)
            if not g:
                continue
            o = _get(OD.format(int(ev["id"])))
            items = (o or {}).get("items") or []
            if not items:
                continue
            it = next((x for x in items if "draft" in
                       ((x.get("provider") or {}).get("name", "")).lower()),
                      items[0])
            hm = (it.get("homeTeamOdds") or {}).get("moneyLine")
            am = (it.get("awayTeamOdds") or {}).get("moneyLine")
            if hm is None or am is None:
                continue
            dh, da = american_to_dec(hm), american_to_dec(am)
            if not (dh and da):
                continue
            mh, ma = devig_log(dh, da)
            back_home = g["ph"] > 0.5
            side = g["h"] if back_home else g["a"]
            pm = g["ph"] if back_home else g["pa"]
            pk = mh if back_home else ma
            dec = dh if back_home else da
            out[g["d"] + "|" + g["a"] + g["h"]] = {
                "book": (it.get("provider") or {}).get("name"),
                "hml": int(hm), "aml": int(am),
                "tot": it.get("overUnder"),
                "oo": it.get("overOdds"), "uo": it.get("underOdds"),
                "mh": round(mh, 4), "side": side, "pm": round(pm, 4),
                "pk": round(pk, 4), "gap": round(pm - pk, 4),
                "edge": round(pm - 1 / dec, 4), "b": bucket(pm, pk)}
            seen += 1
    return out, seen


def payload(force=False):
    now = time.time()
    with _lock:
        if _s["payload"] is not None and not force and now - _s["at"] < TTL:
            return _s["payload"]
    try:
        with open(STATIC) as f:
            p = json.load(f)
    except Exception as e:                     # nothing to serve at all
        _s["error"] = f"payload: {e}"
        return _s["payload"]
    try:
        lines, n = fetch_lines(p.get("games", []))
        if n:
            p["lines"] = lines
            p["lineAt"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC")
            src, err = "live odds", None
        else:
            src, err = "build", "no prices posted"
    except Exception as e:
        src, err, n = "build", str(e), 0
    with _lock:
        _s.update(payload=p, at=now, source=src, error=err, lines=n)
    return p


def status():
    age = time.time() - _s["at"] if _s["at"] else None
    return {"data_source": _s["source"], "refresh_error": _s["error"],
            "lines_live": _s["lines"], "ttl_seconds": TTL,
            "refreshed_seconds_ago": round(age, 1) if age else None}
