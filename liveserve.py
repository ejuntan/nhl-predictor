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
# Scores and prices move at different speeds, so they get different clocks.
# A game's score changes minute to minute; a moneyline drifts over hours.
# One shared timer would mean either stale scores or three times the odds
# calls for nothing.
SCORE_TTL = int(os.environ.get("NHL_SCORE_TTL", "60"))      # 1 minute
ODDS_TTL = int(os.environ.get("NHL_ODDS_TTL", "900"))       # 15 minutes
SCORE = "https://api-web.nhle.com/v1/score/{}"
SB = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={}"
OD = ("https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/"
      "events/{0}/competitions/{0}/odds")
UA = "nhl-predictor/1.0 (personal research)"
ESPN_TO_NHL = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
               "UTAH": "UTA"}

_lock = threading.Lock()
_s = {"payload": None, "scores_at": 0.0, "odds_at": 0.0, "built": 0.0,
      "source": "build", "error": None, "lines": 0}


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


def refresh_scores(p):
    """Update games under way, and retire the ones that have finished.

    A game in progress is the one thing on this page that changes minute to
    minute, so a payload built this morning is wrong about it by the
    evening. The score endpoint is public and costs one call per date.
    """
    inplay = p.get("inplay") or []
    upcoming = p.get("games") or []
    # The in-play list is built when the page is built, and the nightly
    # build runs in the morning, when nothing is under way. So a game that
    # starts tonight would never appear as live at all unless upcoming
    # games are promoted here. Dates come from the payload rather than from
    # a clock, which avoids guessing the league's own date convention.
    days = sorted({g["d"] for g in inplay}
                  | set(sorted({g["d"] for g in upcoming})[:2]))
    moved, promoted = 0, []
    for day in days:
        js = _get(SCORE.format(day))
        if not js:
            continue
        by = {int(x["id"]): x for x in (js.get("games") or [])}

        for g in upcoming:
            if g["d"] != day:
                continue
            row = next((v for v in by.values()
                        if v["homeTeam"]["abbrev"] == g["h"]
                        and v["awayTeam"]["abbrev"] == g["a"]), None)
            if not row or row.get("gameState") not in ("LIVE", "CRIT"):
                continue
            pd_ = row.get("periodDescriptor") or {}
            promoted.append({
                "d": g["d"], "a": g["a"], "h": g["h"],
                "ph": g["ph"], "pa": g["pa"], "src": "logged",
                "hsc": int(row["homeTeam"].get("score") or 0),
                "asc": int(row["awayTeam"].get("score") or 0),
                "st": row.get("gameState"),
                "per": pd_.get("periodType") or "", "pn": pd_.get("number"),
                "clk": (row.get("clock") or {}).get("timeRemaining")})

        for g in inplay:
            if g["d"] != day:
                continue
            row = next((v for v in by.values()
                        if v["homeTeam"]["abbrev"] == g["h"]
                        and v["awayTeam"]["abbrev"] == g["a"]), None)
            if not row:
                continue
            st = row.get("gameState")
            g["hsc"] = int(row["homeTeam"].get("score") or 0)
            g["asc"] = int(row["awayTeam"].get("score") or 0)
            pd_ = row.get("periodDescriptor") or {}
            g["per"] = pd_.get("periodType") or ""
            g["pn"] = pd_.get("number")
            g["clk"] = (row.get("clock") or {}).get("timeRemaining")
            moved += 1
            g["st"] = st
            # A game that finishes during the day is kept, not dropped.
            # Retiring it from this list removed it from the page entirely,
            # because the record it belongs in is only rebuilt by the
            # nightly push -- so four of Thursday's eleven games were
            # showing nowhere at all. It stays here, marked final, until
            # that rebuild folds it into the record properly.
            if st in ("OFF", "FINAL"):
                g["hit"] = int((g["ph"] > 0.5) == (g["hsc"] > g["asc"]))
    if promoted:
        p["inplay"] = inplay + promoted
        started = {(g["d"], g["a"], g["h"]) for g in promoted}
        p["games"] = [g for g in upcoming
                      if (g["d"], g["a"], g["h"]) not in started]
    p["finished_since_build"] = sum(
        1 for g in (p.get("inplay") or []) if g.get("st") in ("OFF", "FINAL"))
    return moved + len(promoted)


def payload(force=False, force_scores=False):
    """The build's payload, with scores and prices refreshed on their own
    clocks.

    The working copy is kept between requests rather than rebuilt from the
    file each time, because a game promoted to in-progress should stay
    promoted. The file is reread only when it changes on disk, which is
    what a deploy looks like from in here.
    """
    now = time.time()
    try:
        mtime = os.path.getmtime(STATIC)
    except OSError as e:
        _s["error"] = f"payload: {e}"
        return _s["payload"]

    with _lock:
        p = _s["payload"]
        stale_build = p is None or mtime != _s["built"]
        # Scores can be forced on their own. They cost one call per date
        # and finish in about a second; the odds cost a lookup per game and
        # take half a minute, which is no use to someone who just pressed
        # reload -- and pointless besides, since a moneyline does not move
        # in the seconds between two page views.
        need_scores = (stale_build or force or force_scores
                       or now - _s["scores_at"] >= SCORE_TTL)
        need_odds = stale_build or force or now - _s["odds_at"] >= ODDS_TTL
        if not (need_scores or need_odds):
            return p

    if stale_build:
        try:
            with open(STATIC) as f:
                p = json.load(f)
        except Exception as e:
            _s["error"] = f"payload: {e}"
            return _s["payload"]
        with _lock:
            _s.update(payload=p, built=mtime, scores_at=0.0, odds_at=0.0)

    err = None
    if need_scores:
        try:
            n = refresh_scores(p)
            with _lock:
                _s["scores_at"] = now
            if n:
                _s["source"] = "live scores"
        except Exception as e:
            err = f"scores: {e}"
    if need_odds:
        try:
            lines, n = fetch_lines(p.get("games", []))
            if n:
                p["lines"] = lines
                p["lineAt"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC")
                _s["source"] = "live odds"
            with _lock:
                _s.update(odds_at=now, lines=n)
        except Exception as e:
            err = f"{err + '; ' if err else ''}odds: {e}"

    with _lock:
        _s["error"] = err
        return _s["payload"]


def status():
    now = time.time()
    p = _s["payload"] or {}

    def age(k):
        return round(now - _s[k], 1) if _s[k] else None

    return {"data_source": _s["source"], "refresh_error": _s["error"],
            "games_in_progress": len(p.get("inplay") or []),
            "lines_live": _s["lines"],
            "scores": {"ttl_seconds": SCORE_TTL,
                       "refreshed_seconds_ago": age("scores_at")},
            "odds": {"ttl_seconds": ODDS_TTL,
                     "refreshed_seconds_ago": age("odds_at")}}
