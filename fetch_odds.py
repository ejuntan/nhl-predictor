"""Collect sportsbook prices for played games, from ESPN's public API.

ESPN records a DraftKings line per game: two-way moneyline, total with its
over/under prices, and the puck line. For a completed game this is the last
line ESPN held, which is at or near the close -- it is not verifiably the
exact closing number, and the comparison is labelled accordingly.

Two stages, both resumable:
  dates  -- scoreboard per calendar day, to map ESPN events to NHL games
  odds   -- the lightweight odds endpoint per event (~7 KB each)
"""
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import requests
from requests.adapters import HTTPAdapter

SB = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={}"
OD = ("https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/"
      "events/{0}/competitions/{0}/odds")
# Its own file: ESPN data is keyed by ESPN event id and is joined to the
# NHL tables at analysis time, so a separate store avoids lock contention
# with the play-by-play crawls.
DB = __file__.rsplit("/", 1)[0] + "/data/odds.sqlite3"
UA = "nhl-predictor/1.0 (personal research)"

# ESPN's abbreviations differ from the NHL's for a handful of clubs.
ESPN_TO_NHL = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
               "UTAH": "UTA", "UTA": "UTA", "WSH": "WSH", "VGK": "VGK"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS espn_events (
  event_id INTEGER PRIMARY KEY, date TEXT, home TEXT, away TEXT
);
CREATE INDEX IF NOT EXISTS i_ee_date ON espn_events(date);

CREATE TABLE IF NOT EXISTS odds (
  event_id INTEGER PRIMARY KEY, provider TEXT,
  home_ml REAL, away_ml REAL, total REAL, over_odds REAL, under_odds REAL,
  spread REAL, details TEXT
);
CREATE TABLE IF NOT EXISTS odds_done (event_id INTEGER PRIMARY KEY, ok INTEGER);
"""
_local = threading.local()


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        s.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=16))
        _local.s = s
    return s


def get(url, tries=3):
    for a in range(tries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code in (404, 400):
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(1.2 * (a + 1))
    return None


def norm(ab):
    if not ab:
        return None
    ab = ab.upper()
    return ESPN_TO_NHL.get(ab, ab)


def fetch_dates(con, start, end):
    d = start
    rows = []
    while d <= end:
        js = get(SB.format(d.strftime("%Y%m%d")))
        if js:
            for e in js.get("events", []) or []:
                comp = (e.get("competitions") or [{}])[0]
                home = away = None
                for c in comp.get("competitors", []) or []:
                    ab = norm(((c.get("team") or {}).get("abbreviation")))
                    if c.get("homeAway") == "home":
                        home = ab
                    else:
                        away = ab
                if home and away:
                    rows.append((int(e["id"]), d.strftime("%Y-%m-%d"), home, away))
        d += timedelta(days=1)
        if len(rows) % 200 < 2 and rows:
            print(f"  {d} … {len(rows)} events", flush=True)
    con.executemany("INSERT OR REPLACE INTO espn_events VALUES (?,?,?,?)", rows)
    con.commit()
    print(f"events stored: {len(rows)}")


def fetch_odds(con, workers=10):
    todo = [r[0] for r in con.execute(
        "SELECT event_id FROM espn_events WHERE event_id NOT IN "
        "(SELECT event_id FROM odds_done)")]
    print(f"odds to fetch: {len(todo)}", flush=True)

    def work(eid):
        js = get(OD.format(eid))
        if not js:
            return eid, None
        items = js.get("items") or []
        if not items:
            return eid, None
        # Prefer DraftKings; otherwise take whatever provider is listed.
        it = next((x for x in items
                   if "draft" in ((x.get("provider") or {}).get("name", "")).lower()),
                  items[0])
        ho = it.get("homeTeamOdds") or {}
        ao = it.get("awayTeamOdds") or {}
        return eid, (
            eid, (it.get("provider") or {}).get("name"),
            ho.get("moneyLine"), ao.get("moneyLine"),
            it.get("overUnder"), it.get("overOdds"), it.get("underOdds"),
            it.get("spread"), it.get("details"))

    done = got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for eid, row in ex.map(work, todo):
            if row:
                con.execute("INSERT OR REPLACE INTO odds VALUES (?,?,?,?,?,?,?,?,?)", row)
                got += 1
            con.execute("INSERT OR REPLACE INTO odds_done VALUES (?,?)",
                        (eid, 1 if row else 0))
            done += 1
            if done % 250 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}  with odds: {got}", flush=True)
    con.commit()
    print(f"  done {done}, with odds {got}")


if __name__ == "__main__":
    con = sqlite3.connect(DB, timeout=90)
    con.executescript(SCHEMA)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "dates"):
        # 2025-26 season through the current 2026-27 games
        fetch_dates(con, date(2025, 9, 20), date(2026, 10, 14))
    if stage in ("all", "odds"):
        fetch_odds(con)
    print("espn_events:", con.execute("SELECT COUNT(*) FROM espn_events").fetchone()[0])
    print("odds rows:", con.execute("SELECT COUNT(*) FROM odds").fetchone()[0])
