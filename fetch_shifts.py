"""Pull shift charts and derive time on ice, including by strength state.

The public play-by-play has no time-on-ice, which forced two compromises:
team rates were per game rather than per 60 minutes, and player workload had
to be guessed from shot counts. Shift charts fix both. Only the derived
per-game aggregates are stored; the raw JSON is parsed and discarded.
"""
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

API = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"
DB = __file__.rsplit("/", 1)[0] + "/data/nhl.sqlite3"
UA = "nhl-predictor/1.0 (personal research)"
SHIFT = 517            # typeCode for a real shift; 505 rows are goal markers

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_toi (
  game_id INTEGER, player_id INTEGER, team TEXT, is_goalie INTEGER,
  toi_total INTEGER, toi_reg INTEGER, toi_5v5 INTEGER, toi_pp INTEGER,
  toi_pk INTEGER, shifts INTEGER
);
CREATE INDEX IF NOT EXISTS i_pt_game ON player_toi(game_id);
CREATE INDEX IF NOT EXISTS i_pt_player ON player_toi(player_id);

CREATE TABLE IF NOT EXISTS team_toi (
  game_id INTEGER, team TEXT, toi_reg INTEGER, toi_5v5 INTEGER,
  toi_pp INTEGER, toi_pk INTEGER, toi_other INTEGER
);
CREATE INDEX IF NOT EXISTS i_tt_game ON team_toi(game_id);

CREATE TABLE IF NOT EXISTS shifts_done (game_id INTEGER PRIMARY KEY, n INTEGER);
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


def get(url, tries=4):
    for a in range(tries):
        try:
            r = _session().get(url, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(1.5 * (a + 1))
    return None


def mmss(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def parse(gid, rows, goalies):
    """Derive per-player and per-team TOI, resolving strength state."""
    shifts = []
    for r in rows:
        if r.get("typeCode") != SHIFT or not r.get("duration"):
            continue
        per = r.get("period")
        st, en = r.get("startTime"), r.get("endTime")
        if not per or not st or not en or per >= 5:
            continue          # period 5 is the shootout
        a = (per - 1) * 1200 + mmss(st)
        b = (per - 1) * 1200 + mmss(en)
        if b <= a:
            continue
        shifts.append((r["playerId"], r["teamAbbrev"], per, a, b))
    if not shifts:
        return [], []

    teams = sorted({s[1] for s in shifts})
    if len(teams) != 2:
        return [], []

    # A player with more than 50 minutes is a goalie even if we have not seen
    # them in net; this backstops the known-goalie set for the odd game.
    tot = {}
    for pid, tm, per, a, b in shifts:
        tot[pid] = tot.get(pid, 0) + (b - a)
    is_g = {pid: int(pid in goalies or tot[pid] > 3000) for pid in tot}

    # Skater on/off events per team -> a step function of skaters on ice.
    ev = {t: [] for t in teams}
    for pid, tm, per, a, b in shifts:
        if is_g.get(pid):
            continue
        ev[tm].append((a, 1))
        ev[tm].append((b, -1))
    for t in teams:
        ev[t].sort()

    marks = sorted({x for t in teams for x, _ in ev[t]})
    if not marks:
        return [], []

    # Walk the timeline, recording each team's skater count per interval.
    idx = {t: 0 for t in teams}
    cnt = {t: 0 for t in teams}
    intervals = []
    for i, m in enumerate(marks):
        for t in teams:
            while idx[t] < len(ev[t]) and ev[t][idx[t]][0] <= m:
                cnt[t] += ev[t][idx[t]][1]
                idx[t] += 1
        if i + 1 < len(marks):
            nxt = marks[i + 1]
            if nxt > m:
                intervals.append((m, nxt, cnt[teams[0]], cnt[teams[1]]))

    t0, t1 = teams
    tt = {t: {"reg": 0, "5v5": 0, "pp": 0, "pk": 0, "other": 0} for t in teams}
    state = []          # (start, end, n0, n1) limited to regulation
    for a, b, n0, n1 in intervals:
        dur = b - a
        if a >= 3600:
            continue            # overtime: 3-on-3, not comparable
        if b > 3600:
            dur = 3600 - a
            b = 3600
        if dur <= 0:
            continue
        state.append((a, b, n0, n1))
        for t, me, you in ((t0, n0, n1), (t1, n1, n0)):
            tt[t]["reg"] += dur
            if me == 5 and you == 5:
                tt[t]["5v5"] += dur
            elif me > you:
                tt[t]["pp"] += dur
            elif me < you:
                tt[t]["pk"] += dur
            else:
                tt[t]["other"] += dur

    # Per-player: total, regulation, and time spent in each strength state.
    pl = {}
    for pid, tm, per, a, b in shifts:
        d = pl.setdefault(pid, {"team": tm, "tot": 0, "reg": 0, "ev": 0,
                                "pp": 0, "pk": 0, "n": 0})
        d["n"] += 1
        d["tot"] += b - a
        lo, hi = a, min(b, 3600)
        if hi <= lo:
            continue
        d["reg"] += hi - lo
        for sa, sb, n0, n1 in state:
            ov = min(hi, sb) - max(lo, sa)
            if ov <= 0:
                continue
            me, you = (n0, n1) if tm == t0 else (n1, n0)
            if me == 5 and you == 5:
                d["ev"] += ov
            elif me > you:
                d["pp"] += ov
            elif me < you:
                d["pk"] += ov

    prows = [(gid, pid, d["team"], is_g.get(pid, 0), d["tot"], d["reg"],
              d["ev"], d["pp"], d["pk"], d["n"]) for pid, d in pl.items()]
    trows = [(gid, t, v["reg"], v["5v5"], v["pp"], v["pk"], v["other"])
             for t, v in tt.items()]
    return prows, trows


def main(workers=10):
    con = sqlite3.connect(DB, timeout=90)
    con.executescript(SCHEMA)
    goalies = {r[0] for r in con.execute(
        "SELECT DISTINCT goalie_id FROM shots WHERE goalie_id IS NOT NULL")}
    print(f"known goalies: {len(goalies)}")
    todo = [r[0] for r in con.execute(
        "SELECT game_id FROM games WHERE state IN ('OFF','FINAL') "
        "AND game_id NOT IN (SELECT game_id FROM shifts_done) ORDER BY date")]
    print(f"shift charts to fetch: {len(todo)}", flush=True)

    def work(gid):
        try:
            d = get(API.format(gid))
            if not d:
                return gid, [], []
            p, t = parse(gid, d.get("data") or [], goalies)
            return gid, p, t
        except Exception as e:
            print(f"  !! {gid}: {e}", flush=True)
            return gid, None, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gid, p, t in ex.map(work, todo):
            if p is None:
                continue
            if p:
                con.executemany(
                    "INSERT INTO player_toi VALUES (?,?,?,?,?,?,?,?,?,?)", p)
            if t:
                con.executemany(
                    "INSERT INTO team_toi VALUES (?,?,?,?,?,?,?)", t)
            con.execute("INSERT OR REPLACE INTO shifts_done VALUES (?,?)",
                        (gid, len(p)))
            done += 1
            if done % 250 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}", flush=True)
    con.commit()
    print(f"  done {done}/{len(todo)}")
    print("player_toi rows:",
          con.execute("SELECT COUNT(*) FROM player_toi").fetchone()[0])
    print("team_toi rows:",
          con.execute("SELECT COUNT(*) FROM team_toi").fetchone()[0])


if __name__ == "__main__":
    main()
