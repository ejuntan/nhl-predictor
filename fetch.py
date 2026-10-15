"""Pull NHL game metadata and shot-level play-by-play into a local SQLite store.

Only the extracted rows are kept -- raw JSON is parsed and discarded, so the
store stays small (~1M shot rows for five seasons).

Two stages:
  games  -- 32 requests per season via club-schedule-season
  shots  -- 1 request per game via gamecenter/{id}/play-by-play

Both stages are resumable: a game already present in `shots_done` is skipped,
so re-running after an interruption costs nothing.
"""
import json
import sqlite3
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

API = "https://api-web.nhle.com/v1"
DB = __file__.rsplit("/", 1)[0] + "/data/nhl.sqlite3"
UA = "nhl-predictor/1.0 (personal research; contact via github)"

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]
# Utah joined for 2024-25 (replacing ARI); ARI is needed for earlier seasons.
LEGACY = {"20222023": ["ARI"], "20232024": ["ARI"]}

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
  game_id INTEGER PRIMARY KEY, season TEXT, game_type INTEGER, date TEXT,
  start_utc TEXT, home TEXT, away TEXT, home_id INTEGER, away_id INTEGER,
  home_score INTEGER, away_score INTEGER, last_period_type TEXT,
  ot_periods INTEGER, neutral INTEGER, state TEXT
);
CREATE INDEX IF NOT EXISTS i_games_date ON games(date);
CREATE INDEX IF NOT EXISTS i_games_season ON games(season);

CREATE TABLE IF NOT EXISTS shots (
  game_id INTEGER, season TEXT, date TEXT, event TEXT, period INTEGER,
  period_type TEXT, time_s INTEGER, team_id INTEGER, is_home INTEGER,
  shooter_id INTEGER, goalie_id INTEGER, x REAL, y REAL, adj_x REAL,
  adj_y REAL, shot_type TEXT, zone TEXT, sit_code TEXT,
  shooters_skaters INTEGER, defenders_skaters INTEGER, goalie_in_net INTEGER,
  prev_event TEXT, prev_dt REAL, prev_dist REAL
);
CREATE INDEX IF NOT EXISTS i_shots_game ON shots(game_id);
CREATE INDEX IF NOT EXISTS i_shots_date ON shots(date);
CREATE INDEX IF NOT EXISTS i_shots_goalie ON shots(goalie_id);

CREATE TABLE IF NOT EXISTS shots_done (game_id INTEGER PRIMARY KEY, n INTEGER);
"""

FENWICK = {"shot-on-goal", "goal", "missed-shot"}
TRACKED = FENWICK | {"blocked-shot"}


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
    for attempt in range(tries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def mmss(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def fetch_games(con, seasons):
    rows = {}
    for season in seasons:
        teams = TEAMS + LEGACY.get(season, [])
        for t in teams:
            d = get(f"{API}/club-schedule-season/{t}/{season}")
            if not d:
                continue
            for g in d.get("games", []):
                if g.get("gameType") not in (2, 3):
                    continue           # skip preseason/exhibition
                ht, at = g["homeTeam"], g["awayTeam"]
                out = g.get("gameOutcome") or {}
                rows[g["id"]] = (
                    g["id"], season, g["gameType"], g["gameDate"],
                    g.get("startTimeUTC"), ht["abbrev"], at["abbrev"],
                    ht.get("id"), at.get("id"),
                    ht.get("score"), at.get("score"),
                    out.get("lastPeriodType"), out.get("otPeriods") or 0,
                    1 if g.get("neutralSite") else 0, g.get("gameState"),
                )
        print(f"  {season}: {len(rows)} cumulative games", flush=True)
    con.executemany(
        "INSERT OR REPLACE INTO games VALUES (%s)" % ",".join("?" * 15),
        list(rows.values()))
    con.commit()
    return len(rows)


def parse_pbp(d):
    """Extract one row per tracked shot attempt."""
    gid = d["id"]
    season = str(d["season"])
    date = d["gameDate"]
    home_id = d["homeTeam"]["id"]
    plays = d.get("plays") or []

    # Pass 1: infer which x-sign each team attacks, per period, from
    # offensive-zone shots. More robust than homeTeamDefendingSide, which is
    # missing in some older games.
    attack = {}
    for p in plays:
        det = p.get("details") or {}
        if p.get("typeDescKey") not in FENWICK or det.get("zoneCode") != "O":
            continue
        x = det.get("xCoord")
        if x is None or x == 0:
            continue
        key = (p["periodDescriptor"]["number"], det.get("eventOwnerTeamId"))
        attack.setdefault(key, []).append(1 if x > 0 else -1)
    attack = {k: (1 if sum(v) > 0 else -1) for k, v in attack.items()}

    rows = []
    prev = None
    for p in plays:
        tdk = p.get("typeDescKey")
        det = p.get("details") or {}
        pd_ = p.get("periodDescriptor") or {}
        per = pd_.get("number")
        tip = p.get("timeInPeriod")
        if per is None or not tip:
            prev = None
            continue
        t_s = (per - 1) * 1200 + mmss(tip)
        x, y = det.get("xCoord"), det.get("yCoord")

        if tdk not in TRACKED:
            if tdk in ("faceoff", "hit", "giveaway", "takeaway", "stoppage",
                       "penalty", "period-start"):
                prev = (tdk, t_s, x, y)
            continue

        team = det.get("eventOwnerTeamId")
        is_home = 1 if team == home_id else 0
        sit = det.get("situationCode") or p.get("situationCode") or ""
        # situationCode digits: away goalie, away skaters, home skaters, home goalie
        sk_sh = sk_def = None
        g_in = 1
        if len(sit) == 4 and sit.isdigit():
            a_g, a_sk, h_sk, h_g = (int(c) for c in sit)
            sk_sh, sk_def = (h_sk, a_sk) if is_home else (a_sk, h_sk)
            g_in = a_g if is_home else h_g      # goalie facing this shot

        side = attack.get((per, team))
        if side is None:
            hds = p.get("homeTeamDefendingSide")
            if hds in ("left", "right"):
                home_net = 1 if hds == "right" else -1
                side = -home_net if is_home else home_net
            elif x is not None:
                side = 1 if (x or 0) >= 0 else -1
            else:
                side = 1
        adj_x = None if x is None else x * side
        adj_y = None if y is None else (y or 0) * side

        p_ev, p_dt, p_dist = None, None, None
        if prev and prev[1] is not None:
            p_ev = prev[0]
            p_dt = t_s - prev[1]
            if prev[2] is not None and x is not None:
                p_dist = ((x - prev[2]) ** 2 + ((y or 0) - (prev[3] or 0)) ** 2) ** 0.5

        rows.append((
            gid, season, date, tdk, per, pd_.get("periodType"), t_s, team,
            is_home, det.get("shootingPlayerId") or det.get("scoringPlayerId"),
            det.get("goalieInNetId"), x, y, adj_x, adj_y,
            det.get("shotType"), det.get("zoneCode"), sit, sk_sh, sk_def,
            g_in, p_ev, p_dt, p_dist,
        ))
        prev = (tdk, t_s, x, y)
    return rows


def fetch_shots(con, workers=10):
    todo = [r[0] for r in con.execute(
        "SELECT game_id FROM games WHERE state IN ('OFF','FINAL') "
        "AND game_id NOT IN (SELECT game_id FROM shots_done) ORDER BY date")]
    total = len(todo)
    print(f"play-by-play to fetch: {total}", flush=True)

    def work(gid):
        """Network only -- all SQLite writes happen on the main thread."""
        try:
            d = get(f"{API}/gamecenter/{gid}/play-by-play")
            return gid, (parse_pbp(d) if d else [])
        except Exception as e:
            print(f"  !! {gid}: {e}", flush=True)
            return gid, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gid, rows in ex.map(work, todo):
            if rows is None:
                continue
            if rows:
                con.executemany(
                    "INSERT INTO shots VALUES (%s)" % ",".join("?" * 24), rows)
            con.execute("INSERT OR REPLACE INTO shots_done VALUES (?,?)",
                        (gid, len(rows)))
            done += 1
            if done % 250 == 0:
                con.commit()
                print(f"  {done}/{total}", flush=True)
    con.commit()
    print(f"  done {done}/{total}", flush=True)


if __name__ == "__main__":
    seasons = ["20222023", "20232024", "20242025", "20252026", "20262027"]
    con = sqlite3.connect(DB, timeout=60)
    con.executescript(SCHEMA)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "games"):
        print("== games ==", flush=True)
        fetch_games(con, seasons)
    if stage in ("all", "shots"):
        print("== shots ==", flush=True)
        fetch_shots(con)
    for s, n, d in con.execute(
            "SELECT season, COUNT(*), MAX(date) FROM games GROUP BY season"):
        print(f"{s}: {n} games, latest {d}")
    print("shot rows:", con.execute("SELECT COUNT(*) FROM shots").fetchone()[0])
