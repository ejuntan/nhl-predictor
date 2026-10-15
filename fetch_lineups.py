"""Pull confirmed starting goalies and scratches for every played game.

Two facts the play-by-play does not carry:

  starter   -- boxscore marks exactly one goalie per team `starter: true`.
               This is authoritative, where the earlier heuristic (the goalie
               who faced the most shots) was a guess.
  scratches -- right-rail lists the players held out of each game, which is
               the only free record of who was actually unavailable.

Resumable: a game already in lineups_done is skipped.
"""
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

BOX = "https://api-web.nhle.com/v1/gamecenter/{}/boxscore"
RAIL = "https://api-web.nhle.com/v1/gamecenter/{}/right-rail"
DB = __file__.rsplit("/", 1)[0] + "/data/nhl.sqlite3"
UA = "nhl-predictor/1.0 (personal research)"

SCHEMA = """
CREATE TABLE IF NOT EXISTS goalie_official (
  game_id INTEGER, team TEXT, is_home INTEGER, player_id INTEGER,
  starter INTEGER, toi_s INTEGER, decision TEXT, saves INTEGER, shots INTEGER
);
CREATE INDEX IF NOT EXISTS i_go_game ON goalie_official(game_id);
CREATE INDEX IF NOT EXISTS i_go_player ON goalie_official(player_id);

CREATE TABLE IF NOT EXISTS scratches (
  game_id INTEGER, team TEXT, is_home INTEGER, player_id INTEGER
);
CREATE INDEX IF NOT EXISTS i_sc_game ON scratches(game_id);
CREATE INDEX IF NOT EXISTS i_sc_player ON scratches(player_id);

CREATE TABLE IF NOT EXISTS lineups_done (
  game_id INTEGER PRIMARY KEY, n_goalies INTEGER, n_scratches INTEGER
);
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
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(1.2 * (a + 1))
    return None


def mmss(t):
    if not t or ":" not in str(t):
        return 0
    m, s = str(t).split(":")[:2]
    try:
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def parse(gid, box, rail):
    goalies, scr = [], []
    if box:
        pbg = box.get("playerByGameStats") or {}
        for side, is_home in (("awayTeam", 0), ("homeTeam", 1)):
            team = (box.get(side) or {}).get("abbrev")
            for g in (pbg.get(side) or {}).get("goalies", []) or []:
                sa = str(g.get("saveShotsAgainst") or "")
                sv, sh = (sa.split("/") + ["", ""])[:2]
                goalies.append((
                    gid, team, is_home, g.get("playerId"),
                    1 if g.get("starter") else 0, mmss(g.get("toi")),
                    g.get("decision"),
                    int(sv) if sv.isdigit() else None,
                    int(sh) if sh.isdigit() else None))
    if rail:
        gi = rail.get("gameInfo") or {}
        for side, is_home in (("awayTeam", 0), ("homeTeam", 1)):
            team = (box.get(side) or {}).get("abbrev") if box else None
            for p in (gi.get(side) or {}).get("scratches", []) or []:
                scr.append((gid, team, is_home, p.get("id")))
    return goalies, scr


def main(workers=10):
    con = sqlite3.connect(DB, timeout=90)
    con.executescript(SCHEMA)
    todo = [r[0] for r in con.execute(
        "SELECT game_id FROM games WHERE state IN ('OFF','FINAL') "
        "AND game_id NOT IN (SELECT game_id FROM lineups_done) ORDER BY date")]
    print(f"games to fetch: {len(todo)}", flush=True)

    def work(gid):
        return gid, parse(gid, get(BOX.format(gid)), get(RAIL.format(gid)))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gid, (goalies, scr) in ex.map(work, todo):
            if goalies:
                con.executemany(
                    "INSERT INTO goalie_official VALUES (?,?,?,?,?,?,?,?,?)",
                    goalies)
            if scr:
                con.executemany("INSERT INTO scratches VALUES (?,?,?,?)", scr)
            con.execute("INSERT OR REPLACE INTO lineups_done VALUES (?,?,?)",
                        (gid, len(goalies), len(scr)))
            done += 1
            if done % 250 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}", flush=True)
    con.commit()
    print(f"  done {done}/{len(todo)}")
    for t in ("goalie_official", "scratches"):
        print(f"{t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")


if __name__ == "__main__":
    main()
