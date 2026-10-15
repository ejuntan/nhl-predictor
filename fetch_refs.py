"""Pull the officiating crew for every played game.

The only pre-game signal left untested for totals. Referees differ in how
many penalties they call, penalties create power plays, and power plays are
where about a fifth of goals come from -- so a crew's tendency is a
plausible source of the between-game variation in scoring that team ratings
do not explain.

The names are in the right-rail endpoint already crawled for scratches, so
this is a re-read of a known endpoint rather than new data.

Note before building anything on this: the NHL does not publish the crew in
its schedule feed, so this is available after the fact but not reliably
before puck drop. Measure first, and only then worry about sourcing it live.
"""
import sqlite3

from fetch_lineups import RAIL, get
from concurrent.futures import ThreadPoolExecutor

DB = __file__.rsplit("/", 1)[0] + "/data/nhl.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS officials (
  game_id INTEGER, role TEXT, name TEXT, number INTEGER
);
CREATE INDEX IF NOT EXISTS i_off_game ON officials(game_id);
CREATE INDEX IF NOT EXISTS i_off_name ON officials(name);
CREATE TABLE IF NOT EXISTS officials_done (game_id INTEGER PRIMARY KEY, n INTEGER);
"""


def parse(gid, rail):
    if not rail:
        return []
    gi = rail.get("gameInfo") or {}
    rows = []
    for role in ("referees", "linesmen"):
        for o in gi.get(role) or []:
            nm = (o.get("fullName") or {}).get("default")
            if nm:
                rows.append((gid, role[:-1], nm, o.get("sweaterNumber")))
    return rows


def main(workers=10):
    con = sqlite3.connect(DB, timeout=90)
    con.executescript(SCHEMA)
    todo = [r[0] for r in con.execute(
        "SELECT game_id FROM games WHERE state IN ('OFF','FINAL') "
        "AND game_type = 2 "
        "AND game_id NOT IN (SELECT game_id FROM officials_done) "
        "ORDER BY date")]
    print(f"games to fetch: {len(todo)}", flush=True)

    def work(gid):
        return gid, parse(gid, get(RAIL.format(gid)))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for gid, rows in ex.map(work, todo):
            if rows:
                con.executemany("INSERT INTO officials VALUES (?,?,?,?)", rows)
            con.execute("INSERT OR REPLACE INTO officials_done VALUES (?,?)",
                        (gid, len(rows)))
            done += 1
            if done % 500 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}", flush=True)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM officials").fetchone()[0]
    refs = con.execute(
        "SELECT COUNT(DISTINCT name) FROM officials WHERE role='referee'"
    ).fetchone()[0]
    cov = con.execute(
        "SELECT COUNT(*) FROM officials_done WHERE n > 0").fetchone()[0]
    print(f"officials rows {n}; distinct referees {refs}; "
          f"games with a crew {cov}/{done}")


if __name__ == "__main__":
    main()
