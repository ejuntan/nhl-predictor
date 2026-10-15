"""Rebuild the shots-on-goal model's caches.

These two files were produced interactively while the props model was being
developed, which meant nothing in the repository could regenerate them --
the daily refresh updated the game predictions and left the props half of
the page frozen. This closes that gap.

  data/player_games.pkl   one row per skater-game with rolling features
  data/props_preds.pkl    walk-forward predictions, used for the page's
                          past-games panels and the market grades
"""
import sqlite3

import props_model as P

DB = "data/nhl.sqlite3"


def main():
    con = sqlite3.connect(DB, timeout=60)
    pg = P.build(con)
    pg.to_pickle("data/player_games.pkl")
    pr = P.run(pg)
    pr.to_pickle("data/props_preds.pkl")
    print(f"props predictions {len(pr)} rows, "
          f"through {pr.date.max()}")
    return pr


if __name__ == "__main__":
    main()
