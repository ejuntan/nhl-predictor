"""Fit the shot-level xG model and store a value for every shot."""
import pickle
import sqlite3

import pandas as pd

from xg import apply_xg, fit_xg

if __name__ == "__main__":
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    shots = pd.read_sql("SELECT * FROM shots", con)
    xg = fit_xg(shots, train_seasons={"20222023", "20232024", "20242025"},
                valid_seasons={"20252026", "20262027"})
    shots["xg"] = apply_xg(xg, shots)
    pickle.dump(xg, open("data/xg_model.pkl", "wb"))
    shots[["game_id", "xg"]].to_sql("shot_xg", con, if_exists="replace",
                                    index=False)
    con.execute("CREATE INDEX IF NOT EXISTS i_sx ON shot_xg(game_id)")
    con.commit()
    print("saved data/xg_model.pkl")
