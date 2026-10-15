"""Build one feature row per game using only information available pre-game.

Every rolling quantity is recorded *before* the game it describes is folded
in, so a row for game N only ever contains games 1..N-1. Season boundaries
regress each team's rating partway back to the league mean, which matters a
great deal in October when the current season holds only a handful of games.
"""
import numpy as np
import pandas as pd

# Arena location and standard UTC offset, for travel and body-clock effects.
ARENAS = {
    "ANA": (33.8078, -117.8765, -8), "BOS": (42.3662, -71.0621, -5),
    "BUF": (42.8750, -78.8764, -5),  "CAR": (35.8033, -78.7219, -5),
    "CBJ": (39.9694, -83.0060, -5),  "CGY": (51.0375, -114.0519, -7),
    "CHI": (41.8807, -87.6742, -6),  "COL": (39.7487, -105.0077, -7),
    "DAL": (32.7905, -96.8103, -6),  "DET": (42.3410, -83.0552, -5),
    "EDM": (53.5469, -113.4978, -7), "FLA": (26.1585, -80.3255, -5),
    "LAK": (34.0430, -118.2673, -8), "MIN": (44.9447, -93.1009, -6),
    "MTL": (45.4961, -73.5693, -5),  "NJD": (40.7336, -74.1711, -5),
    "NSH": (36.1593, -86.7785, -6),  "NYI": (40.7118, -73.7260, -5),
    "NYR": (40.7505, -73.9934, -5),  "OTT": (45.2969, -75.9272, -5),
    "PHI": (39.9012, -75.1720, -5),  "PIT": (40.4395, -79.9896, -5),
    "SEA": (47.6221, -122.3540, -8), "SJS": (37.3328, -121.9012, -8),
    "STL": (38.6268, -90.2026, -6),  "TBL": (27.9427, -82.4518, -5),
    "TOR": (43.6435, -79.3791, -5),  "UTA": (40.7683, -111.9011, -7),
    "VAN": (49.2778, -123.1088, -8), "VGK": (36.1028, -115.1783, -8),
    "WPG": (49.8927, -97.1436, -6),  "WSH": (38.8981, -77.0209, -5),
    "ARI": (33.4255, -111.9400, -7),
}

FENWICK = ("shot-on-goal", "goal", "missed-shot")
# Minimum ice time per strength state before a rate is trusted, in hours.
MIN_H_5V5, MIN_H_ST = 0.1, 0.02
# Team-strength stats: EWMA half-life in games, and how much of a rating
# carries across a season boundary (the rest reverts to the league mean).
TEAM_STATS = ["xgf_5v5", "xga_5v5", "cf_5v5", "ca_5v5", "gf_5v5", "ga_5v5",
              "xgf_pp", "gf_pp", "xga_pk", "ga_pk", "gf_all", "ga_all",
              "xgf_all", "xga_all", "sf_5v5", "sa_5v5"]
HALF_LIFE = 20.0
CARRYOVER = 0.70
GOALIE_SHOT_HALFLIFE = 2500.0
GOALIE_SHRINK_K = 1500.0


def haversine(a, b):
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[0]), np.radians(b[1])
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def team_game_stats(games, shots):
    """Aggregate shot rows into one stat row per (game, team)."""
    s = shots.merge(games[["game_id", "home", "away", "home_id", "away_id"]],
                    on="game_id", how="inner")
    s["team"] = np.where(s.is_home == 1, s.home, s.away)
    s["opp"] = np.where(s.is_home == 1, s.away, s.home)
    s["goal"] = (s.event == "goal").astype(int)
    s["fen"] = s.event.isin(FENWICK).astype(int)
    s["sog"] = s.event.isin(("shot-on-goal", "goal")).astype(int)
    s["xg"] = s.xg.fillna(0.0)

    both_g = s.goalie_in_net == 1
    ev5 = (s.shooters_skaters == 5) & (s.defenders_skaters == 5) & both_g
    pp = (s.shooters_skaters > s.defenders_skaters) & both_g
    pk = (s.shooters_skaters < s.defenders_skaters) & both_g
    # Exclude 3-on-3 overtime and empty-net play from "all situations" so the
    # team ratings describe regulation hockey.
    reg = s.period_type == "REG"

    s["w_ev5"], s["w_pp"], s["w_pk"] = ev5 & reg, pp & reg, pk & reg
    s["w_all"] = reg & both_g

    def agg(mask, prefix):
        t = s[mask]
        f = t.groupby(["game_id", "team"]).agg(
            **{f"{prefix}f": ("xg", "sum"), f"g{prefix}f": ("goal", "sum"),
               f"c{prefix}f": ("fen", "sum"), f"s{prefix}f": ("sog", "sum")})
        a = t.groupby(["game_id", "opp"]).agg(
            **{f"{prefix}a": ("xg", "sum"), f"g{prefix}a": ("goal", "sum"),
               f"c{prefix}a": ("fen", "sum"), f"s{prefix}a": ("sog", "sum")})
        a.index.names = ["game_id", "team"]
        return f.join(a, how="outer")

    ev = agg(s.w_ev5, "ev")
    pw = agg(s.w_pp, "pp")
    kl = agg(s.w_pk, "pk")
    al = agg(s.w_all, "al")
    out = ev.join(pw, how="outer").join(kl, how="outer").join(al, how="outer")
    out = out.fillna(0.0).reset_index()

    r = pd.DataFrame({"game_id": out.game_id, "team": out.team})
    r["xgf_5v5"], r["xga_5v5"] = out.evf, out.eva
    r["cf_5v5"], r["ca_5v5"] = out.cevf, out.ceva
    r["sf_5v5"], r["sa_5v5"] = out.sevf, out.seva
    r["gf_5v5"], r["ga_5v5"] = out.gevf, out.geva
    r["xgf_pp"], r["gf_pp"] = out.ppf, out.gppf
    r["xga_pk"], r["ga_pk"] = out.pka, out.gpka
    r["xgf_all"], r["xga_all"] = out.alf, out.ala
    r["gf_all"], r["ga_all"] = out.galf, out.gala
    return r


def goalie_game_stats(games, shots):
    """One row per (game, goalie): workload, xG faced and goals allowed."""
    s = shots[shots.event.isin(FENWICK) & shots.goalie_id.notna()].copy()
    # `shots` already carries season and date; merging them again would
    # collide and produce season_x / season_y.
    s = s.merge(games[["game_id", "home", "away"]], on="game_id", how="inner")
    # The goalie faces shots from the other team.
    s["gteam"] = np.where(s.is_home == 1, s.away, s.home)
    s["goal"] = (s.event == "goal").astype(int)
    s["xg"] = s.xg.fillna(0.0)
    g = s.groupby(["game_id", "season", "date", "goalie_id", "gteam"]).agg(
        shots_faced=("goal", "size"), goals_allowed=("goal", "sum"),
        xg_faced=("xg", "sum")).reset_index()
    g = g.rename(columns={"gteam": "team", "goalie_id": "goalie"})
    g["goalie"] = g.goalie.astype("int64")
    # Starter = the goalie who faced the most shots for that team.
    g = g.sort_values(["game_id", "team", "shots_faced"],
                      ascending=[True, True, False])
    g["is_starter"] = (g.groupby(["game_id", "team"]).cumcount() == 0).astype(int)
    return g


def to_per_60(tg, team_toi, impute=False):
    """Convert counting stats to rates per 60 minutes of that strength state.

    Shift charts give real time on ice, so 5v5 and special-teams stats no
    longer have to be treated as per-game. A team that spends 10 minutes
    shorthanded is no longer credited with the same denominator as one that
    spends two.

    The NHL does not publish shift charts for every game -- 104 of the games
    held here have none, including the first 52 of 2026-27. Dropping those
    team-games (the original behaviour) does two kinds of damage: the game
    itself loses its ratings, and every later game loses the history, so a
    team's rating goes stale for as long as the gap lasts. Imputing the
    denominator instead keeps the rating current. The split of a game into
    strength states is a league-level constant, so the median from the
    pre-holdout seasons is used; nothing game-specific is borrowed.
    """
    m = tg.merge(team_toi, on=["game_id", "team"],
                 how="left" if impute else "inner")
    cols = ["toi_5v5", "toi_pp", "toi_pk"]
    if impute:
        early = m[(m.game_id // 1000000) < 2025]
        for c in cols:
            fill = early[c].median()
            if not np.isfinite(fill):
                fill = m[c].median()
            m[c] = m[c].fillna(fill)
    out = m.copy()
    h5 = np.maximum(m.toi_5v5 / 3600, MIN_H_5V5)
    for c in ("xgf_5v5", "xga_5v5", "cf_5v5", "ca_5v5", "gf_5v5", "ga_5v5",
              "sf_5v5", "sa_5v5"):
        out[c] = m[c] / h5
    hpp = np.maximum(m.toi_pp / 3600, MIN_H_ST)
    hpk = np.maximum(m.toi_pk / 3600, MIN_H_ST)
    out["xgf_pp"] = m.xgf_pp / hpp
    out["gf_pp"] = m.gf_pp / hpp
    out["xga_pk"] = m.xga_pk / hpk
    out["ga_pk"] = m.ga_pk / hpk
    # All-situations stats are already per game, which is the right unit:
    # every team plays 60 minutes of regulation.
    return out[["game_id", "team"] + TEAM_STATS]


def roll_team(games, tg, carry=True):
    """Pre-game EWMA ratings per team, with season-boundary regression.

    `carry` decides what happens to a team-game whose stats are missing --
    the NHL publishes no shift chart for 104 of the games held here, so the
    per-60 table has no row for them. Dropping the team-game (carry=False,
    the original behaviour) means the *game* gets no rating either, and
    `build_matrix` then fills the hole with a literal zero, which after
    standardisation is a far more extreme input than any real team. With
    carry=True the rating is still emitted from the state carried forward,
    and only the update is skipped: a stale rating instead of an absurd one.
    """
    if carry:
        long = pd.concat([
            games[["game_id", "season", "date", "home", "away"]].assign(
                team=games[side]) for side in ("home", "away")])
        d = long.merge(tg, on=["game_id", "team"], how="left")
    else:
        d = tg.merge(games[["game_id", "season", "date", "home", "away"]],
                     on="game_id")
    d = d.sort_values(["date", "game_id"]).reset_index(drop=True)
    league = {c: d[c].mean() for c in TEAM_STATS}
    alpha = 1 - 0.5 ** (1 / HALF_LIFE)

    state, season_of, rows = {}, {}, []
    for r in d.itertuples(index=False):
        t = r.team
        if t not in state or season_of.get(t) != r.season:
            if t in state:   # new season: pull partway back to league mean
                state[t] = {c: league[c] + CARRYOVER * (state[t][c] - league[c])
                            for c in TEAM_STATS}
            else:
                state[t] = dict(league)
            season_of[t] = r.season
        cur = state[t]
        rows.append({"game_id": r.game_id, "team": t,
                     **{f"r_{c}": cur[c] for c in TEAM_STATS}})
        if pd.isna(getattr(r, TEAM_STATS[0])):
            continue                      # no stats for this game: hold
        for c in TEAM_STATS:
            cur[c] = (1 - alpha) * cur[c] + alpha * getattr(r, c)
    return pd.DataFrame(rows), state, league


def roll_goalie(gg):
    """Pre-game goalie rating: shrunk goals-saved-above-expected per shot."""
    g = gg.sort_values(["date", "game_id"]).reset_index(drop=True)
    state, rows = {}, []
    for r in g.itertuples(index=False):
        st = state.setdefault(r.goalie, {"n": 0.0, "xg": 0.0, "ga": 0.0})
        n, xg, ga = st["n"], st["xg"], st["ga"]
        gsax = (xg - ga) / n if n > 0 else 0.0
        rows.append({"game_id": r.game_id, "goalie": r.goalie,
                     "g_gsax_raw": gsax,
                     "g_gsax": gsax * (n / (n + GOALIE_SHRINK_K)),
                     "g_shots_seen": n})
        decay = 0.5 ** (r.shots_faced / GOALIE_SHOT_HALFLIFE)
        st["n"] = n * decay + r.shots_faced
        st["xg"] = xg * decay + r.xg_faced
        st["ga"] = ga * decay + r.goals_allowed
    return pd.DataFrame(rows), state


def schedule_features(games):
    """Rest, back-to-backs, travel distance and time-zone shift per team-game."""
    long = pd.concat([
        games[["game_id", "date", "home", "away", "season"]].assign(
            team=lambda x: x.home, is_home=1, venue=lambda x: x.home),
        games[["game_id", "date", "home", "away", "season"]].assign(
            team=lambda x: x.away, is_home=0, venue=lambda x: x.home),
    ])[["game_id", "date", "season", "team", "is_home", "venue"]]
    long["d"] = pd.to_datetime(long.date)
    long = long.sort_values(["team", "d", "game_id"]).reset_index(drop=True)

    out = []
    for team, grp in long.groupby("team", sort=False):
        prev_d = prev_v = None
        prev_season = None
        hist = []
        for r in grp.itertuples(index=False):
            if prev_season != r.season:
                prev_d = prev_v = None
                hist = []
            rest = (r.d - prev_d).days if prev_d is not None else 7
            rest = min(rest, 10)
            dist = tz = 0.0
            if prev_v is not None and prev_v in ARENAS and r.venue in ARENAS:
                a, b = ARENAS[prev_v], ARENAS[r.venue]
                dist = haversine(a, b)
                tz = b[2] - a[2]
            n4 = sum(1 for h in hist if 0 < (r.d - h).days <= 4)
            out.append({"game_id": r.game_id, "team": team, "rest": rest,
                        "b2b": int(rest <= 1), "games_last4": n4,
                        "travel_km": dist, "tz_shift": tz})
            hist.append(r.d)
            prev_d, prev_v, prev_season = r.d, r.venue, r.season
    return pd.DataFrame(out)


def assemble(games, tg, gg, rt, rg, sf):
    """Join per-team pre-game state into one row per game (home vs away)."""
    starters = gg[gg.is_starter == 1][["game_id", "team", "goalie",
                                       "shots_faced"]]
    starters = starters.merge(rg, on=["game_id", "goalie"], how="left")

    def side(abbrev_col, pre):
        a = games[["game_id", abbrev_col]].rename(columns={abbrev_col: "team"})
        a = a.merge(rt, on=["game_id", "team"], how="left")
        a = a.merge(sf, on=["game_id", "team"], how="left")
        a = a.merge(starters, on=["game_id", "team"], how="left")
        a = a.drop(columns=["team"])
        return a.rename(columns={c: f"{pre}_{c}" for c in a.columns
                                 if c != "game_id"})

    df = games.merge(side("home", "h"), on="game_id") \
              .merge(side("away", "a"), on="game_id")

    # Targets
    df["home_goals"] = df.home_score
    df["away_goals"] = df.away_score
    df["home_win"] = (df.home_score > df.away_score).astype(int)
    df["went_ot"] = df.last_period_type.isin(["OT", "SO"]).astype(int)
    # Regulation score: an OT/SO winner's final goal came after regulation.
    df["h_reg"] = df.home_goals - ((df.went_ot == 1) & (df.home_win == 1))
    df["a_reg"] = df.away_goals - ((df.went_ot == 1) & (df.home_win == 0))
    df["reg_outcome"] = np.select(
        [df.h_reg > df.a_reg, df.h_reg < df.a_reg], [2, 0], default=1)
    df["total_goals"] = df.home_goals + df.away_goals
    return df
