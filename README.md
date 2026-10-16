# NHL game model

Predicts NHL game outcomes from the league's public play-by-play: win
probability, regulation three-way, goal totals and the puck line. See
[MODEL.md](MODEL.md) for the method and the honest scorecard.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Pipeline

```bash
./.venv/bin/python fetch.py          # games + play-by-play -> data/nhl.sqlite3
./.venv/bin/python fetch_shifts.py   # shift charts -> time on ice by strength
./.venv/bin/python fit_xg.py         # shot-level xG model
./.venv/bin/python build_features.py # leak-free per-game features
./.venv/bin/python backtest.py       # walk-forward evaluation
./.venv/bin/python analyze.py        # ablation + subset skill
./.venv/bin/python ceiling.py        # how much headroom is left
./.venv/bin/python live.py           # game predictions
./.venv/bin/python live_props.py     # shots-on-goal prices
./.venv/bin/python make_csvs.py      # dashboard datasets -> out/
./.venv/bin/python make_csvs2.py
```

Or just `./daily.sh`, which runs the whole chain in dependency order in
about three minutes: new results, features, the walk-forward backtest that
produces the page's finished-game predictions, the shots-on-goal caches,
tomorrow's predictions, live prices for the value finder, the page itself,
all four test suites, and the frozen prediction log. It uses `set -e`, so a
non-zero exit means a named stage failed and nothing downstream ran.
`refresh.sh` is the older, shorter path and does not rebuild the page.

**Daily, unattended.** A scheduled task (`nhl-daily-refresh`, in the Claude
app's Scheduled section) runs `daily.sh` at 09:09 and republishes the page
to its artifact URL, which is the part a shell script cannot do. It only
runs while the app is open, and otherwise catches up at next launch. For a
refresh that does not depend on the app, `com.tanejun.nhl-daily.plist`
installs a LaunchAgent at 08:30 — it keeps the data and `tool/index.html`
current but cannot republish.

`fetch.py` is resumable — games already in `shots_done` are skipped, so
re-running it each day costs only the new games.

## Files

| File | What it does |
|---|---|
| `fetch.py` | NHL API -> SQLite. Stage `games` or `shots`. |
| `fetch_shifts.py` | Shift charts -> per-player and per-team time on ice, with strength state reconstructed from on-ice counts. |
| `xg.py` | Shot-level expected-goals model. |
| `features.py` | Rolling team/goalie ratings, schedule and travel. |
| `model.py` | Poisson score grid, Dixon-Coles, tie inflation, OT model, Elo and Log5 baselines. |
| `backtest.py` | Walk-forward backtest and the metrics tables. |
| `analyze.py` | Leave-one-group-out ablation and subset skill. |
| `live.py` | Fits on everything completed, predicts the next 8 days. |
| `props_model.py` | Negative-binomial shots-on-goal model, ice time as exposure. |
| `live_props.py` | Prop prices for upcoming games. |
| `saves_model.py`, `saves_v2.py` | Goalie saves, and the level correction that made it priceable. |
| `live_saves.py` | Saves prices for the probable starter in each upcoming game. |
| `totals_eval.py`, `totals_improve.py`, `totals_direct.py` | Whether game totals carry any information (they do not). |
| `fetch_refs.py`, `refs_eval.py` | Officiating crews, and whether they move the total (they do not). |
| `ceiling.py` | Oracle study: what each kind of information is worth. |
| `adjust.py` | Opponent-adjusted team ratings (measured null; kept for reference). |
| `experiment.py`, `sig.py` | Variant comparison on validation, with paired significance tests. |
| `variants.py` | Builds every feature-frame version in one pass, on one identical game list. |
| `compare.py` | Scores all versions walk-forward; paired bootstrap over games. |
| `flips.py` | How far each change moves the probabilities, and what it decides. |
| `errors.py` | Mistakes by situation, per season, against the market. |
| `widen.py` | Tests widening the distribution when the starting goalie is unusual. |
| `freeze.py` | Freezes the model, then logs predictions before puck drop. |
| `fetch_odds.py`, `odds_eval.py` | Archived closing prices, de-vigged, and the profitability tests. |
| `fetch_lineups.py`, `lineups.py` | Confirmed starters and scratches; absence features. |
| `build_props.py` | Rebuilds the shots-on-goal caches (nothing used to). |
| `edge_test.py` | Prices every situation at real posted odds: the bucket returns. |
| `live_edge.py` | Live prices for the upcoming slate, labelled by measured bucket. |
| `daily.sh` | The whole chain, in order, with tests. |
| `make_csvs.py` | Writes `out/*.csv` for the dashboard. |

## The predictions page

`tool/` holds a self-contained page, organised one day at a time.

**Finished days** show what the model predicted against what happened: the
final score, whether the pick was right, predicted versus actual total, and
per-skater predicted versus actual shots. Each past day in the picker carries
its own record, and a strip above the slate shows that day's record alongside
the season to date. These come from the walk-forward backtest, so every
prediction shown was made by a model fit only on earlier seasons.

**Upcoming days** show who the model leans toward, by how much, the expected
score, totals, overtime chance and the probable goalies. Shots-on-goal prices
for that game sit in a collapsed panel under each card.

Ticking "Show prices & edge" turns it into a calculator: type a sportsbook
price and it returns edge, expected value per $100 and a fractional-Kelly
stake. Markets carry grades from the held-out test, so a market the model is
measurably bad at is flagged SKIP no matter how large the apparent edge.

Rebuild its data payload with `export_tool.py`, then reassemble:

```bash
./.venv/bin/python export_tool.py   # -> out/tool_data.json
./.venv/bin/python tool/build.py    # -> tool/index.html
```

Three test suites guard the page. `tool/mathtest.js` checks the odds, EV and
Kelly maths against hand-computed values; `tool/daytest.js` checks date
formatting and the plain-English lean wording; `tool/rectest.js` checks every
per-day and season record against the backtest, and that each past row's
result agrees with its own final score. Run all three before publishing:

```bash
JSC=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
$JSC tool/mathtest.js && $JSC tool/daytest.js
# rectest.js needs /tmp/payload.json and /tmp/truth.json; see its header
```

Note for whoever edits the odds parser: sportsbook pages render the minus sign
as U+2212, not a hyphen. A naive `[^0-9+.-]` strip turns "-110" into "+110"
(decimal 2.1 instead of 1.909) and inflates every favourite's edge by about ten
points. `mathtest.js` covers it.

## Two models

**Game winners** — capped by the sport. An oracle knowing each team's true
full-season strength beats this model by only 0.014 log loss, which is about
where published closing lines sit. Knowing what the goalie actually did takes
accuracy from 55% to 87%, so most of what decides a hockey game is not
forecastable by anyone.

**Shots on goal** — five times the signal, because volume comes from ice time
and role rather than variance. Removes 13% of baseline error and beats a
player's own recent average at t=8.3 over 47,000 skater-games.

**Goalie saves** — removes 13.1% of league-average error, and the usual
handicap is actively misleading: a goalie's own recent average is *worse*
than guessing the league mean, because past save counts mostly reflect the
opponents he happened to face.

**Game totals** — close to decoration. The expected total explains 0.33% of
the variance in real totals, and the sportsbook's own posted total is no
better than the league average either. Shown, graded `reference`, never
bettable.

## Frozen, and running forward

The model was frozen on 2026-10-14 as `nhl-v2-20261014`. Ten versions had by
then been compared on the same 1,414 holdout games (MODEL.md section 13) and
none beat the shipped one, so the binding problem is no longer which variant
to pick -- it is that those 1,414 games have been looked at four times.

```bash
./.venv/bin/python freeze.py freeze   # record the config + code hashes
./.venv/bin/python freeze.py log      # append predictions, before the games
./.venv/bin/python freeze.py score    # score the ones that have finished
```

The log is append-only: a `game_id` already present is left alone, so a line
move or a goalie announcement cannot quietly improve a past prediction.
`score` refuses to be quiet about drift -- it compares the current file
hashes to the manifest and says so if the code has changed.

At 80% power it takes 989 fresh games to resolve a 0.010 log-loss difference
against the market, and ~2,500 to separate two versions of this model. Don't
read the first fifty.

## Running it on Render

```bash
./publish.sh          # refresh, rebuild, commit the page, push
```

`app.py` serves the page `daily.sh` builds, and refreshes the part of it
that moves during a day. The model's own prices do not change between
pushes — the ratings behind them only move when games finish — but the
sportsbook's numbers change all day, and those need nothing local. So
`liveserve.py` refetches prices and live scores on a 15-minute timer,
recomputes the no-vig probabilities, the edge and the measured bucket, and
the server swaps that block into the page on its way out. It also promotes
an upcoming game to in-progress once the league says it has started, which
matters because the nightly build runs in the morning, when nothing is under
way — without that, a game beginning tonight would never show as live.

The refresh is lazy, not a background thread: the first request after the
15 minutes are up pays for it and everyone after that is served the cache.
On Render's free plan the instance also sleeps after about 15 idle minutes,
so the first visit after a quiet spell waits for both the wake and the
refresh. The page itself is unchanged and stays a
single self-contained file; it does not know it is being served live.

What that leaves static until the next push: the model's prices, the
finished-game panels (they come from the walk-forward backtest) and the prop
tables (they need a 45 MB frame). `/healthz` says which is which.

Routes:
`/` the page, `/data.json` its payload, `/healthz` a freshness check.
`render.yaml` deploys it on the free plan with `requirements-web.txt` —
Flask, requests and gunicorn, no modelling stack, so the build takes
seconds. The de-vig is reimplemented there in plain Python rather than
importing numpy for arithmetic on twenty numbers; `mathtest.js` and a
cross-check against `odds_eval.devig_logarithmic` keep the two in step
(they agree to 3e-16).

**The model is not rebuilt on Render, deliberately.** The MLB apps in this
account can compute at request time because they pull from a free API with
no local store. This cannot: it needs five seasons of play-by-play (679,052
shot attempts), the shift charts, and a three-minute walk-forward backtest,
and `data/nhl.sqlite3` is 144 MB — past GitHub's file limit and far past
what belongs in a repo that commits daily. So the built page is the deploy
artifact: 550 KB of HTML with the data inlined, rebuilt locally and pushed.
`.gitignore` keeps the databases and pickles out and whitelists the two
files the service actually serves.

First-time setup, once you have a repo:

```bash
git remote add origin git@github.com:<you>/nhl-predictor.git && git push -u origin main
```

Then on Render: **New → Blueprint**, point it at the repo. No environment
variables, no secrets. Every later `./publish.sh` pushes and Render
redeploys within a minute or two.

One difference from the artifact worth knowing: **a Render web service URL
is public.** Anyone with the link can open it. The artifact is private to
your account unless you share it.

## Data note

MoneyPuck sits behind a Cloudflare bot wall, so this uses the NHL's own
`api-web.nhle.com` and computes its own xG from shot coordinates instead.
Closing prices come from ESPN's odds endpoint (`fetch_odds.py`), which
covers all 1,414 holdout games -- the earlier seasons have none, which is
why no correction can be fitted on "the market disagrees".

The NHL publishes no shift chart for 104 of these games, including the first
52 of 2026-27. `features.py` carries each team's last rating forward through
a gap rather than dropping the team-game; dropping it used to leave 210
team-sides with a zero where a rate belongs, which cost 0.031 log loss on
the 48 holdout games affected.
