# The NHL model

Five seasons of NHL play-by-play (2022-23 through 2026-27), pulled from the
league's own public API. 5,694 completed games, 678,682 shot attempts. No
paid feeds and no scraping behind a bot wall.

## 1. Expected goals (`xg.py`)

A logistic model on unblocked shot attempts (Fenwick: shot-on-goal, goal,
missed-shot). Features:

| Group | Features |
|---|---|
| Geometry | distance, log distance, angle, angle², distance×angle, behind-net flag, abs(y) |
| Shot type | 11 one-hots (wrist, snap, slap, tip-in, backhand, deflected, …) |
| Strength | skater differential clipped to ±2, empty net, 3-on-3, 4-on-4 |
| Sequence | rebound (≤3s), fast rebound (≤1s), rush (≤5s and ≥40ft), seconds since last event, off-faceoff |

Shot coordinates are normalised so every attack runs toward `x = +89`. The
attacking end is inferred per period-and-team from the sign of offensive-zone
shots, which is more reliable than `homeTeamDefendingSide` (missing in some
games).

Strength state comes from `situationCode`, a 4-digit field ordered
`away goalie, away skaters, home skaters, home goalie`. Verified against a
known power-play goal.

**Fit on 2022-23 to 2024-25 only**, so the game-level backtest on 2025-26
onward is not helped by an xG model that saw its outcomes.

- Train: AUC 0.7704, log loss 0.2239 (n=366,309)
- Holdout: AUC 0.7665, log loss 0.2286 (n=128,589)
- Public models reach ~0.78 using pre-shot puck movement, which the public
  play-by-play does not carry.

## 2. Team and goalie ratings (`features.py`)

Every rolling value is recorded **before** the game it describes is folded
in. Leakage check: correlation between a team's pre-game xG rating and that
same game's realised xG is 0.28 — the right magnitude. Near 1.0 would mean
the game leaked in.

- **Team**: EWMA with a 20-game half-life over 16 stats (5v5 xG for/against,
  Corsi, shots, goals; power play, penalty kill; all-situations). At each
  season boundary a rating is pulled 30% back to the league mean
  (`CARRYOVER = 0.70`).
- **Goalie**: goals saved above expected per shot faced, decayed with a
  2,500-shot half-life and shrunk by `n/(n+1500)`. Shots are in Fenwick
  units (~46/game), so the shrinkage constant is roughly 33 games. A goalie
  needs a very large sample before the model believes them.
- **Schedule**: rest days, back-to-back, games in the last four days, travel
  distance by haversine between arena coordinates, and time-zone shift.

## 3. Game model (`model.py`)

Not a win/loss classifier. The shape follows the sport:

1. Two Poisson rates for **regulation** goals, from ridge-penalised Poisson
   GLMs on ~50 features.
2. A joint regulation score matrix, with a **Dixon-Coles** correction on the
   four lowest-score cells (`tau` fit by grid search on the training
   likelihood; settles around -0.13 to -0.22).
3. A **tie-inflation** parameter `kappa` on the diagonal, moment-matched to
   the observed regulation-tie rate. Independent Poissons badly
   under-produce level games because hockey has score effects; without this
   the model is overconfident near 50/50. `kappa` lands around 1.36-1.50.
4. A small logistic model for who wins once regulation ends level.
5. **Scoring drift**: both rates are rescaled to the league's trailing
   250-game scoring level, computed only from games already played.

Moneyline, puck line and totals all come off the same matrix, so the prices
are mutually consistent. Final total = regulation total, plus one goal
whenever the game needed OT or a shootout.

Why structural rather than a direct classifier: a direct logistic on the
same features scores 0.2463 Brier against the structural model's 0.2435.

## 4. Results

Walk-forward: each test season is predicted by a model fit only on prior
seasons. 2025-26 and 2026-27 are the clean holdout — nothing in the pipeline
had seen them.

**Clean holdout, 1,414 games:**

| Method | Accuracy | Brier | Log loss |
|---|---|---|---|
| This model | 54.6% | 0.2435 | 0.6797 |
| Elo (goal differential) | 53.2% | 0.2478 | 0.6888 |
| Log5 on points % | 52.8% | 0.2564 | 0.7102 |
| Always pick home | 52.4% | 0.2496 | 0.6923 |
| *Closing line (published)* | *~57.5%* | *~0.236* | *~0.66* |

**By season** — skill tracks how spread out the league is:

| Season | Accuracy | Brier | Skill vs base | Team spread | OT rate |
|---|---|---|---|---|---|
| 2023-24 | 58.6% | 0.2362 | 4.9% | 0.106 | 20.7% |
| 2024-25 | 59.6% | 0.2361 | 4.0% | 0.089 | 20.7% |
| 2025-26 | 54.3% | 0.2446 | 2.0% | 0.079 | 24.9% |

2025-26 was the most compressed season in the data and had the most
overtime. The drop is the league, not a regression in the model.

**Calibration**: seven of eight probability bands contain the diagonal
inside their own 95% interval. A causal temperature recalibration moved
Brier by 0.0001, which confirms the bin-level wobble is sampling noise
rather than systematic overconfidence.

**Derived markets**: puck line well calibrated (28.3% predicted / 30.3%
actual). Totals slightly low. Overtime still under-predicted (21.7% vs
24.5%) — the largest remaining structural miss.

## 5. What the ablation says

Dropping each feature group and refitting:

| Dropped | Δ log loss |
|---|---|
| schedule (rest/b2b/travel) | +0.00157 |
| goalie (gsax) | +0.00093 |
| corsi | +0.00014 |
| actual goals | -0.00014 |
| finishing/goaltending residual | -0.00022 |
| xG | -0.00029 |
| special teams | -0.00094 |

Almost nothing is individually load-bearing. xG, Corsi and actual goals are
collinear proxies for one latent team-strength factor, so removing any one
is absorbed by the others. Only schedule and goalie add unique signal.
Special teams and the finishing residual are net noise on this holdout —
**left in deliberately**, because pruning against the same games the report
quotes would be fitting the holdout.

## 6. Known limitations

- **No confirmed starting goalies.** The model guesses (recent workhorse,
  backup on a back-to-back); a real bettor knows by puck drop. This is the
  single largest gap between the backtest and live use.
- No injuries, line combinations or coaching changes.
- 5v5 rates are per game, not per 60 minutes: time-on-ice by strength state
  is not in the public play-by-play, so power-play time is not normalised.
- The high-skill slices (back-to-backs, unproven goalies) were selected
  after seeing the holdout. Smallest is 185 games. Treat as leads.
- The market comparison is a published figure, not a head-to-head against
  real closing odds. No free archive of historical NHL closing moneylines
  exists.

## 7. Verdict

**Accuracy 6/10.** It beats every baseline built against it and its
probabilities are honest, but it removes only ~2% of baseline error on the
clean holdout.

**Beating the line 2/10.** Log loss ~0.68 against a market near 0.66. When
the market is sharper than you, betting into the close is negative
expectation before the ~4% hold. The two points are for narrow spots —
back-to-backs, unproven goalies — on small samples.

---

## 8. Where the headroom actually is (measured, 2026-10-14)

Before adding anything, I measured what is left to win. The answer changed
the roadmap.

### Opponent-adjusted ratings: a measured null

The rolling ratings are raw, so a soft schedule flatters a team. I built
proper schedule-adjusted ratings (`adjust.py`: ridge on attack + opponent
defence + home, refit weekly, time-weighted) and tested them with
development on 2023-25 and the test set scored once.

| | log loss | gain | t | 95% bootstrap |
|---|---|---|---|---|
| Validation 2023-25 | 0.66343 vs 0.66447 | +0.00104 | +0.77 | [-0.0016, +0.0036] |
| Test 2025-27 | 0.67871 vs 0.67965 | +0.00095 | +0.72 | [-0.0015, +0.0036] |

Not significant, and better on only ~50% of games. The gain replicates in
direction across both sets, so the effect is probably real and simply tiny.
Why: NHL schedules are near-balanced over 82 games, so there is little
schedule bias to remove. Raw and adjusted net xG correlate at 0.954.

### The ceiling on team-strength modelling

`ceiling.py` builds deliberately-cheating oracle features and asks how good
a model could possibly get. Scored on the clean holdout (n=1,414):

| Information available | Accuracy | Brier | Log loss |
|---|---|---|---|
| Always pick home | 52.4% | 0.2496 | 0.6923 |
| What this model knows | 54.8% | 0.2431 | 0.6789 |
| **True full-season team strength** (hindsight) | 58.4% | 0.2368 | **0.6649** |
| Published closing line | ~57.5% | ~0.236 | ~0.660 |
| ... plus the goalie's actual game | 86.6% | 0.0934 | 0.3034 |

Three things follow.

1. **Perfect team-strength knowledge is worth only 0.014 log loss.** The
   model already captures 46% of that — and the rest requires hindsight.
   Better ratings are not where the gap lives.
2. **The closing line sits at roughly perfect-team-strength level, slightly
   past it.** The market's edge over this model is not better team
   modelling; it is the information arriving in the hours before puck drop —
   confirmed goalies, scratches, lineup news.
3. **Goaltending variance dominates everything.** Knowing what the starter
   actually did takes accuracy from 55% to 87%. That term is the reason the
   ceiling is so low for everyone, model and market alike, and it is mostly
   not forecastable.

### Volume is far more predictable than outcomes

`props.py`: a Poisson on pre-game possession features explains **11.4%** of
the variance in team 5v5 shot attempts, against the **2.4%** of outcome
error the win model removes. Shot volume is driven by pace and possession,
which persist; game outcomes are driven by which goalie got hot. Roughly
five times the signal, on data already in the database.

### Revised priorities

**Not worth more work** — team-strength features. Opponent adjustment is a
measured null, the ablation shows every strength proxy is collinear, and
total headroom is 0.014 with omniscience.

**Worth doing, bounded** —
1. Scrape confirmed starting goalies. The backtest uses the real starter;
   live, the model guesses. This closes the single largest backtest-to-live
   gap and is the market's main advantage.
2. Fix the overtime under-prediction (21.7% predicted vs 24.5% actual).
   It biases every near-even game.
3. Shift charts (`api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=`)
   give per-player shifts, which yields true time-on-ice by strength state
   and on-ice player identification. This is the only channel of genuinely
   new free information left, and it also fixes the per-game-rather-than-
   per-60 limitation in §6.

**The real answer for betting** — change the target. Moneyline is capped by
goaltending variance for everyone. Volume props (shots on goal, saves) carry
five times the signal and sit in thinner markets.

---

## 9. Second round: what was built, kept and discarded

### Shift charts (new data)

`api.nhle.com/stats/rest/en/shiftcharts` returns every player's shifts with
start and end times per period. Crawled for all 5,694 games and reduced to
213,022 per-player and 11,178 per-team ice-time records.

Strength state is reconstructed by walking the on-ice skater count for both
teams through the timeline, with goalies excluded via the known-goalie set
(backstopped by anyone over 50 minutes). Two internal checks pass on all
5,589 usable games: both teams always agree on five-on-five time, and each
team's power-play time exactly equals the other's penalty-kill time.

Newly measurable league averages:

| Measure | Minutes per game |
|---|---|
| Five-on-five | 48.17 |
| Power play | 5.53 |
| Penalty kill | 5.53 |
| Four-on-four and other | 0.77 |
| Average skater (regulation) | 16.41 |

### Changes to the win model

| Change | Measured | Verdict |
|---|---|---|
| Per-60 rates from real ice time | -0.00076 log loss on validation | **kept** |
| Retune tie weight on recent games | OT 21.7% -> 25.0% vs 24.5% actual | **kept** |
| Opponent-adjusted team ratings | +0.001 log loss, t=0.77 | dropped |
| Drop special teams + finishing residual | -0.0003 log loss | dropped |

The overtime fix is the substantive one. Independent Poisson rates badly
under-produce level games, and `kappa` was moment-matched to the training
seasons while the league's tie rate drifted. Retuning it every 50 games from
the trailing window closed the gap almost exactly. It was validated in both
directions: on the validation seasons the model was *over*-predicting ties
and the multiplier correctly fell to ~0.88.

Moneyline metrics moved from 0.2435 to 0.2445 Brier — within the ±0.0013
standard error measured in §8, so the win model is unchanged in accuracy
terms. That is the expected result for a model already at its ceiling. What
improved is that the internals are now physically correct and the derived
markets are properly priced.

### Shots on goal (new model)

`props_model.py`. Negative binomial on a skater's shots, with his expected
ice time as an exposure offset:

```
log(E[shots]) = log(expected ice time) + beta . X
```

Features are log rates: shots per 60 overall, at five-on-five and on the
power play; unblocked attempts per 60; power-play share of ice time;
opponent shots allowed per 60 and blocks per 60; home; shifts; five-on-five
ice time. Negative binomial rather than Poisson because shot counts are
overdispersed (fitted size ~16.5); a Poisson prices Over 3.5 too tightly.

**The first specification lost to the trivial baseline.** Fitting the count
directly on linear features with a log link, at ridge alpha 1.0, scored
-0.33% against simply using a player's own trailing average. The error was
twofold: wrong functional form for a log link, and over-regularisation
squashing the spread of predictions. The rate-with-offset form at alpha
0.001 fixed it.

Test set, 47,179 skater-games (2025-26 and 2026-27), scored once:

| Line | Hit rate | vs league avg | vs player's own avg | t |
|---|---|---|---|---|
| Over 1.5 | 42.5% | 13.34% | 0.79% | 8.35 |
| Over 2.5 | 21.5% | 13.25% | 0.63% | 5.62 |
| Over 3.5 | 9.9% | 10.26% | 0.58% | 3.96 |

Count prediction: R² 0.193 against 0.183 for the player's own average.

The headline comparison: **13.3% of baseline error removed, against the win
model's 2.4%.** Just as important, 47,000 skater-games gives the statistical
power that 1,400 games never could — a 0.6% edge is t=5.6 here, where a
0.1% edge on game winners is permanently unmeasurable.

Calibration: 5 of 7 probability bands inside their 95% interval. The two
highest bands run low (said 49.6%, got 52.7%; said 58.9%, got 63.7%), which
is a global-dispersion artefact and matters most on Over 3.5.

### Revised verdict

| | Score | Why |
|---|---|---|
| Picking game winners | **6/10** | Beats every baseline, honestly calibrated, and near a ceiling set by the sport rather than the method |
| Predicting shots on goal | **7/10** | Five times the signal, and a large enough sample to prove a small edge is real |
| Beating a closing line | **3/10** | Moneyline: no, the market is sharper. Props: real edge, but books charge roughly double the vig and the edge over a hand baseline is under 1% |

### Still open

- Confirmed starting goalies. Unchanged and still the largest live gap.
- On-ice player identification is now possible from the shift data but not
  yet used; it would support player-level ratings rather than team-level.
- The shots model's upper probability bands need a non-constant dispersion.
- Saves props are a natural next target: same structure, goalie side, driven
  by opponent shot volume, which §8 showed is the predictable part.

---

## 10. Is there a major improvement left? (measured 2026-10-14)

Short answer: no, and both models are now bounded.

### The shots model's ceiling (`props_ceiling.py`)

Each oracle replaces one input with its true value for that game:

| Information | R² | Over 1.5 error cut |
|---|---|---|
| What the model knows | 0.193 | 13.34% |
| + the ice time he actually got | 0.222 | **15.42%** |
| + the shot rate he actually produced | 0.951 | 71.85% |
| + both (the outcome itself) | 1.000 | 74.85% |

**Perfect ice-time knowledge is worth +2.08 points.** Perfect rate knowledge is
worth +58, but that is essentially the shot count itself — the same role
goaltending variance plays in the win model. Both models therefore have the
same shape: a small forecastable component and a dominant irreducible one.

### Goalie saves: a new market, measured (`saves_model.py`)

The hypothesis was that a high-count market would be more predictable, since
relative noise falls as 1/sqrt(mean): saves average 25.4 against 1.5 for
skater shots. **The hypothesis was wrong.**

| Baseline | Saves | Skater shots |
|---|---|---|
| Error cut vs league mean | 5.79% | 13.34% |
| Error cut vs the player's own average | **6.12%** (t=6.9) | 0.79% (t=8.4) |

Why the low count still wins: skater shots have a large *between-player*
component (a first-line sniper takes four a night, a fourth-liner half of one)
and that is highly predictable from who he is. Goalies do not differ that way.

| Spread in saves per start | |
|---|---|
| Between goalies (sd of per-goalie means) | 2.36 |
| Within a goalie (mean of per-goalie sd) | 6.31 |

So a start's save count is mostly the night's shot volume, not the goalie.

**The useful finding is the baseline, not the model.** A goalie's own recent
average is *worse than simply guessing the league mean*, and gets worse the
longer the window:

| Window | R² |
|---|---|
| League mean | 0.000 |
| Own EWMA, 12-game half-life | -0.017 |
| Own EWMA, 25-game | -0.036 |
| Own EWMA, 50-game | -0.134 |
| Own EWMA, 100-game | -0.369 |

The intuitive handicap — look at what the goalie has been doing — is actively
harmful here, because past shot volume reflects the opponents he happened to
face. That is why the model's edge over the hand baseline is eight times larger
on saves than on skater shots.

**Not yet usable.** Calibration fails in the upper bands (said 48%, got 63%;
3 of 7 bands outside their 95% interval). The Poisson GLM plus a single global
dispersion compresses the spread, the same fault as the first shots
specification. It needs a non-constant dispersion before it can be priced.

### Ranked, honestly

1. **Confirmed starting goalies.** Operational, not statistical, and still the
   largest gap between backtest and live use.
2. **Saves props**, after fixing the dispersion. Largest measured edge over how
   a person would actually handicap the market.
3. **Better ice-time forecasting** for skater shots. Bounded at +2.08 points,
   realistically a third of that. Tractable from teammate absences, score
   script, and recent trend against the long average.

### Deliberately not doing

**On-ice player ratings from the shift data.** The shift crawl kept only
per-game totals, not the intervals, so this needs a second full crawl. The
payoff is a better *estimator* of team strength, which §8 bounded at 0.014 log
loss in total. The exception would be lineup-change situations, where
team-level ratings go stale — but that is a narrow slice, and the cost is high.

### The structural reason

Neither model is short of cleverness; both are short of information that does
not exist before puck drop. The market's own log loss of about 0.660 is not
evidence that books are sharp so much as that the sport caps everyone at
roughly the same place. A major improvement would have to come from better
*information* — confirmed goalies, confirmed lineups, injury news, in-game
state — rather than from better statistics.

---

## 11. The four-item work list, measured (2026-10-14)

All four were attempted. Two produced clean negative results that overturn
earlier recommendations in this document, including my own priority ranking.

### 1. Confirmed starting goalies — worth nothing, measurably

New data: the boxscore marks one goalie per team `starter: true`. Crawled for
all 5,694 games (`fetch_lineups.py`, 22,033 rows) and validated — exactly one
starter in all 10,330 team-games, none with zero ice time, median 60 minutes.

| Question | Answer |
|---|---|
| Did the backtest use the right goalie? | **97.47%** of team-games. The shots-faced proxy was sound. |
| Would the live heuristic name the right starter? | **64.02%** |
| — back-to-back backup rule | 74.82% (n=969) |
| — recent workhorse rule | 62.93% (n=9,253) |

Then the experiment that matters (`goalie_cost.py`): rebuild the features with
the *guessed* goalie's rating instead of the actual starter's, and re-run the
identical backtest.

| | accuracy | Brier | log loss |
|---|---|---|---|
| Confirmed starter | 0.5460 | 0.24452 | 0.68187 |
| Guessed starter | 0.5375 | 0.24451 | 0.68183 |

**Knowing the starter is worth +0.00004 log loss, paired t = −0.06.** Nothing.

So the feature was tested harder (`goalie_strength.py`): a starter/backup
indicator (+0.00118), rolling start share (+0.00079), lighter shrinkage at
K=300 (−0.00002), heavier at K=3000 (+0.00005), all together (+0.00111). Every
variant neutral or worse. The heavy shrinkage was already correct.

This reconciles with §8's oracle. Knowing what the goalie *did* takes accuracy
from 55% to 87%; knowing *which goalie it is* predicts that performance
almost not at all. Goaltending decides hockey games and remains
unforecastable, and those two facts are consistent.

One conditional result worth keeping: games where a live board could name both
starters are themselves more predictable (log loss 0.675 against 0.686), since
settled rotations go with settled situations.

**Conclusion: a confirmed-starter feed is not worth building for this model.**
That contradicts the priority I gave it in §8 and §10.

### 2. A genuine closing-line comparison — found, and it corrects §8

ESPN's public API records a sportsbook line per game (`fetch_odds.py`,
`odds_eval.py`). Matched **1,414 of 1,414** holdout games; DraftKings on 1,038,
ESPN BET on 376. This is the last line ESPN held, at or near the close, so it
is called the *recorded* line rather than the closing line.

| Source | accuracy | Brier | log loss |
|---|---|---|---|
| Model | 0.5460 | 0.24452 | 0.68187 |
| Market, no-vig (proportional) | 0.5559 | 0.24321 | **0.67919** |
| Market, no-vig (logarithmic) | 0.5566 | 0.24341 | 0.67961 |
| Always home | 0.5240 | 0.24942 | 0.69199 |

**A correction.** §8 compared against a published figure of roughly 0.660 and
concluded the market was clearly sharper. Measured on these exact games the
market is at **0.679**, and the model's 0.682 is **not significantly worse**
(paired t = +0.90). The earlier gap was overstated: 2025-26 was a high-parity
season and hard for everyone, so a literature number from ordinary seasons was
the wrong benchmark.

The conclusion survives anyway, on better evidence:

- **Calibration**: market 8 of 8 bands inside their 95% interval, model 6 of 8.
- **Profitability** at the actual posted prices, median hold 4.26%:

| Threshold | Bets | ROI | 95% interval | Beats zero |
|---|---|---|---|---|
| 0% | 983 | +0.53% | [−6.1, +7.4] | no |
| 1% | 783 | +1.20% | [−6.3, +8.7] | no |
| 2% | 635 | +1.37% | [−7.4, +10.0] | no |
| 3% | 497 | +2.21% | [−7.4, +11.7] | no |
| 5% | 295 | −1.00% | [−13.4, +11.5] | no |
| 8% | 96 | −2.47% | [−24.5, +20.5] | no |

Two things kill it. ROI never clears zero at any threshold. And **ROI falls as
the threshold rises** — a model with real edge earns more per bet as it gets
more selective, because it is picking its stronger opinions. This earns less,
which means the large apparent edges are the model's own errors.

- **The decisive test**: on the 256 games where model and market pick opposite
  sides, the model is right **47.3%** of the time (121 of 256, z = −0.88).
  A model holding information the market lacks wins those well above half.

**Conclusion: no exploitable edge. Now properly demonstrated rather than
asserted from a borrowed benchmark.**

### 3. Lineup absences — real signal, not extractable

New data: 36,217 scratches (`fetch_lineups.py`), turned into the share of each
team's usual ice time that is missing (`lineups.py`), weighted by the absent
players' normal roles.

The raw relationship looks strong. Home win rate by quintile of the absence
gap: 50.0%, 52.0%, 51.6%, 55.5%, **59.7%**.

But nothing improved the model: scratch count (+0.00042), ice-time share
missing (+0.00001), biggest single absence (+0.00152), all together
(+0.00173).

A residual test says why only partly. Regressed against the model's errors
there *is* signal, concentrated entirely in the extreme quintile (predicted
55.9%, actual 60.8%, significant), with corr(gap, residual) = +0.031, t = 1.99
over 4,038 games. So the relationship is non-linear and the linear term
diluted it. Thresholded and hinged encodings were then tried at the 80th, 85th
and 90th percentiles — every one scored worse on validation.

**Conclusion: the signal is at the edge of detectability, and capturing it
costs more in variance than it gains. Team ratings absorb most of it already,
since a club that has been missing players has a rating that reflects that.**

### 4. Shots on goal — dispersion partly fixed, ice time a dead end

**Dispersion.** The fault was upper bands running low (said 59%, got 65%).
Four fixes, in order:

| Attempt | Result |
|---|---|
| Mean correction on log(predicted) | no effect |
| Negative-binomial size as a function of the mean | no effect |
| Direct logistic model per line, dropping the count distribution | no effect (2/7 bands) |
| Temperature fit on a held-out season | bottom of the range fixed, top still low |

The diagnosis the first two attempts missed: a low-minutes skater is a
*mixture* — sat most of the night, or played a normal shift — which inflates
variance while concentrating mass at zero. A negative binomial has to put that
variance in the right tail instead. That is a distributional fault, not a
parameter one, which is why size and mean corrections did nothing.

Temperature scaling found values below one in every fold (0.85, 0.95, 0.975,
shrinking as training data grows), confirming the probabilities are genuinely
compressed. It calibrates the bottom of the range, where 98% of the mass sits,
and leaves the top three bands a few points low.

**Conclusion: kept, with the residual tail bias stated as a known limit. Over
3.5 is the line it affects.**

**Ice time.** §10 bounded perfect ice-time knowledge at +2.08 points. Built a
ridge model on rolling ice time, shifts, team absences, role share and home
(`toi_model.py`):

| | R² | MAE |
|---|---|---|
| His own trailing average | 0.6902 | 1.89 min |
| Ice-time model | 0.6917 | 1.89 min |

0.50% of the remaining error. Plugged in as the shots model's exposure it is
worth **+0.043 points** against the +2.08 ceiling — 2% of the headroom.

**Conclusion: not worth wiring in. Ice time is already near-deterministic from
a player's recent average, which is why the trailing average was hard to beat.**

### What this round established

Three of the four were negative, and the two most confident recommendations in
this document — confirmed goalies and better ice time — are now measured as
worth 0.00004 log loss and 0.043 points respectively. The one that improved
things is the odds work, and what it improved is the honesty of the comparison
rather than the model.

The pattern across §8, §10 and §11 is consistent: every input that can be
known before puck drop is close to exhausted, and what decides hockey games is
not knowable in advance by anyone, the market included.

---

## 12. Current verdict (supersedes §7, §9 and §10)

The model is essentially unchanged by §11. What changed is the evidence.

| | Score | Moved? | Why |
|---|---|---|---|
| Picking game winners | **6/10** | no | Statistically level with real sportsbook prices on log loss (0.682 vs 0.679, t=0.90), which is better than §8 claimed. But the market still wins on accuracy, Brier and calibration — four metrics all leaning one way. |
| Predicting shots on goal | **7/10** | no | Calibration now correct across 98% of the probability mass after the temperature fix. The top bands still run low, and the fix slightly lowered Brier while improving the edge over a hand baseline. Mixed, so unchanged. |
| Beating a closing line | **3/10** | no | The gap to the market is smaller than §8 said, which pulls up. Three independent tests now show no exploitable edge, which pulls down. They cancel. |

Two of these three ratings previously rested on a published benchmark of
roughly 0.660 that turned out to be the wrong yardstick for this holdout.
They now rest on 1,414 games matched to actual posted prices. Same numbers,
far better foundations.

The honest summary of §11: a great deal of measurement, almost no
improvement. Its value was negative knowledge — a confirmed-starter feed and
a lineup pipeline would both have been built for nothing.

## 13. The four-step validation plan, executed (2026-10-14)

Everything in sections 8-12 compared one model to a benchmark. This section
does something different: it compares the model to *its own earlier
versions*, on identical games, and asks whether the changes made along the
way were real. The answer is mostly no, and one of them was a bug.

Scripts: `variants.py` builds every feature frame in one pass so the game
list is identical; `compare.py` scores them walk-forward; `flips.py` asks
what each change actually decides; `errors.py` categorises the mistakes;
`widen.py` tests the one idea that came out of them; `freeze.py` ends the
retrospective phase.

### Step 1: every version, same 1,414 games

Each variant is refit season by season on prior seasons only. Significance
is a paired bootstrap over games -- the same resampled game list scores both
models, which removes the game-to-game difficulty that dominates the
variance of any single model's score.

| variant | acc | Brier | log loss | Δ log loss vs shipped | 95% CI |
|---|---|---|---|---|---|
| v1 counting stats, kappa fixed | .5460 | .24346 | .67965 | **-0.00222** | [-0.0052, +0.0007] |
| v1 counting stats + tie retune | .5417 | .24346 | .67964 | -0.00224 | [-0.0052, +0.0007] |
| v2 per-60, kappa fixed | .5453 | .24456 | .68194 | +0.00006 | [-0.0004, +0.0006] |
| **v2 per-60 + tie retune (shipped)** | .5460 | .24452 | .68187 | — | — |
| v2 per-60, imputed ice time | .5509 | .24485 | .68254 | +0.00066 | [-0.0013, +0.0027] |
| v2 per-60, no zero-filled ratings | .5474 | .24485 | .68254 | +0.00066 | [-0.0012, +0.0025] |
| v2 + causal recalibration | .5460 | .24434 | .68159 | -0.00028 | [-0.0035, +0.0029] |
| v2 + confirmed-goalie encodings | .5446 | .24464 | .68212 | +0.00025 | [-0.0009, +0.0014] |
| v2 + lineup absences | .5467 | .24412 | .68101 | -0.00086 | [-0.0024, +0.0007] |
| v2 + opponent-adjusted ratings | .5495 | .24447 | .68171 | -0.00016 | [-0.0018, +0.0015] |

Not one confidence interval excludes zero. The most interesting row is the
first: **the original counting-stat ratings beat the shipped per-60 ratings**
by 0.0022 log loss, P(better) = 0.933. Section 9 had recorded the per-60
change as "kept" on the strength of -0.00076 on the validation seasons. On
the clean holdout it is worse.

Log loss by season, which is what stopped that becoming a revert:

| variant | 2023-24* | 2024-25* | 2025-26 | 2026-27 |
|---|---|---|---|---|
| v1 counting stats | .66452 | .66442 | .68201 | .64935 |
| v2 per-60 (shipped) | .66093 | .66569 | .68300 | .66738 |
| v2 per-60, no zero-filled ratings | .66093 | .66455 | .68475 | .65409 |
| v2 per-60, imputed ice time | .66093 | .66448 | .68520 | .64825 |

(* the xG model saw those outcomes; only the pattern across seasons is being
read.) The v1 advantage is +0.0036 in one season and -0.0180 in another. It
is one 102-game season, which is exactly the trap this plan warns about.

### What was actually wrong

The NHL publishes no shift chart for 104 of the 5,694 games held here --
the end of 2024-25 and the **first 52 games of 2026-27**. I checked the
endpoint directly: the data does not exist, it was not a failed crawl.

`to_per_60` inner-joined the shift table, so those team-games vanished.
`assemble` then left-joined ratings onto all games, leaving 210 team-sides
with no rating, and `build_matrix` filled the hole with `fillna(0.0)` -- a
literal zero where a rate of 2.5 xG per 60 belongs, which after
standardisation is a more extreme input than any team has ever been.

Splitting the holdout on whether the shift chart exists:

| | n | v1 log loss | shipped log loss | difference |
|---|---|---|---|---|
| shift chart published | 1,366 | .68064 | .68186 | -0.00122 |
| no shift chart | 48 | .65161 | .68233 | **-0.03072** |

The entire v1 advantage is 48 games fed garbage. `roll_team` now carries the
team's last rating forward instead (`carry=True`), so a missing shift chart
costs staleness rather than an absurd input. 210 zero-filled sides, none
left.

**This fix is not justified by the pooled metric**, which moves +0.00066 the
wrong way with a CI straddling zero. It is justified because the input was
indefensible, and because 2026-27 -- the season being predicted right now,
half of which has no shift charts -- improves by 0.013 log loss. Ice-time
imputation is implemented too (`to_per_60(impute=True)`) and scores about
the same; it is off, because carrying a rating forward invents nothing and
imputing a denominator does.

### Step 3: what each change actually decides

On a flipped game the two models picked opposite sides, so exactly one is
right. That makes the flip set an exact binomial trial and needs no
bootstrap.

| change | mean abs Δp | flips | right on flips | p |
|---|---|---|---|---|
| v1 counting stats | .0210 | 118 (8.3%) | 59/118 = .500 | 1.00 |
| per-60, no zero-fill | .0109 | 72 (5.1%) | 37/72 = .514 | .91 |
| per-60, imputed ice time | .0121 | 79 (5.6%) | 43/79 = .544 | .50 |
| confirmed-goalie encodings | .0089 | 48 (3.4%) | 23/48 = .479 | .89 |
| lineup absences | .0104 | 65 (4.6%) | 33/65 = .508 | 1.00 |
| opponent-adjusted ratings | .0124 | 79 (5.6%) | 42/79 = .532 | .65 |
| causal recalibration | .0213 | 0 | — | — |

Every change that moves picks moves them at a coin-flip rate. The best any
variant manages on its own flipped games is 54.4%, with a CI from 42.8% to
65.7%. Recalibration is the clearest case: it moves the probability further than
any other change here -- 43% of games by more than two points -- and never
changes a single pick. That is exactly what temperature scaling is for, and
the reason accuracy is the wrong metric to judge it by.

### Step 2: the mistakes, by situation

Realised accuracy minus the confidence the model claimed, pooled over four
test seasons, with how many seasons share the sign:

| situation | n | acc | claimed | gap | t | seasons |
|---|---|---|---|---|---|---|
| model backs the market's underdog | 229 | .459 | .533 | **-0.074** | -1.79 | 2/2 |
| game went to OT/SO | 890 | .530 | .593 | -0.063 | -3.12 | 3/3 |
| an unusual goalie started | 1,669 | .560 | .597 | **-0.037** | -2.16 | 3/4 |
| moderate favourite (.55-.65) | 1,883 | .571 | .594 | -0.022 | -0.64 | 3/4 |
| long away trip (>2000 km) | 504 | .577 | .596 | -0.019 | -0.10 | 3/3 |
| coin flip (conf < .55) | 1,330 | .512 | .525 | -0.013 | +0.38 | 3/4 |
| strong favourite (conf ≥ .65) | 825 | .696 | .707 | -0.012 | +0.36 | 2/3 |
| either side back-to-back | 1,028 | .608 | .606 | +0.002 | +1.45 | 3/4 |

The OT row is the most significant and the least meaningful: it conditions
on the outcome. Games that reach OT are close games, and OT is close to a
coin flip, so every forecaster looks overconfident there -- the market's log
loss on those same games is 0.7151 against the model's 0.7138. Discarded as
a finding.

Two rows survive. Against the market on the 1,414 priced games:

| situation | n | model acc | market acc | model - market log loss |
|---|---|---|---|---|
| strong favourite (conf ≥ .65) | 183 | .694 | .694 | **-0.0091** |
| either side back-to-back | 384 | .570 | .568 | -0.0047 |
| an unusual goalie started | 579 | .542 | .567 | +0.0043 |
| model backs the market's underdog | 229 | .459 | .542 | **+0.0115** |

That is the whole model in two lines. Where it is confident and the market
agrees, it beats the market. Where it disagrees with the market on which
team should be favoured, it is wrong 54% of the time and loses 0.0115 log
loss. It amplifies consensus well and contradicts it badly, which is a
precise restatement of section 11's finding that there is no edge.

It is also not fixable from here: odds exist only for the holdout seasons,
so any correction fitted on "the market disagrees" would be fitted on the
same games it was tested on.

### Step 2 follow-up: the goalie finding is a variance fault

The unusual-starter row says something new. Section 11 swapped the confirmed
starter's rating in and changed log loss by 0.00004, because knowing *which*
goalie plays says little about *how* he plays. But not knowing should widen
the distribution rather than move it -- and the model does not widen.

`widen.py` fits three corrections on 2023-25 and applies them unchanged to
the holdout. The group temperature confirms the diagnosis exactly: T = 0.902
with a settled starter, **1.366 when the starter is unusual**. On those 579
holdout games the calibration gap halves, from -0.0425 to -0.0206.

Pooled, it is worth -0.00031 log loss, CI [-0.0024, +0.0017].

So the fault is real, correctly diagnosed and correctly fixed, and worth
nothing. It is not shipped: it would be a fourth parameter fitted on the
holdout era for a gain a quarter the size of its own confidence interval.

### Step 4: frozen

The selection pressure is now the binding problem. Nothing fits on the
holdout, but the *choices* -- which features to keep, which encodings to
try, which of ten variants to call best -- have been made in the light of
how they scored on the same 1,414 games, four rounds running. No leakage
check detects that.

`freeze.py` ends it. The configuration and a SHA-256 fingerprint of every
file that produces a prediction are recorded in `data/frozen.json`;
`freeze.py log` appends predictions to `data/frozen_log.csv` before the
games are played and refuses to touch a `game_id` already there, so a line
move or a goalie announcement cannot quietly improve a past prediction;
`freeze.py score` scores what has finished and warns if the code hashes have
drifted.

Frozen 2026-10-14 as `nhl-v2-20261014`, trained through 2026-10-13, 48
predictions logged for 2026-10-14 to 2026-10-21.

How long before the log can say anything? The per-game log-loss difference
against the market has a standard deviation of 0.112, so at 80% power:

| difference to detect | games needed |
|---|---|
| 0.020 log loss | 248 |
| 0.010 | 989 |
| 0.005 | 3,956 |

And to separate two versions of this model from each other, where the
per-game differences are much smaller but so is the effect: 2,479 games to
call 0.002 between the shipped model and the zero-fill fix. A season is
1,312 games. That is the real answer to "can this be improved" -- not that
the ideas are bad, but that distinguishing a good one from a bad one takes
more hockey than exists.

### Ratings after this round

Unchanged: 6/10 winners, 7/10 shots on goal, 3/10 beating a line. One bug
fixed, which the pooled metric does not reward and the current season does.
Nine candidate changes measured against the shipped model; not one
confidence interval excludes zero; every one of them flips picks at
chance. The model is where it was, and now it is frozen, logged, and will be
judged on games it has never seen.

## 14. Pricing the buckets, and a correction (2026-10-15)

Section 13 measured log loss by situation. Log loss is not money, so
`edge_test.py` re-ran the same splits by backing the model's side at the
actual posted American price and paying the actual vig. Buckets are defined
on information available before the game: whether the market favours the
same side, and how confident the model is. The pick'em boundary is 0.495 --
a coin-flip line is not the market naming an underdog.

| situation | n | win% | ROI per $1 | 95% CI | value picks within it |
|---|---|---|---|---|---|
| model backs the market's dog | 212 | .476 | +0.4% | ±14.2 | 202, +1.5% |
| conf < 60%, market agrees | 758 | .509 | **−12.8%** | ±6.1 | 171, −13.7% |
| conf ≥ 60%, market agrees | 444 | .642 | −1.8% | ±6.8 | 204, +2.7% |
| every model pick | 1,414 | .563 | −7.3% | ±4.5 | — |
| **this page's value filter** | 577 | .532 | **−2.6%** | −10.2 to +5.0 | — |

**A correction.** Earlier in this work I reported the underdog bucket as
"measured to lose, −3.7%, 45.9%". That figure came from a boundary that
counted pick'em lines as the market naming an underdog. Corrected, the
bucket is 212 games at 47.6% and **+0.4% per dollar** -- it loses the pick
more often than not, but underdog prices pay more, so the return is a wash
with an interval of ±14 points. It is not established as a loser. The claim
that the model's three largest apparent edges sat in a losing bucket was
wrong; they sit in an unmeasurable one.

What survives, and is now the only established result in this file about
money: **model confidence under 60% with the book agreeing on the side
returned −12.8%, with the entire confidence interval below zero.** That is
54% of the board, and the one thing the page can say with authority.

The value filter itself -- any price above the model's own number, which is
what the MLB page's finder uses -- returned −2.6% over 577 picks, interval
−10.2% to +5.0%. After roughly 4.5% vig, that is what no edge looks like:
indistinguishable from a wash, in both directions.

### Why "unproven" was a bad label

The page briefly labelled the strong-favourite bucket UNPROVEN, which told a
reader nothing: unproven how, and in which direction? The chips now carry
the measurement instead -- `−12.8% ±6`, `+0.4% ±14`, `−1.8% ±7` -- so the
estimate and its uncertainty are both visible, and a reader can see that no
bucket's interval clears zero. `tool/linetest.py` asserts that: no chip may
ever read as an established winner, and exactly one reads as an established
loser.

## 15. Are the total estimates accurate? (measured 2026-10-16)

`totals_eval.py`. The page prints an expected total on every card and the
backtest reported calibration at three fixed lines, which is the wrong
question: a forecast that always says "6.2 goals" can be perfectly
calibrated and carry no information whatever.

Three harder questions, on the clean holdout.

### 1. Does it beat guessing the league average?

| | MAE | RMSE | R² vs the mean |
|---|---|---|---|
| model | 1.837 | 2.300 | **−0.0008** |
| always the league average | 1.849 | 2.299 | 0 |

Marginally better on absolute error (t=−1.78, p=0.076), marginally worse on
squared error. No skill either way.

The reason is visible in one line: **the model's estimates have a standard
deviation of 0.272 against the actual 2.299**, and span 5.56 to 7.62 while
real games run from 1 to 16. Correlation with the realised total is 0.057,
so the expected total explains **0.33% of the variance**.

### 2. Does it beat the number the sportsbook posted?

| | MAE | RMSE | R² vs the mean |
|---|---|---|---|
| model | 1.839 | 2.303 | −0.0008 |
| posted total | 1.842 | 2.316 | −0.0123 |

Dead level (t=−0.20, p=0.84), and the two estimates correlate only 0.353.

The striking part is the second row. **The sportsbook's own total is also no
better than the league average** at predicting a game's total, and by R² it
is slightly worse. This is not a fault in the model. Nobody can forecast how
many goals a hockey game will contain, because the quantity is nearly all
variance. The MLB model in this account reached the same conclusion from the
other side: its total MAE was 3.607 against the market's 3.527.

### 3. When it disagrees with the line, is it right?

**51.24% on 1,411 non-push games**, 95% CI [48.6%, 53.8%]. A −110 price
needs 52.4%. Split by how far apart the two numbers are, no bucket separates
from chance; the largest disagreements (over half a goal, calling the over)
go 52.5% on 398 games. `edge_test.py` priced it directly and found −4.6% per
dollar at any edge, −7.3% at a 4% threshold with the interval below zero.

### The calibration slope

Regressing the outcome on the model's own log-odds at the book's line:

| market | slope |
|---|---|
| game totals | **+0.283** |
| moneyline, for comparison | +0.785 |

A slope of 1 means the probabilities carry their stated information; 0 means
none. At 0.28, a 56% from this model is really about 52%. Calibration *on
average* is fine — 53.7% predicted against 51.9% actual at a 5.5 line, 46.8%
against 50.2% at 6.5 — because the average is near the base rate and the
model never strays far from it. The spread is what fails.

### What changed on the page

Nothing in the model. Refitting a slope on these same 1,414 games is exactly
the selection pressure §13 froze the model to stop. What changed is what the
page claims: **Over 5.5 and Over 6.5 move from `usable` to `reference`**,
which forces SKIP in the UI regardless of how large an apparent edge is, and
the limitations panel now states the 0.33% and the 51.2% outright. Over 4.5
stays `biased` — at 77.1% predicted against 80.9% actual it is the one
totals number that is also wrong on average.

## 16. Can the totals be improved? (measured 2026-10-16)

§15 showed the total estimates carry almost no information. This asks
whether that is fixable. Four tests, ceiling first.

### The ceiling (`totals_improve.py`)

An oracle that knows each team's realised full-season scoring and conceding
rates — it uses the future, so no causal model can beat it:

| | MAE | R² vs the league average | sd of estimate |
|---|---|---|---|
| oracle, using the future | 1.8121 | **+0.0465** | 0.499 |
| the model | 1.8596 | −0.0060 | 0.330 |
| always the league average | 1.8654 | 0 | 0 |

**Perfect season-long knowledge of both teams is worth 0.053 goals of mean
absolute error.** Even the oracle's estimates only span half a goal against
the 2.3-goal spread of real totals. That is the entire budget.

A note against my own first test: the Poisson decomposition says the
variance left for between-game differences is **−0.968**, totals being
*under*-dispersed relative to a constant-rate count. That is real — the
overtime rule adds exactly one goal to a tied game, never two or none, which
truncates the distribution — but it means the process is not Poisson, so the
arithmetic cannot be read as "zero signal". The oracle is the honest bound.

### Three ways at it, all null

**Shrink the estimates.** The slope of actual on predicted is 0.311 on the
validation seasons (0.484 on the holdout), well under 1, so the model
overstates its own differences. Shrinking by the fitted slope — fitted on
validation, applied to the holdout — improves R² (+0.0011 to +0.0040) and
worsens MAE (1.8367 to 1.8400). Paired t=+0.67, p=0.504.

**Bypass the score grid** (`totals_direct.py`). The grid exists to keep the
moneyline, puck line and totals mutually consistent, and that constraint
could be costing the total its spread. A ridge trained directly on the total
— the most favourable setup available, optimising exactly the target —
scores MAE 1.8371 against the grid's 1.8367. **t=+0.08, p=0.934.** The grid
is not the bottleneck; the information is not in the features. A naive
rolling goals-for/against estimate is significantly *worse* (t=+2.81,
p=0.005).

**Referees** (`fetch_refs.py`, `refs_eval.py`). The one pre-game signal
never tested, and a plausible one: crews differ in penalties called,
penalties make power plays, power plays are about a fifth of scoring. The
names turned out to be in the right-rail endpoint already crawled for
scratches, so this cost one re-read: 21,417 official assignments, 51
referees, 100% coverage.

| | |
|---|---|
| spread of referees' average totals (39 with 100+ games) | 0.1396 |
| spread chance alone produces | 0.1477 |
| **excess attributable to the referee** | **0.0000** |
| one-way ANOVA across crews | F=0.916, p=0.617 |

Referees differ *less* than sampling noise would produce. The apparent range
— Trevor Hanson 5.93, Carter Sandlak 6.46 — is entirely the noise of 200 to
300 games each. Adding the crew's causal history to the model: MAE 1.8360 to
1.8362, t=+0.54, p=0.587; the crew's own history correlates +0.011 with the
total.

### The answer

No. The ceiling is 4.6% of variance on a generous oracle, the model reaches
none of it, and the three routes to it — better calibration, a better
functional form, a new signal — measure as nothing. The market's own total
is also worse than the league average by R², which is the clearest evidence
that this is a property of hockey rather than a deficiency here.

The referee table is kept. It is the right input for a penalty or
power-play market, which this project does not price, and it cost one crawl
of an endpoint already in use.

## 17. The one improvement that worked: saves props (2026-10-16)

§10 shelved the goalie-saves model as "not yet usable" on a calibration
failure, and blamed the dispersion. Re-diagnosed, the blame was wrong and
the fix is one the win model already uses.

### Re-diagnosis

The shipped version's calibration across all three lines:

| | |
|---|---|
| calibration slope | **+1.122** |
| intercept | **+0.170** |
| bands inside their 95% interval | 5 of 8 |

A slope *above* 1 means the probabilities are too timid, not too confident,
and the positive intercept means the over is systematically under-priced.
Both point at the level of `mu`, not the shape of the distribution around
it. §10's note that "the Poisson GLM plus a single global dispersion
compresses the spread" was the wrong reading of its own numbers.

### Dispersion was not the fault

Three dispersion treatments, each walk-forward, none fitted on the holdout:

| | slope | bands inside | Brier |
|---|---|---|---|
| fitted in-sample (shipped) | +1.108 | 5/8 | 0.18544 |
| fitted on last season's out-of-sample residuals | +1.108 | 5/8 | 0.18544 |
| allowed to depend on the mean, r = a·μ^b | +1.118 | 5/8 | 0.18546 |

Identical to three decimal places. The negative binomial was already fitting
the spread correctly; the centre was in the wrong place.

### What worked

A causal level correction, the same device the win model uses for league
scoring drift: compare actual to predicted saves over the trailing 400
starts and scale `mu` by the ratio, clipped to ±5%, using only starts
already played. The mean multiplier on the holdout came out **1.0080** —
the league was producing 0.8% more saves than a model fitted on older
seasons expected.

| | slope | intercept | bands inside | Brier |
|---|---|---|---|---|
| shipped | +1.108 | +0.165 | 5/8 | 0.18544 |
| Platt, fitted on validation | +1.003 | +0.092 | 6/8 | 0.18529 |
| **causal level correction** | +1.096 | +0.095 | **7/8** | 0.18538 |

The two bands that failed before now sit inside their intervals: 0.40-0.50
went from a +0.049 gap to +0.026, and 0.60-0.70 from +0.095 to +0.065. The
only band still outside holds 11 prices.

Platt scores a nominally perfect slope but leaves two middle bands outside,
and it is a correction fitted on past seasons rather than one that tracks
the league as it moves. The causal version is preferred for the same reason
the win model's tie retune was: it was validated in both directions and it
keeps working when the league changes.

### What it is worth

| | MAE | R² vs the league mean |
|---|---|---|
| model | 5.024 | **+0.131** |
| model, level corrected | 5.034 | +0.130 |
| the goalie's own recent average | 5.317 | +0.036 |
| league mean | 5.476 | 0 |

Discrimination is unchanged — this is a calibration fix, not new
information, exactly like the temperature correction on skater shots. What
changed is that the prices can now be shown. The model removes **13.1%** of
the error a league-average guess makes, against 0.33% for game totals and
2.4% for game winners, and it beats the hand baseline by a wide margin for
the reason §10 found: a goalie's past save count reflects the opponents he
happened to face, so the intuitive handicap is actively misleading.

Saves is now the second-best market in this project, behind skater shots and
far ahead of anything game-level. It is not yet on the page: that needs a
live path for the probable starter's expected shots faced, which does not
exist yet.

## 18. The puck line and the overtime price (measured 2026-10-16)

The same test §15 applied to totals, applied to the two remaining derived
markets. Calibration on average cannot tell a forecast from a constant, so
each market gets a Brier skill score against always quoting the base rate
and a calibration slope.

| market | predicted | actual | Brier skill | slope | bands inside |
|---|---|---|---|---|---|
| **home −1.5** | .2726 | .3023 | **+0.0279** | +1.111 | 7/7 |
| moneyline | .5431 | .5229 | +0.0181 | +0.785 | 8/8 |
| over 5.5 | .5555 | .5708 | +0.0005 | +0.588 | 5/5 |
| **goes to overtime** | .2497 | .2452 | **−0.0019** | +0.338 | 5/6 |

The page had these two graded exactly backwards.

### The puck line is the sharpest market on the card

It removes 2.79% of the base rate's error, more than the moneyline's 1.81%,
every probability band lands inside its interval, and the slope above 1 says
it is if anything too timid. Its one fault was level: it said 27.3% and the
home side won by two or more 30.2% of the time, the same sign in both
holdout seasons.

That is the §17 fault again — ordering right, centre off — so it takes the
§17 fix: a trailing-window shift, applied in log-odds because adding a flat
three points to a 10% price and a 50% price are not the same adjustment, and
fitted only on games already played.

| | predicted | actual | Brier | skill | slope |
|---|---|---|---|---|---|
| as shipped | .2726 | .3023 | .20504 | +0.0279 | +1.111 |
| with the offset | .2920 | .3023 | .20470 | +0.0264 | +0.947 |

The level gap closes from −0.0297 to −0.0103 and the slope moves to 0.947.
Brier skill gives up 0.0015, which is the honest trade: a market quoted
three points low is one where the wrong side always looks like the value,
and that matters more on a page built to compare a model price to a book
price. **Grade raised from `biased` to `usable`.**

### Overtime is decoration

Right on average — the §9 tie retune did its job, 25.0% against 24.5% — and
close to uninformative. A Brier skill of −0.0019 means it is *slightly worse
than quoting the league's overtime rate*, and a slope of 0.338 means the
number moves without meaning it. Its spread is sd 0.029 on a quantity whose
base rate is 24.5%.

This is the same diagnosis as game totals and for the same reason: whether a
game is level after sixty minutes is nearly all variance. **Grade lowered
from `usable` to `reference`**, so the UI forces SKIP.

### The freeze held, and said so

Changing `backtest.py` and `live.py` tripped the code-hash check in
`freeze.py`, which is what it is for. The moneyline is untouched: `p_home`,
`exp_total`, `p_ot` and `p_over_55` match the previous build to six decimals
across all 4,038 backtested games, and only `p_home_m15` moved (mean 0.039).
Re-frozen as **`nhl-v2-20261015`** carrying `moneyline_unchanged_since:
nhl-v2-20261014`, so the 60 predictions already logged remain a continuous
prospective record for the market they were logged to test.

## 19. Auditing every number on the page (2026-10-16)

Every quantity the page still displays, scored the same way: Brier skill
against always quoting the base rate, and the calibration slope. Measured on
the clean holdout.

| market | n | skill | slope | bands | verdict |
|---|---|---|---|---|---|
| shots over 1.5 | 47,345 | **+0.1334** | +1.058 | 4/6 | keep |
| shots over 2.5 | 47,345 | **+0.1323** | +1.051 | 5/6 | keep |
| shots over 3.5 | 47,345 | +0.1025 | +1.026 | 5/6 | keep, **upgraded** |
| saves over 24.5 | 2,655 | +0.0572 | +1.270 | 4/6 | keep, **downgraded** |
| saves over 27.5 | 2,655 | +0.0563 | +1.310 | 5/6 | keep, **downgraded** |
| saves over 30.5 | 2,655 | +0.0463 | +1.293 | 6/6 | keep |
| home −1.5 | 1,419 | +0.0279 | +0.947 | 7/7 | keep |
| moneyline | 1,419 | +0.0181 | +0.785 | 8/8 | keep, reference |
| over 5.5 goals | 1,419 | +0.0005 | +0.588 | 5/5 | **removed** |
| goes to overtime | 1,419 | −0.0019 | +0.338 | 5/6 | **removed** |

### A correction on saves

§17 reported that the saves model "removes 13.1% of the error a
league-average guess makes" and the page was graded `primary` on that basis.
That figure is the R² on the **save count**. It is not the error cut on the
quantity anyone actually bets, which is the probability of clearing a line.

Measured on that: **+4.6% to +5.7%**, under half what the shots markets
manage, and the slopes near 1.3 say the probabilities are still too timid
even after the §17 level correction. The market is real and worth showing —
the baseline it beats is genuinely perverse, since a goalie's own recent
average is worse than guessing the league mean — but it is a `usable`
market, not a `primary` one. Grades corrected.

The lesson generalises: an error cut on a count and a Brier skill on a line
are different numbers, and quoting the flattering one is how a market gets
over-graded. Every grade in `export_tool.py` now rests on the same measure.

### Over 3.5 shots upgraded

Graded `caution` since §9 on the strength of two high bands running about
four points above prediction. That predated the temperature correction in
§11. Re-measured it scores +0.1025 with a slope of 1.026 and 5 of 6 bands
inside their intervals — slightly weaker than the lower lines and in the
tail, but properly calibrated. Raised to `usable`.

### What came off the page

Goal totals and the overtime price, from the upcoming tiles, from the price
calculator, and from `MARKETS` — which is what removes the price boxes,
since the page can only offer a calculator for a market it lists.

Also removed from finished games: the predicted-versus-actual total, "said
over 5.5" and "said overtime". A tick beside a coin-flip forecast implies a
skill the measurement denies. The final score and the actual total stay, as
facts rather than forecasts.

What survives the audit is five markets across two families: skater shots,
goalie saves, the puck line, the moneyline as reference, and the game
record. Every one of them has a measured skill score and a calibration slope
behind its grade.

## 20. One unit on every prediction (2026-10-16)

`flat_bet.py`. Real posted prices exist only for the moneyline, so this
splits into what was measured and what is conditional on a price never
observed.

### The moneyline, at the price the book actually posted

| | bets | result | ROI | 95% CI |
|---|---|---|---|---|
| **every game** | 1,414 | **−103.8u** | **−7.34%** | [−11.83%, −2.85%] |
| only where the model shows an edge | 577 | −14.7u | −2.55% | [−10.16%, +5.05%] |
| model under 60%, book agrees | 741 | −87.5u | −11.81% | [−18.00%, −5.61%] |
| strong favourite, book agrees | 444 | −7.9u | −1.79% | [−8.70%, +5.13%] |
| backing the book's underdog | 212 | +0.9u | +0.43% | [−13.86%, +14.72%] |

The book's median hold on these games is 4.26%, so a bettor with no edge
loses roughly that. Betting every pick loses nearly twice it, because the
largest bucket is also the worst one.

**A correction.** §14 recorded −8.1% for "every model pick". That figure is
the 1,185 games where the model and the book agree on the side, not all
1,414. Flat-betting every pick is **−7.34%**. The page carried the wrong
number and has been fixed.

### The props, where no price was ever observed

The useful question is not what would have been won but how much vig the
edge can carry. Both bounds are computable: a book pricing every line at the
league base rate is the softest it could plausibly be, and one pricing at
the realised rate in the model's own probability band is the hardest.

| market | hold | vs a lazy book | vs a sharp book |
|---|---|---|---|
| shots over 1.5 | 0% | +31.6% | **+0.4%** |
| | 6% | +24.1% | −5.3% |
| shots over 2.5 | 0% | +36.5% | **+0.2%** |
| | 6% | +28.7% | −5.5% |
| saves over 27.5 | 0% | +18.9% | **−0.9%** |
| | 6% | +12.1% | −6.5% |

The second column is the whole project in one number. **Against a book that
prices these lines correctly, the model's edge is 0.4%, 0.2% and −0.9% —
nothing, before any vig at all.** Its enormous margin in the first column is
an edge over a naive baseline, not over a bookmaker.

This is the same conclusion the moneyline reached, arriving from the other
direction. The model removes 13% of the error a league-average guess makes
on shots on goal, which is a real and substantial piece of modelling, and it
buys nothing against a counterparty who has also done the modelling. Prop
markets are softer than moneylines, so the true answer sits somewhere
between the columns — but the right column is the one to plan against.
