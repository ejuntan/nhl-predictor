# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Anyone with the link. The owner is the primary user and reads it on desktop,
but the page is shared and must stand up to a cold reader who arrives with no
tour and no prior knowledge of the model.

The dominant job is **shopping for a bet**: finding where the model's number
disagrees with a price a sportsbook is posting, and deciding whether that
disagreement is worth acting on. Checking tonight's slate and reviewing how
the model has been doing are secondary but real.

## Product Purpose

Predict NHL game outcomes and player shot volume from the league's own public
data, publish those predictions before puck drop, and report honestly how they
have actually performed against real posted prices.

Success is not profit and the page does not claim it. Success is that a reader
can tell, for any number shown, how much skill is behind it — and that the
page never implies more than has been measured.

## Positioning

Two things a neighbouring public model page could not truthfully copy:

1. **Every claim on the page is a measurement with an interval**, including the
   ones that embarrass the model. The page states that flat betting its picks
   lost 7.3%, that its value picks lost 2.6%, and that one named situation is
   an established loser. Markets it could not justify were removed rather than
   labelled.
2. **A prospective frozen log.** The model was frozen on 2026-10-15 and every
   prediction since is logged before puck drop, so the record is not a
   backtest's self-report.

## Operating Context

- Read on **desktop**, usually before a slate, often beside a sportsbook's own
  page with prices open in another tab.
- The reader is comparing two numbers: the model's probability and the book's
  implied one. The decision is whether the gap is real.
- A slate is 2–14 games. The page shows one day at a time, with finished days
  reachable to the left and ~8 upcoming days to the right.

## Capabilities and Constraints

**What it shows:** per-game win probabilities with the regulation/overtime
split, puck line, a board ranking every game by edge against posted prices, a
fifteen-prop daily shortlist, shots-on-goal and goalie-saves props, finished
games against what was predicted, and a season record.

**Markets by measured grade:** shots on goal 1.5/2.5 (primary), puck line and
goalie saves (usable), moneyline (reference — shown for context, not to bet
into). Goal totals and the overtime price were **removed** as uninformative.

**Technical constraints that future work must respect:**

- The page is one self-contained HTML file built by `tool/build.py` from
  `tool/head.html` + `tool/body.html` + `tool/app.js` plus an inlined JSON
  payload. **Never edit `tool/index.html` directly** — it is a build artifact.
- Vanilla JS, no framework, no build step beyond that concatenation. Served by
  Flask on Render, which swaps the data block at request time.
- Five suites gate every publish: `rectest.js`, `linetest.py`, `proptest.py`,
  `mathtest.js`, `daytest.js`. A failure stops the publish entirely.
- The payload key contract between `export_tool.py` and `app.js` is live; any
  renamed or dropped key breaks the page and the tests.
- Rebuilt once daily by `daily.sh`; scores refresh server-side on a 60s floor
  and odds on 900s. The page states its own age when the build has stalled.

**Terminology the page uses:** edge, break-even, no-vig, puck line, the board,
value pick, measured bucket, Brier skill, calibration.

## Brand Commitments

Name: **NHL Daily Predictions**. Voice: plain, specific, and unflattering to
itself — it states what was measured, with the sample size and the interval,
and says plainly when something does not work. No hype, no hedging, no
implied certainty. Not affiliated with the NHL or any sportsbook, and not
advice; this must remain visible.

## Evidence on Hand

Real, measured, and already in the payload or `MODEL.md`:

- **Live record:** 167-114 (59.4%) over 281 games as of 2026-11-09, of which a
  growing share are logged pre-game rather than backtested.
- **Clean holdout:** 54.8% accuracy over ~1,417 games the xG fit never saw.
- **Against real prices (1,414 games):** model log loss 0.682 vs the market's
  0.679; one unit on every pick returned **−7.3%** [−11.9%, −2.9%]; value picks
  **−2.6%**; the sub-60%-confidence bucket **−12.8%** with its whole interval
  below zero — the one established loser.
- **Shots on goal:** removes 13.3% of a league-average baseline's error and
  beats a player's own recent average at t=8.3 over 47,000 skater-games.
- **Prop shortlist:** over 178 holdout days its side came in 64.3% against the
  66.0% the model claimed; needs −180 to break even.
- **Overtime winner:** 49.6% on 357 holdout overtimes, Brier skill −4.73%
  [−9.60%, −0.52%] — worse than a coin flip.
- **Back-to-backs:** the road team loses ~7–9.9 points of win probability; the
  home side's effect is not established.
- Built from 5,697 games, 679,052 shot attempts, 213,137 ice-time records.

**Must not be fabricated:** sportsbook prices. Where no price was observed the
page says so rather than inventing one. There is no free archive of historical
NHL closing moneylines, so the market comparison uses the prices actually
captured and says which.

## Product Principles

1. **Every number carries its own evidence.** A figure with no measured skill
   behind it comes off the page rather than being labelled.
2. **The reader is shopping, so lead with the comparison.** The model's number
   next to the book's, the gap between them, and what that gap has historically
   returned.
3. **Tell the reader when to walk away.** The established losing situation is
   marked on the game it applies to, not buried in a footnote.
4. **A stalled page must confess.** Looking alive while serving frozen
   predictions is the worst failure this page can have.
5. **Honesty is the product, not a disclaimer.** The limits are primary
   content and must stay legible to a cold reader.

## Accessibility & Inclusion

Standard web expectations: visible keyboard focus, respect for
`prefers-reduced-motion`, a working light and dark theme, and no information
conveyed by color alone — the board's chips, the hit/miss marks and the edge
column all need a non-color cue.
