---
version: 1
slug: "tool-index-html"
primary_target: "tool/index.html"
related_targets: ["tool/head.html","tool/body.html","tool/app.js"]
---

Scope: `tool/index.html` — the whole NHL Daily Predictions page, built from
`tool/head.html` + `tool/body.html` + `tool/app.js` by `tool/build.py`.
Visitor mode: **Operate**. Audience: anyone with the link, desktop-first.
Job: find where the model's number disagrees with a posted price and decide
whether the gap is worth acting on. Constraints: never edit `index.html`
directly; the payload key contract with `export_tool.py` is live; five test
suites gate every publish.

## Direction contract

THESIS: A slate is a departure board. Fourteen games rank by edge in fixed
columns that never move, and a price that has changed since you last looked
keeps its lamp until you have seen it. It refuses the arrangement this
category always ships — a sidebar, a row of KPI tiles, soft-shadowed cards
on neutral grey — because a board is read across, not browsed.

OWN-WORLD: Gate-board vernacular rendered as vitreous navy enamel. Board
#16243d with cream lettering #F0EAD8, page ground #E5E6E2 in light and
#0F1520 in dark, steel seams. The board is always the enamel panel, the way
an airport sign was before it was a screen, framed by a seam so it stays a
panel on a dark page too. One saturated colour, vermilion #E8502A, and it
carries the negative only: the miss, the losing edge, the situation measured
to lose. The lamp is cream at full brightness instead, the loudest thing
available on enamel and colliding with nothing. Division is hairline seam,
never card shadow; no rounded-box stack. Barlow Condensed for board rows
and display, Barlow for prose, Azeret Mono uppercase and letterspaced for
every column label, status word and code. Figures are tabular in fixed
cells.

STORY: The visitor sees a ranked board, reads down the edge column, finds the
rows whose price moved, and lands on a game card that tells them what the
model thinks, what the book is posting, and what that exact situation has
historically returned — including, in red, the one situation measured to lose.
They leave knowing which gaps are real and which are the vig.

FIRST VIEWPORT: A thin ink masthead strip with the product name in condensed
caps, the slate date and the season record set as board figures at the right.
Beneath it the day register as one ruled strip of fixed cells, the selected
day amber. Then the board itself, full width, a matte dark panel: mono column
labels over rows ranked by edge, each row carrying pick, game, model, market,
price, break-even, edge, and the measured-bucket verdict as the last column.
Rows whose price moved since this viewer last looked carry an amber left lamp.
No sidebar, no tiles, no cards above the board.

FORM: The gate board, dealt as a challenger and chosen over the assigned
Statistical Abstract and over the Daily Form pick. seed:b0ebcb1871882e4c

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
