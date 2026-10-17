(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("nhl-data").textContent);
  var GAMES = DATA.games, PROPS = DATA.props, MK = DATA.markets;
  var PAST = DATA.past || [], PASTP = DATA.pastProps || [];
  var RECORD = DATA.record || { w: 0, l: 0, acc: 0 };
  var LINES = DATA.lines || {}, BUCKETS = DATA.buckets || {};
  var SAVES = DATA.saves || [];
  var INPLAY = DATA.inplay || [];
  var VP = DATA.valuePicks || null;
  var LINE_AT = DATA.lineAt || "";
  // Past days come first, then games under way, then the upcoming slate.
  // A game in progress used to appear in neither list: it is no longer
  // FUT, so it was not upcoming, and not yet FINAL, so it was not past.
  var ALL = PAST.concat(INPLAY, GAMES);

  var TEAM = {
    ANA: "Anaheim", BOS: "Boston", BUF: "Buffalo", CAR: "Carolina",
    CBJ: "Columbus", CGY: "Calgary", CHI: "Chicago", COL: "Colorado",
    DAL: "Dallas", DET: "Detroit", EDM: "Edmonton", FLA: "Florida",
    LAK: "Los Angeles", MIN: "Minnesota", MTL: "Montreal", NJD: "New Jersey",
    NSH: "Nashville", NYI: "NY Islanders", NYR: "NY Rangers", OTT: "Ottawa",
    PHI: "Philadelphia", PIT: "Pittsburgh", SEA: "Seattle", SJS: "San Jose",
    STL: "St. Louis", TBL: "Tampa Bay", TOR: "Toronto", UTA: "Utah",
    VAN: "Vancouver", VGK: "Vegas", WPG: "Winnipeg", WSH: "Washington"
  };
  var name = function (abbr) { return TEAM[abbr] || abbr; };

  // Below this, an edge is inside the vig and not worth calling an edge.
  // It used to be a control; it is a constant because every measurement
  // behind this page says the threshold is not where the money is.
  var MIN_EDGE = 3;

  var state = { day: null, odds: false, fmt: "am", prices: {} };

  // ---------- jumping from the board to a card ----------
  // The payload carries no game id, but a date and two teams name a game
  // uniquely, which is all an anchor needs.
  function anchorId(g) { return "g-" + g.d + "-" + g.a + "-" + g.h; }

  /** Scroll that game's card into view and mark it, so the row you clicked
   *  and the card you land on are visibly the same game. */
  function jumpTo(g) {
    var card = document.getElementById(anchorId(g));
    if (!card) return;
    var still = false;
    try {
      still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (e) { /* no matchMedia */ }
    try {
      card.scrollIntoView({ behavior: still ? "auto" : "smooth",
                            block: "center" });
    } catch (e) { card.scrollIntoView(); }
    card.classList.remove("land");
    void card.offsetWidth;                 // restart the animation
    card.classList.add("land");
    clearTimeout(jumpTo.t);
    jumpTo.t = setTimeout(function () { card.classList.remove("land"); }, 1600);
  }

  // ---------- storage (per viewer; can be unavailable) ----------
  function load() {
    try {
      var raw = localStorage.getItem("nhl-daily-v1");
      if (!raw) return;
      var s = JSON.parse(raw);
      ["odds", "fmt"].forEach(function (k) {
        if (s[k] !== undefined && s[k] !== null) state[k] = s[k];
      });
      if (s.prices && typeof s.prices === "object") state.prices = s.prices;
    } catch (e) { /* private window or blocked storage */ }
  }
  var st = null;
  function save() {
    clearTimeout(st);
    st = setTimeout(function () {
      try {
        localStorage.setItem("nhl-daily-v1", JSON.stringify({
          odds: state.odds, fmt: state.fmt, prices: state.prices
        }));
      } catch (e) { /* ignore */ }
    }, 250);
  }

  // ---------- odds maths (checked in tool/mathtest.js) ----------
  function toDecimal(raw) {
    if (raw === null || raw === undefined || raw === "") return null;
    // Sportsbook pages render the minus as U+2212 or a dash, and that is what
    // lands in the field when a price is pasted. Normalise first, or "-110"
    // silently becomes +110.
    var txt = String(raw)
      .replace(/[−‒–—―－]/g, "-")
      .replace(/＋/g, "+")
      .replace(/[^0-9+.-]/g, "");
    var v = parseFloat(txt);
    if (!isFinite(v) || v === 0) return null;
    if (v >= 100 || v <= -100) return v > 0 ? 1 + v / 100 : 1 + 100 / (-v);
    if (v > 1 && v < 100) return v;
    return null;
  }
  function decToAmerican(d) {
    if (!d || d <= 1) return null;
    var p = 1 / d;
    return p >= 0.5 ? -Math.round(100 * p / (1 - p)) : Math.round(100 * (1 - p) / p);
  }
  function fairOdds(p) {
    if (!(p > 0 && p < 1)) return "—";
    if (state.fmt === "dec") return (1 / p).toFixed(2);
    var a = decToAmerican(1 / p);
    return a === null ? "—" : (a > 0 ? "+" + a : String(a));
  }
  function evaluate(p, dec) {
    if (!dec || !(p > 0 && p < 1)) return null;
    return { edge: p - 1 / dec, ev100: (p * (dec - 1) - (1 - p)) * 100 };
  }
  function verdict(ev, grade) {
    if (!ev) return { cls: "chip-none", txt: "—" };
    if (grade === "reference" || grade === "biased") return { cls: "chip-skip", txt: "SKIP" };
    if (ev.edge * 100 < MIN_EDGE) return { cls: "chip-thin", txt: "THIN" };
    if (grade === "caution") return { cls: "chip-thin", txt: "CARE" };
    return { cls: "chip-play", txt: "PLAY" };
  }

  var pct = function (v) { return Math.round(v * 100) + "%"; };
  var pct1 = function (v) { return (v * 100).toFixed(1) + "%"; };
  var signed = function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1); };
  var signedInt = function (v) {
    if (v === null || v === undefined) return "—";
    return (v > 0 ? "+" : "") + v;
  };
  // A posted price, in whichever format the viewer picked. The odds-format
  // control has to reach every price on the page, not only the ones the
  // calculator derives.
  function priceShown(american) {
    if (american === null || american === undefined) return "—";
    if (state.fmt !== "dec") return signedInt(american);
    var d = toDecimal(american);
    return d === null ? "—" : d.toFixed(2);
  }

  /** Plain-English strength of the lean, so a number is not the only cue. */
  function leanWord(p) {
    var m = Math.max(p, 1 - p);
    if (m < 0.54) return "near a coin flip";
    if (m < 0.60) return "a slight lean";
    if (m < 0.67) return "a clear favourite";
    return "a strong favourite";
  }

  // ---------- small DOM helpers ----------
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function td(tr, text, cls) { var c = el("td", cls, text); tr.appendChild(c); return c; }
  // A posted price fills the box so the calculator is answering about a
  // real number, not a blank. A number the viewer typed always wins.
  function seedPrices() {
    Object.keys(LINES).forEach(function (k) {
      var L = LINES[k];
      var put = function (suffix, v) {
        if (v === null || v === undefined) return;
        var key = "g|" + k + "|" + suffix;
        if (!state.prices[key]) state.prices[key] = String(v);
      };
      put("ph", L.hml);
      put("pa", L.aml);
      if (L.tot === 5.5) { put("o55", L.oo); }
      if (L.tot === 6.5) { put("o65", L.oo); }
    });
  }

  function oddsInput(key, onChange) {
    var i = el("input", "odds-in");
    i.type = "text";
    i.placeholder = state.fmt === "dec" ? "1.90" : "-110";
    i.setAttribute("aria-label", "Sportsbook price");
    if (state.prices[key]) { i.value = state.prices[key]; i.classList.add("has"); }
    i.addEventListener("input", function () {
      var v = i.value.trim();
      if (v === "") { delete state.prices[key]; i.classList.remove("has"); }
      else { state.prices[key] = v; i.classList.add("has"); }
      save();
      onChange();
    });
    return i;
  }

  // ---------- days ----------
  /** Every day on the board, with that day's prediction record if it is done. */
  function dayList() {
    var map = {};
    PAST.forEach(function (g) {
      var e = map[g.d] || (map[g.d] = { d: g.d, n: 0, w: 0, l: 0, past: true });
      e.n += 1;
      if (g.hit) e.w += 1; else e.l += 1;
    });
    GAMES.forEach(function (g) {
      var e = map[g.d] || (map[g.d] = { d: g.d, n: 0, w: 0, l: 0, past: false });
      e.n += 1;
    });
    INPLAY.forEach(function (g) {
      var e = map[g.d] || (map[g.d] = { d: g.d, n: 0, w: 0, l: 0, past: false });
      e.n += 1;
      e.live = (e.live || 0) + 1;
    });
    return Object.keys(map).sort().map(function (d) { return map[d]; });
  }
  function dayRecord(d) {
    var w = 0, l = 0;
    PAST.forEach(function (g) {
      if (g.d !== d) return;
      if (g.hit) w += 1; else l += 1;
    });
    return { w: w, l: l, n: w + l };
  }
  function firstUpcoming() {
    return GAMES.length ? dayList().filter(function (x) { return !x.past; })[0].d : null;
  }
  function parseDay(d) { return new Date(d + "T12:00:00Z"); }
  var WD = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  var MO = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function dayShort(d) {
    var t = parseDay(d);
    return { wd: WD[t.getUTCDay()], dd: t.getUTCDate(), mo: MO[t.getUTCMonth()] };
  }
  function dayLong(d) {
    var s = dayShort(d);
    var full = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday"][parseDay(d).getUTCDay()];
    return full + " " + s.dd + " " + s.mo;
  }

  function renderDays() {
    var host = document.getElementById("days");
    host.textContent = "";
    var list = dayList();
    var drawnSep = false;
    var selected = null;
    list.forEach(function (x) {
      // One divider between finished days and the upcoming slate.
      if (!x.past && !drawnSep && PAST.length) {
        host.appendChild(el("div", "daysep"));
        drawnSep = true;
      }
      var s = dayShort(x.d);
      var cls = "day";
      if (x.past) {
        cls += " past " + (x.w > x.l ? "hit" : (x.w < x.l ? "miss" : ""));
      }
      var b = el("button", cls);
      b.type = "button";
      b.setAttribute("aria-pressed", String(x.d === state.day));
      b.appendChild(el("span", "dw", s.wd));
      b.appendChild(el("span", "dd", String(s.dd)));
      b.appendChild(el("span", "dn", x.past ? x.w + "-" + x.l
        : x.n + (x.n === 1 ? " game" : " games")));
      b.title = x.past
        ? dayLong(x.d) + ": model went " + x.w + "-" + x.l
        : dayLong(x.d) + ": " + x.n + " scheduled";
      b.addEventListener("click", function () {
        state.day = x.d;
        renderDays();
        renderSlate();
      });
      if (x.d === state.day) selected = b;
      host.appendChild(b);
    });
    // Bring the chosen day into view without scrolling the page itself.
    if (selected) {
      host.scrollLeft = Math.max(0, selected.offsetLeft - host.clientWidth / 2
        + selected.offsetWidth / 2);
    }
  }

  // ---------- one game ----------
  function savesFor(g) {
    return SAVES.filter(function (s) {
      return s.d === g.d && (s.t === g.h || s.t === g.a)
        && (s.o === g.h || s.o === g.a);
    }).sort(function (a, b) { return b.xs - a.xs; });
  }

  function savesBlock(g) {
    var rows = savesFor(g);
    if (!rows.length) return null;
    var d = el("details", "props");
    d.appendChild(el("summary", null,
      "Goalie saves — " + rows.length +
      (rows.length === 1 ? " starter" : " starters")));
    var w = el("div", "pwrap");
    var t = el("table", "ptable");
    var th = el("thead"), htr = el("tr");
    ["Goalie", "Team", "Minutes", "Saves", "o24.5", "o27.5", "o30.5"]
      .forEach(function (x) { htr.appendChild(el("th", null, x)); });
    th.appendChild(htr);
    t.appendChild(th);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      var c = el("td", "pname");
      c.appendChild(document.createTextNode(r.g));
      if (r.bk) c.appendChild(el("span", "flag", "backup assumed"));
      tr.appendChild(c);
      td(tr, r.t);
      td(tr, r.toi.toFixed(1), "num");
      td(tr, r.xs.toFixed(1), "num");
      td(tr, pct(r.p245), "num");
      td(tr, pct(r.p275), "num");
      td(tr, pct(r.p305), "num");
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    w.appendChild(t);
    d.appendChild(w);
    d.appendChild(el("p", "lwhy",
      "These remove 4.6% to 5.7% of the base rate's error, under half what "
      + "shots on goal manage. What makes the market interesting is the "
      + "baseline it beats: a goalie's own recent average is worse than "
      + "guessing the league mean, because past save counts mostly reflect "
      + "the opponents he happened to face."));
    return d;
  }

  function propsFor(g) {
    var pool = isPast(g) ? PASTP : PROPS;
    return pool.filter(function (p) {
      return p.d === g.d &&
        ((p.t === g.h && p.o === g.a) || (p.t === g.a && p.o === g.h));
    }).sort(function (a, b) { return b.xs - a.xs; });
  }
  function isLive(g) { return !!g.st; }
  function isPast(g) {
    return !isLive(g) && g.hsc !== undefined && g.hsc !== null;
  }
  function gkey(g) { return g.d + "|" + g.a + g.h; }
  function lineFor(g) { return LINES[gkey(g)] || null; }

  // The chip on a posted line is the bucket this game falls into, and the
  // bucket is a measurement: what backing the model's side in that
  // situation actually returned over 1,414 games with real prices. The chip
  // shows that number rather than a word -- "UNPROVEN" told a viewer
  // nothing, whereas "+2.7% +/-10" says both the estimate and that it is
  // indistinguishable from zero.
  function bucketChip(b) {
    var def = BUCKETS[b] || {};
    var roi = def.roi;
    if (roi === undefined || roi === null) return { cls: "chip-none", txt: "—" };
    var txt = signed(roi * 100) + "%";
    if (def.ci) txt += " ±" + Math.round(def.ci * 100);
    return { cls: roi >= 0 ? "chip-thin" : "chip-skip", txt: txt,
             why: def.why || "", roi: roi, n: def.n };
  }

  // The price at which the model's own probability breaks even. Shop above
  // this and the model thinks you have value; below it, you do not. Same
  // idea as the MLB page's "need >=" column.
  function breakEven(p) {
    if (!p) return "—";
    var dec = 1 / p;
    return state.fmt === "dec" ? dec.toFixed(2) : decToAmerican(dec);
  }

  function lineBlock(g) {
    var L = lineFor(g);
    var box = el("div", "line");
    if (!L) {
      box.appendChild(el("span", "lmuted", "No price posted yet."));
      return box;
    }
    var ch = bucketChip(L.b);
    // One row of figures, not a second bordered grid: the card already has
    // one above it, and two identical grids flattened the card.
    [["book", L.book],
     ["price", priceShown(L.side === g.h ? L.hml : L.aml)],
     ["model", pct1(L.pm)],
     ["line", pct1(L.pk)],
     ["need", breakEven(L.pm)],
     ["edge", signed(L.edge * 100) + "%"]].forEach(function (x, i) {
      var c = el("span", "lf" + (i === 5 ? (L.edge > 0 ? " pos" : " neg") : ""));
      c.appendChild(el("i", "k", x[0]));
      c.appendChild(el("b", null, String(x[1])));
      box.appendChild(c);
    });
    var chip = el("span", "chip " + ch.cls, ch.txt);
    chip.title = ch.why;
    box.appendChild(chip);
    return box;
  }

  function priceRow(label, p, key, grade) {
    var dec = toDecimal(state.prices[key]);
    var ev = evaluate(p, dec);
    var tr = el("tr");
    td(tr, label);
    td(tr, pct1(p), "num");
    td(tr, fairOdds(p), "num zero");
    var c = el("td");
    c.appendChild(oddsInput(key, renderSlate));
    tr.appendChild(c);
    var e = td(tr, ev ? signed(ev.edge * 100) + "%" : "—",
               "num " + (ev ? (ev.edge > 0 ? "pos" : "neg") : "zero"));
    var v = verdict(ev, grade);
    td(tr, ev ? signed(ev.ev100) : "—",
       "num hide-s " + (ev ? (ev.ev100 > 0 ? "pos" : "neg") : "zero"));
    var vd = el("td");
    vd.appendChild(el("span", "chip " + v.cls, v.txt));
    tr.appendChild(vd);
    return tr;
  }

  function liveCard(g) {
    var fin = g.st === "FINAL" || g.st === "OFF";
    var card = el("div", "game " + (fin ? "done " + (g.hit ? "won" : "lost")
                                        : "live"));
    card.id = anchorId(g);
    var head = el("div", "ghead");
    var mu = el("div", "matchup");
    mu.appendChild(document.createTextNode(name(g.a)));
    mu.appendChild(el("span", "at", "at"));
    mu.appendChild(document.createTextNode(name(g.h)));
    head.appendChild(mu);
    var pick = g.ph > 0.5 ? g.h : g.a;
    var conf = Math.max(g.ph, g.pa);
    var res = el("div", "res");
    if (!fin && g.pn) {
      var per = g.per === "OT" ? "OT" : (g.per === "SO" ? "SO" : "P" + g.pn);
      res.appendChild(el("span", "lmuted",
        per + (g.clk ? " · " + g.clk : "")));
    }
    if (fin) {
      if (g.per === "OT" || g.per === "SO") {
        res.appendChild(el("span", "lmuted", g.per));
      }
      res.appendChild(el("span", "chip " + (g.hit ? "chip-hit" : "chip-miss"),
        g.hit ? "called it" : "missed"));
    } else {
      res.appendChild(el("span", "chip chip-live",
        g.st === "CRIT" ? "final minutes" : "in progress"));
    }
    head.appendChild(res);
    card.appendChild(head);

    var sc = el("div", "score");
    var lead = g.asc === g.hsc ? null : (g.asc > g.hsc ? g.a : g.h);
    var aw = el("span", lead === g.a ? "win" : "wl", g.a + " " + g.asc);
    var hm = el("span", lead === g.h ? "win" : "wl", g.h + " " + g.hsc);
    sc.appendChild(aw);
    sc.appendChild(el("span", "wl", "–"));
    sc.appendChild(hm);
    card.appendChild(sc);

    var p = el("p", "lwhy");
    p.appendChild(document.createTextNode(
      "Called " + name(pick) + " at " + pct1(conf) + " before the game"));
    // Never let a logged prediction and a backtest one pass for the same
    // thing: they come from different model instances and can disagree on
    // the side.
    p.appendChild(el("span", "flag", g.src === "logged"
      ? "logged pre-game" : "backtest estimate"));
    if (fin) {
      // Said plainly, because the day's record above will not match a
      // count of the cards until the overnight rebuild.
      p.appendChild(document.createTextNode(
        " · final since this page was built, so not in the record below yet"));
    }
    card.appendChild(p);
    return card;
  }

  function gameCard(g) {
    var done = isPast(g);
    var card = el("div", "game" + (done
      ? " done " + (g.hit ? "won" : "lost") : ""));
    card.id = anchorId(g);

    // header: matchup, then either the pick or how the pick turned out
    var head = el("div", "ghead");
    var m = el("div", "matchup");
    m.appendChild(document.createTextNode(name(g.a)));
    m.appendChild(el("span", "at", "at"));
    m.appendChild(document.createTextNode(name(g.h)));
    head.appendChild(m);
    var favHome = g.ph >= g.pa;
    var favName = favHome ? name(g.h) : name(g.a);
    if (done) {
      var res = el("div", "res");
      if (g.src === "backtest") {
        var b = el("span", "flag", "backtest estimate");
        b.title = "Played before the pre-game log began, so this is the "
          + "walk-forward model's call rather than one that was recorded "
          + "in advance.";
        res.appendChild(b);
      }
      res.appendChild(el("span", "chip " + (g.hit ? "chip-hit" : "chip-miss"),
        g.hit ? "CALLED IT" : "MISSED"));
      var sub = el("span", "pick");
      sub.appendChild(document.createTextNode("picked "));
      sub.appendChild(el("b", null, favName));
      sub.appendChild(document.createTextNode(" at " + pct(Math.max(g.ph, g.pa))));
      res.appendChild(sub);
      head.appendChild(res);
    } else {
      var pick = el("div", "pick");
      pick.appendChild(document.createTextNode("Model leans "));
      pick.appendChild(el("b", null, favName));
      pick.appendChild(document.createTextNode(" — " + leanWord(g.ph)));
      head.appendChild(pick);
    }
    card.appendChild(head);

    // final score, for a game that has been played
    if (done) {
      var sc = el("div", "score");
      var awayWon = g.asc > g.hsc;
      var sa = el("span", awayWon ? "win" : "wl", g.a + " " + g.asc);
      var sh = el("span", awayWon ? "wl" : "win", g.hsc + " " + g.h);
      sc.appendChild(sa);
      sc.appendChild(el("span", "wl", "–"));
      sc.appendChild(sh);
      if (g.wentot) sc.appendChild(el("span", "ot", "OT/SO"));
      card.appendChild(sc);
    }

    // win probability split
    var sp = el("div", "split");
    var labels = el("div", "splitlabels");
    var la = el("span", "away");
    la.appendChild(el("span", "t", g.a + " "));
    la.appendChild(el("span", "p", pct(g.pa)));
    var lh = el("span", "home");
    lh.appendChild(el("span", "p", pct(g.ph)));
    lh.appendChild(el("span", "t", " " + g.h));
    labels.appendChild(la);
    labels.appendChild(lh);
    sp.appendChild(labels);
    var bar = el("div", "bar");
    var ia = el("i", "ia"); ia.style.width = (g.pa * 100) + "%";
    var ih = el("i", "ih"); ih.style.width = (g.ph * 100) + "%";
    bar.appendChild(ia); bar.appendChild(ih);
    sp.appendChild(bar);
    card.appendChild(sp);

    // four stats that matter
    var stats = el("div", "stats");
    function stat(k, v, n) {
      var s = el("div", "stat");
      s.appendChild(el("span", "k", k));
      s.appendChild(el("span", "v", v));
      if (n) s.appendChild(el("span", "n", n));
      stats.appendChild(s);
    }
    if (done) {
      function vs(predicted, actual) {
        var w = el("span", "vs");
        w.appendChild(document.createTextNode(predicted));
        w.appendChild(el("span", "arrow", "→"));
        w.appendChild(el("span", "act", actual));
        return w;
      }
      var s1 = el("div", "stat");
      s1.appendChild(el("span", "k", "Score, predicted vs final"));
      var v1 = el("span", "v");
      v1.appendChild(vs(g.la.toFixed(1) + "–" + g.lh.toFixed(1),
                        g.asc + "–" + g.hsc));
      s1.appendChild(v1);
      s1.appendChild(el("span", "n", g.a + " – " + g.h));
      stats.appendChild(s1);

      // Only the fact, not the forecast: the expected total and the
      // overtime price are not markets this page carries, so showing how
      // close they came would imply a skill neither has.
      stat("Total goals", String(g.atot), "final");
    } else {
      // The expected total and the overtime price are not shown. Both are
      // right on average and neither moves with the game: the total
      // explains 0.33% of the variance in real totals, and the overtime
      // price scores slightly worse than quoting the league rate. A number
      // on a card implies it says something about this game, and those two
      // do not. The finished-game panels below still show what was
      // predicted against what happened, because that is the record.
      stat("Expected score", g.la.toFixed(1) + " – " + g.lh.toFixed(1),
           g.a + " – " + g.h);
      stat("Win by 2+", pct(g.pl), name(g.h));
    }
    card.appendChild(stats);

    // goalies, with the backup guess flagged (upcoming games only)
    var gl = el("div", "goalies");
    if (done) gl.hidden = true;
    [[g.a, g.ag, g.ab, g.ar], [g.h, g.hg, g.hb, g.hr]].forEach(function (x) {
      var d = el("div", "goalie");
      d.appendChild(el("span", "k", x[0] + " goalie · " + x[3] +
        (x[3] === 1 ? " day rest" : " days rest")));
      var v = el("span", "v", x[1]);
      if (x[2]) v.appendChild(el("span", "flag", "backup assumed"));
      d.appendChild(v);
      gl.appendChild(d);
    });
    card.appendChild(gl);

    // The posted line, for games not yet played. Shown whether or not the
    // calculator is on: the disagreement is the interesting number, and it
    // does not depend on the viewer typing anything.
    if (!done) card.appendChild(lineBlock(g));

    // prices, only in odds mode and only for games not yet played
    if (state.odds && !done) {
      var pr = el("div", "prices");
      pr.appendChild(el("h4", null, "Check a price"));
      var w = el("div", "pwrap");
      var t = el("table", "ptable");
      var th = el("thead");
      var htr = el("tr");
      ["Market", "Model", "Fair", "Book", "Edge", "EV/$100", ""]
        .forEach(function (h, i) {
          var c = el("th", i === 5 ? "hide-s" : null, h);
          htr.appendChild(c);
        });
      th.appendChild(htr);
      t.appendChild(th);
      var tb = el("tbody");
      var id = g.d + "|" + g.a + g.h;
      tb.appendChild(priceRow(name(g.h) + " win", g.ph, "g|" + id + "|ph", MK.moneyline.grade));
      tb.appendChild(priceRow(name(g.a) + " win", g.pa, "g|" + id + "|pa", MK.moneyline.grade));
      tb.appendChild(priceRow(name(g.h) + " −1.5", g.pl, "g|" + id + "|pl", MK.puckline.grade));
      t.appendChild(tb);
      w.appendChild(t);
      pr.appendChild(w);
      card.appendChild(pr);
    }

    // goalie saves, for games not yet played
    if (!done) {
      var sb = savesBlock(g);
      if (sb) card.appendChild(sb);
    }

    // shots props for this game, collapsed
    var ps = propsFor(g);
    if (ps.length) {
      var det = el("details", "props");
      var sum = el("summary", null, done
        ? "Shots on goal — predicted vs actual for " + ps.length + " skaters"
        : "Shots on goal — " + ps.length + " skaters");
      det.appendChild(sum);
      var w2 = el("div", "pwrap");
      var t2 = el("table", "ptable");
      var th2 = el("thead");
      var htr2 = el("tr");
      var cols = done
        ? ["Player", "Ice time", "Shots predicted", "Actual", "Said over 2.5", ""]
        : ["Player", "Ice time", "Shots", "Over 1.5", "Over 2.5", "Over 3.5"];
      if (state.odds && !done) cols = cols.concat(["Book o2.5", "Edge", ""]);
      cols.forEach(function (h) { htr2.appendChild(el("th", null, h)); });
      th2.appendChild(htr2);
      t2.appendChild(th2);
      var tb2 = el("tbody");
      ps.forEach(function (p) {
        var tr = el("tr");
        var n = td(tr, "");
        n.appendChild(el("span", "pname", p.n));
        n.appendChild(el("span", "zero", "  " + p.t));
        td(tr, p.toi.toFixed(1) + " min", "num");
        td(tr, p.xs.toFixed(1), "num");
        if (done) {
          td(tr, String(p.asog), "num act");
          td(tr, pct(p.p25), "num");
          var over = p.asog > 2.5;
          var called = (p.p25 >= 0.5) === over;
          td(tr, over ? "over" : "under",
             "num hitmark " + (called ? "pos" : "neg"));
          tb2.appendChild(tr);
          return;
        }
        td(tr, pct(p.p15), "num");
        td(tr, pct(p.p25), "num");
        td(tr, pct(p.p35), "num");
        if (state.odds) {
          var key = "p|" + p.d + "|" + p.n + "|p25";
          var dec = toDecimal(state.prices[key]);
          var ev = evaluate(p.p25, dec);
          var c = el("td");
          c.appendChild(oddsInput(key, renderSlate));
          tr.appendChild(c);
          td(tr, ev ? signed(ev.edge * 100) + "%" : "—",
             "num " + (ev ? (ev.edge > 0 ? "pos" : "neg") : "zero"));
          var v = verdict(ev, MK.sog_25.grade);
          var vd = el("td");
          vd.appendChild(el("span", "chip " + v.cls, v.txt));
          tr.appendChild(vd);
        }
        tb2.appendChild(tr);
      });
      t2.appendChild(tb2);
      w2.appendChild(t2);
      det.appendChild(w2);
      card.appendChild(det);
    }
    return card;
  }

  // ---------- slate ----------
  /** How the model did on the chosen day, and across the season so far. */

  // The measured record, as a table rather than a paragraph of confidence
  // intervals. It was four sentences repeated in four places; it is now one
  // disclosure the reader opens when they want the numbers.
  function evidencePanel() {
    var d = el("details", "evidence");
    var sum = el("summary");
    sum.appendChild(el("span", "ehead", "Measured at real prices"));
    sum.appendChild(el("span", "esum", VP
      ? "this model's value picks returned " + signed(VP.roi * 100)
        + "% per dollar over " + VP.n + " past bets"
      : "no edge established"));
    d.appendChild(sum);

    var w = el("div", "pwrap");
    var t = el("table", "ptable");
    var th = el("thead"), htr = el("tr");
    ["Situation", "Games", "Won", "Return", "Interval"]
      .forEach(function (x) { htr.appendChild(el("th", null, x)); });
    th.appendChild(htr);
    t.appendChild(th);
    var tb = el("tbody");
    var label = { thin: "Model under 60%, book agrees",
                  dog: "Model backs the book's underdog",
                  fav: "Strong favourite, book agrees" };
    ["thin", "dog", "fav"].forEach(function (k) {
      var b = BUCKETS[k];
      if (!b) return;
      var tr = el("tr");
      td(tr, label[k]);
      td(tr, String(b.n), "num");
      td(tr, pct1(b.win), "num");
      td(tr, signed(b.roi * 100) + "%",
         "num " + (b.roi + b.ci < 0 ? "neg" : "zero"));
      td(tr, "±" + (b.ci * 100).toFixed(1), "num zero");
      tb.appendChild(tr);
    });
    if (VP) {
      var tr = el("tr", "erow");
      td(tr, "Any price above the model's number");
      td(tr, String(VP.n), "num");
      td(tr, pct1(VP.win), "num");
      td(tr, signed(VP.roi * 100) + "%", "num zero");
      td(tr, signed(VP.lo * 100) + " to " + signed(VP.hi * 100), "num zero");
      tb.appendChild(tr);
    }
    t.appendChild(tb);
    w.appendChild(t);
    d.appendChild(w);
    d.appendChild(el("p", "lwhy",
      "One line is established: when the model is under 60% and the book "
      + "agrees on the side, the whole interval sits below zero. That is "
      + "54% of a typical board. Everything else straddles zero, so an edge "
      + "shown below is a number, not a verdict."));
    return d;
  }

  function segControl(opts, current, pick, label) {
    var seg = el("div", "seg");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", label);
    opts.forEach(function (o) {
      var b = el("button", null, o[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", current === o[0] ? "true" : "false");
      b.addEventListener("click", function () { pick(o[0]); });
      seg.appendChild(b);
    });
    return seg;
  }

  function valueRows(todays) {
    return todays.map(function (g) {
      var L = lineFor(g);
      return L ? { g: g, L: L, ch: bucketChip(L.b) } : null;
    }).filter(Boolean);
  }

  function renderRecord(todays, anyDone) {
    var host = document.getElementById("record");
    host.textContent = "";
    var rec = dayRecord(state.day);

    if (anyDone && rec.n) {
      host.appendChild(el("span", "big", rec.w + "-" + rec.l));
      host.appendChild(document.createTextNode(
        "on " + dayLong(state.day) + " (" + pct(rec.w / rec.n) + ")"));
    } else {
      host.appendChild(el("span", "big", String(todays.length)));
      host.appendChild(document.createTextNode(
        todays.length === 1 ? "game to come" : "games to come"));
    }
    host.appendChild(el("span", "sep", "·"));
    host.appendChild(document.createTextNode("season so far "));
    host.appendChild(el("span", "big",
      RECORD.w + "-" + RECORD.l));
    host.appendChild(document.createTextNode(
      "(" + pct(RECORD.acc) + " of " + (RECORD.w + RECORD.l) + " games)"));

    // Games that finished after this page was built are shown on the day
    // but are not in the figures above, which come from the build. Saying
    // so is better than a day reading 6-0 above four cards marked missed.
    var late = INPLAY.filter(function (g) {
      return g.d === state.day && (g.st === "FINAL" || g.st === "OFF");
    });
    if (late.length) {
      var w = late.filter(function (g) { return g.hit; }).length;
      host.appendChild(el("span", "sep", "·"));
      host.appendChild(el("span", "lmuted",
        late.length + (late.length === 1 ? " game" : " games")
        + " finished since this page was built ("
        + w + "-" + (late.length - w) + "), not counted above"));
    }
    host.hidden = false;
  }

  var sortBy = "edge";

  function rankedTable(rows) {
    var box = el("div", "strip");
    var h = el("div", "striphead");
    h.appendChild(el("h3", null, "The board"));
    h.appendChild(el("span", "lmuted", LINE_AT ? "prices " + LINE_AT : ""));
    h.appendChild(segControl(
      [["am", "American"], ["dec", "Decimal"]], state.fmt,
      function (v) { state.fmt = v; save(); renderSlate(); },
      "Odds format"));
    var seg = segControl(
      [["edge", "Edge"], ["conf", "Model"], ["gap", "Disagreement"]], sortBy,
      function (v) { sortBy = v; renderSlate(); }, "Sort by");
    h.appendChild(seg);
    box.appendChild(h);

    var key = { edge: function (r) { return r.L.edge; },
                conf: function (r) { return r.L.pm; },
                gap: function (r) { return Math.abs(r.L.gap); } }[sortBy];
    rows = rows.slice().sort(function (a, b) { return key(b) - key(a); });

    var w = el("div", "pwrap");
    var t = el("table", "ptable");
    var th = el("thead"), htr = el("tr");
    ["Pick", "Game", "Model", "Line", "Price", "Need", "Edge",
     "Measured"].forEach(function (x) {
      htr.appendChild(el("th", null, x));
    });
    th.appendChild(htr);
    t.appendChild(th);
    var tb = el("tbody");
    // Rank badges go on the value picks only, in edge order, so the three
    // the page would have listed separately are findable in one table.
    var ranked = rows.slice().filter(function (r) { return r.L.edge > 0; })
      .sort(function (a, b) { return b.L.edge - a.L.edge; });
    rows.forEach(function (r) {
      var rank = ranked.indexOf(r) + 1;
      var tr = el("tr", (r.L.edge > 0 ? "val " : "") + "jump");
      tr.title = "Go to " + name(r.g.a) + " at " + name(r.g.h);
      tr.addEventListener("click", function () { jumpTo(r.g); });
      var p = el("td", "pname");
      if (rank > 0 && rank <= 3) p.appendChild(el("span", "badge", "#" + rank));
      p.appendChild(document.createTextNode(name(r.L.side)));
      tr.appendChild(p);
      // A real link in the Game cell, so the jump is reachable from the
      // keyboard and not only from a click anywhere on the row.
      var gc = el("td");
      var a = el("a", "gl", r.g.a + " @ " + r.g.h);
      a.href = "#" + anchorId(r.g);
      a.addEventListener("click", function (ev) {
        ev.preventDefault(); ev.stopPropagation(); jumpTo(r.g);
      });
      gc.appendChild(a);
      gc.appendChild(el("span", "jumpmark", "\u2193"));
      tr.appendChild(gc);
      td(tr, pct1(r.L.pm), "num");
      td(tr, pct1(r.L.pk), "num");
      td(tr, priceShown(r.L.side === r.g.h ? r.L.hml : r.L.aml), "num");
      td(tr, breakEven(r.L.pm), "num zero");
      td(tr, signed(r.L.edge * 100) + "%",
         "num " + (r.L.edge > 0 ? "pos" : "neg"));
      var c = el("td");
      var chip = el("span", "chip " + r.ch.cls, r.ch.txt);
      chip.title = r.ch.why;
      c.appendChild(chip);
      tr.appendChild(c);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    w.appendChild(t);
    box.appendChild(w);

    var est = rows.filter(function (r) {
      return (r.ch.roi + (BUCKETS[r.L.b] || {}).ci) < 0;
    }).length;
    var pos = ranked.length;
    box.appendChild(el("p", "lwhy",
      pos + " of " + rows.length + " games price above the model's number. "
      + est + " are in the one situation measured to lose."));
    return box;
  }

  function valueFinder(todays) {
    var rows = valueRows(todays);
    var frag = document.createDocumentFragment();
    if (!rows.length) {
      var e = el("div", "strip");
      e.appendChild(el("p", "lmuted",
        "No posted prices for this day yet, so there is nothing to shop."));
      frag.appendChild(e);
      return frag;
    }
    frag.appendChild(evidencePanel());
    frag.appendChild(rankedTable(rows));
    return frag;
  }

  function renderSlate() {
    var host = document.getElementById("slate");
    host.textContent = "";
    var todays = ALL.filter(function (g) { return g.d === state.day; });
    var anyDone = todays.some(isPast);
    var anyLive = todays.some(isLive);
    var title = document.getElementById("dayTitle");
    title.textContent = dayLong(state.day) + " ";
    title.appendChild(el("span", "cnt",
      anyLive ? "· " + todays.filter(function (x) {
                  return isLive(x) && x.st !== "FINAL" && x.st !== "OFF";
                }).length + " under way"
              : (anyDone ? "· results" : "· predictions")));
    renderRecord(todays, anyDone);

    if (!todays.length) {
      host.appendChild(el("div", "empty", "No games scheduled."));
      return;
    }
    if (!anyDone) host.appendChild(valueFinder(todays));
    var frag = document.createDocumentFragment();
    todays.forEach(function (g) {
      frag.appendChild(isLive(g) ? liveCard(g) : gameCard(g));
    });
    host.appendChild(frag);
  }

  // ---------- wiring ----------
  function init() {
    load();
    seedPrices();
    // Open on today's slate, with finished days reachable to the left.
    state.day = firstUpcoming() || (dayList().length ? dayList()[0].d : null);
    document.getElementById("asof").textContent = "generated " + DATA.generated;
    document.getElementById("gen").textContent = DATA.generated;

    var om = document.getElementById("oddsMode");
    om.checked = state.odds;
    om.addEventListener("change", function (e) {
      state.odds = e.target.checked; save(); renderSlate();
    });
    renderDays();
    renderSlate();
  }

  init();
})();
