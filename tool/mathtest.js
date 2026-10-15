var state = { fmt: "am", bankroll: 1000, kelly: 0.25 };
function toDecimal(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  // Sportsbook pages render the minus as U+2212 or a dash, and that is what
  // lands in the field when a price is pasted. Normalise before parsing, or
  // "-110" silently becomes +110.
  var txt = String(raw)
    .replace(/[\u2212\u2012\u2013\u2014\u2015\uFF0D]/g, "-")
    .replace(/\u002B|\uFF0B/g, "+")
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
function evaluate(p, dec) {
  if (!dec || !(p > 0 && p < 1)) return null;
  var implied = 1 / dec;
  var edge = p - implied;
  var evPer1 = p * (dec - 1) - (1 - p);
  var kellyFull = (dec - 1) > 0 ? (p * dec - 1) / (dec - 1) : 0;
  return { implied: implied, edge: edge, ev100: evPer1 * 100,
           stake: Math.max(0, kellyFull) * state.kelly * state.bankroll };
}
function ck(label, got, want, tol) {
  tol = tol || 1e-6;
  var ok = (want === null) ? got === null : Math.abs(got - want) <= tol;
  print((ok ? "PASS  " : "FAIL  ") + label + "  got=" + got + "  want=" + want);
  return ok;
}
var all = true;
print("-- parsing book prices --");
all &= ck("-110 -> decimal", toDecimal("-110"), 1 + 100/110, 1e-9);
all &= ck("+150 -> decimal", toDecimal("+150"), 2.5, 1e-9);
all &= ck("1.90 stays decimal", toDecimal("1.90"), 1.90, 1e-9);
all &= ck("2.5 stays decimal", toDecimal("2.5"), 2.5, 1e-9);
all &= ck("blank -> null", toDecimal(""), null);
all &= ck("junk -> null", toDecimal("abc"), null);
all &= ck("en-dash minus 110", toDecimal("−110"), 1 + 100/110, 1e-9);

print("-- fair odds from a probability --");
all &= ck("p=0.50 -> -100", decToAmerican(1/0.50), -100);
all &= ck("p=0.60 -> -150", decToAmerican(1/0.60), -150);
all &= ck("p=0.40 -> +150", decToAmerican(1/0.40), 150);
all &= ck("p=0.25 -> +300", decToAmerican(1/0.25), 300);
all &= ck("p=0.80 -> -400", decToAmerican(1/0.80), -400);

print("-- edge / EV / stake, hand-checked --");
var e = evaluate(0.55, toDecimal("-110"));
all &= ck("implied of -110", e.implied, 110/210, 1e-9);
all &= ck("edge at p=.55 vs -110", e.edge, 0.55 - 110/210, 1e-9);
all &= ck("EV per $100", e.ev100, 5.0, 1e-7);
all &= ck("quarter-Kelly stake on $1000", e.stake, (0.05/(100/110))*0.25*1000, 1e-6);

print("-- a price with no value must come out negative --");
var n = evaluate(0.45, toDecimal("-110"));
all &= ck("edge is negative", n.edge < 0 ? 1 : 0, 1);
all &= ck("EV is negative", n.ev100 < 0 ? 1 : 0, 1);
all &= ck("stake floors at 0", n.stake, 0);

print("-- fair price must give exactly zero edge and zero EV --");
var z = evaluate(0.60, 1/0.60);
all &= ck("edge", z.edge, 0, 1e-12);
all &= ck("EV", z.ev100, 0, 1e-9);
all &= ck("stake", z.stake, 0, 1e-12);

print(all ? "\nALL MATH CHECKS PASS" : "\nFAILURES PRESENT");
