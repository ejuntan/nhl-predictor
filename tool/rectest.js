var fs = read("/tmp/payload.json");
var D = JSON.parse(fs);
var T = JSON.parse(read("/tmp/truth.json"));
var PAST = D.past, RECORD = D.record;
var INPLAY = D.inplay || [];

// same functions the page uses
function isFinished(g) { return g.st === "FINAL" || g.st === "OFF"; }
// Games that ended after the build: the page counts these on top of the
// build's own record, so the previous night's late games are not a day late.
var LATE = INPLAY.filter(function (g) {
  return isFinished(g) && g.hit !== undefined && g.hit !== null;
});
function dayRecord(d) {
  var w = 0, l = 0;
  PAST.concat(LATE).forEach(function (g) { if (g.d !== d) return; if (g.hit) w += 1; else l += 1; });
  return { w: w, l: l, n: w + l };
}
function dayList() {
  var map = {};
  PAST.forEach(function (g) {
    var e = map[g.d] || (map[g.d] = { d: g.d, n: 0, w: 0, l: 0, past: true });
    e.n += 1; if (g.hit) e.w += 1; else e.l += 1;
  });
  (D.games||[]).forEach(function (g) {
    var e = map[g.d] || (map[g.d] = { d: g.d, n: 0, w: 0, l: 0, past: false });
    e.n += 1;
  });
  return Object.keys(map).sort().map(function (d) { return map[d]; });
}

var all = true;
function ck(l, g, w) { var ok = g === w; if(!ok) all=false;
  print((ok?"PASS  ":"FAIL  ")+l+"  got="+g+"  want="+w); }

print("-- per-day record matches the source data --");
Object.keys(T.byday).sort().forEach(function (d) {
  var r = dayRecord(d);
  var late = LATE.filter(function (g) { return g.d === d; });
  // The truth file is built from finished games in the database, so a day
  // with results that arrived after the build is expected to run ahead.
  var lw = late.filter(function (g) { return g.hit; }).length;
  ck(d, r.w + "-" + r.l,
     (T.byday[d][0] + lw) + "-" + (T.byday[d][1] + late.length - lw));
});
print("-- season total --");
ck("total", RECORD.w + "-" + RECORD.l, T.total[0] + "-" + T.total[1]);
ck("wins+losses = games", RECORD.w + RECORD.l, PAST.length);
var sw = 0, sl = 0;
dayList().forEach(function (x) { if (x.past) { sw += x.w; sl += x.l; } });
ck("day records sum to the season", sw + "-" + sl, RECORD.w + "-" + RECORD.l);
ck("accuracy", Math.round(RECORD.acc*10000)/10000, Math.round(T.total[0]/(T.total[0]+T.total[1])*10000)/10000);

print("-- the hit flag agrees with the final score and the stated pick --");
var bad = 0;
PAST.forEach(function (g) {
  var homeWon = g.hsc > g.asc;
  var pickedHome = g.ph > 0.5;
  if ((pickedHome === homeWon ? 1 : 0) !== g.hit) bad += 1;
});
ck("rows where hit disagrees with score vs pick", bad, 0);

print("-- past and upcoming days do not overlap --");
var pd = {}, gd = {};
PAST.forEach(function (g) { pd[g.d] = 1; });
(D.games||[]).forEach(function (g) { gd[g.d] = 1; });
var overlap = Object.keys(pd).filter(function (d) { return gd[d]; });
ck("overlapping days", overlap.length, 0);
ck("day buttons", dayList().length, Object.keys(pd).length + Object.keys(gd).length);

print("-- every past game carries a score and a result --");
var missing = PAST.filter(function (g) {
  return g.hsc === undefined || g.asc === undefined || g.hit === undefined ||
         g.atot === undefined;
}).length;
ck("incomplete past rows", missing, 0);
var totmismatch = PAST.filter(function (g) { return g.hsc + g.asc !== g.atot; }).length;
ck("rows where goals do not sum to the total", totmismatch, 0);

print("-- games under way are never also in the past list --");
// This is what makes counting the late finishers safe: if the build could
// put one game in both lists, every such game would be counted twice.
var pkey = {};
PAST.forEach(function (g) { pkey[g.d + "|" + g.a + g.h] = 1; });
ck("in-play games also listed as past",
   INPLAY.filter(function (g) { return pkey[g.d + "|" + g.a + g.h]; }).length, 0);

print("-- every finished in-play game carries a usable result --");
var lbad = 0;
INPLAY.forEach(function (g) {
  if (!isFinished(g)) return;
  if (g.hit === undefined || g.hit === null) { lbad += 1; return; }
  // a tie would make the hit flag meaningless; hockey has no ties
  if (g.hsc === g.asc) { lbad += 1; return; }
  var want = (g.ph > 0.5) === (g.hsc > g.asc) ? 1 : 0;
  if (g.hit !== want) lbad += 1;
});
ck("finished in-play rows with a wrong or missing result", lbad, 0);
ck("late finishers counted", LATE.length,
   INPLAY.filter(isFinished).length);

print(all ? "\nALL RECORD CHECKS PASS" : "\nFAILURES PRESENT");
