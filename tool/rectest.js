var fs = read("/tmp/payload.json");
var D = JSON.parse(fs);
var T = JSON.parse(read("/tmp/truth.json"));
var PAST = D.past, RECORD = D.record;

// same functions the page uses
function dayRecord(d) {
  var w = 0, l = 0;
  PAST.forEach(function (g) { if (g.d !== d) return; if (g.hit) w += 1; else l += 1; });
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
  ck(d, r.w + "-" + r.l, T.byday[d][0] + "-" + T.byday[d][1]);
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

print(all ? "\nALL RECORD CHECKS PASS" : "\nFAILURES PRESENT");
