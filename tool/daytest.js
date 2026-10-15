var WD=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
var MO=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function parseDay(d){return new Date(d+"T12:00:00Z");}
function dayShort(d){var t=parseDay(d);return {wd:WD[t.getUTCDay()],dd:t.getUTCDate(),mo:MO[t.getUTCMonth()]};}
function dayLong(d){var s=dayShort(d);var full=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][parseDay(d).getUTCDay()];return full+" "+s.dd+" "+s.mo;}
function leanWord(p){var m=Math.max(p,1-p);
  if(m<0.54)return "near a coin flip";
  if(m<0.60)return "a slight lean";
  if(m<0.67)return "a clear favourite";
  return "a strong favourite";}
var all=true;
function ck(l,g,w){var ok=g===w;print((ok?"PASS  ":"FAIL  ")+l+"  got="+g+"  want="+w);return ok;}
print("-- dates (cross-checked against python above) --");
all&=ck("2026-10-14 short","("+dayShort("2026-10-14").wd+" "+dayShort("2026-10-14").dd+" "+dayShort("2026-10-14").mo+")","(Wed 14 Oct)");
all&=ck("2026-10-14 long",dayLong("2026-10-14"),"Wednesday 14 Oct");
all&=ck("2026-10-15 long",dayLong("2026-10-15"),"Thursday 15 Oct");
all&=ck("2026-10-17 long",dayLong("2026-10-17"),"Saturday 17 Oct");
all&=ck("2026-10-21 long",dayLong("2026-10-21"),"Wednesday 21 Oct");
print("-- month rollover and leap year --");
all&=ck("2026-11-01",dayLong("2026-11-01"),"Sunday 1 Nov");
all&=ck("2026-12-31",dayLong("2026-12-31"),"Thursday 31 Dec");
all&=ck("2028-02-29",dayLong("2028-02-29"),"Tuesday 29 Feb");
print("-- lean wording is symmetric around 50% --");
all&=ck("p=0.50",leanWord(0.50),"near a coin flip");
all&=ck("p=0.535",leanWord(0.535),"near a coin flip");
all&=ck("p=0.46 (same as .54)",leanWord(0.46),leanWord(0.54));
all&=ck("p=0.57",leanWord(0.57),"a slight lean");
all&=ck("p=0.43",leanWord(0.43),"a slight lean");
all&=ck("p=0.65",leanWord(0.65),"a clear favourite");
all&=ck("p=0.35",leanWord(0.35),"a clear favourite");
all&=ck("p=0.73",leanWord(0.73),"a strong favourite");
all&=ck("p=0.27",leanWord(0.27),"a strong favourite");
print(all?"\nALL PASS":"\nFAILURES");
