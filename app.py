"""Serve the built predictions page.

Deliberately dumb. Everything expensive -- five seasons of play-by-play, the
shift charts, the walk-forward backtest -- happens on the machine that has
the 144 MB SQLite database, and the result is a single 550 KB HTML file.
This serves that file.

The alternative was a Render service that rebuilds the model itself, like
the MLB apps do. It does not work here: those compute from a free API at
request time with no local store, whereas this needs 679,000 shot records
and a three-minute backtest, and the database cannot go in git at all. So
the page is the build artifact and this is the web server for it.

The prices are the exception. They move all day and need nothing local, so
`liveserve` refetches them on a timer and the served page carries current
numbers between pushes -- the model's own prices do not change during a day,
because the ratings behind them only move when games finish.

  /          the page, with the freshest prices available
  /data.json the payload behind it
  /healthz   what is live, what shipped with the build, and how old each is
"""
import json
import os
import re
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, send_file

import liveserve

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "tool", "index.html")
PAYLOAD = os.path.join(ROOT, "out", "tool_data.json")

app = Flask(__name__)


def _payload():
    with open(PAYLOAD) as f:
        return json.load(f)


@app.after_request
def headers(r):
    # The page is rebuilt once a day, so a short cache is right: long enough
    # to absorb a reload, short enough that the morning refresh shows up.
    if r.status_code == 200 and r.mimetype in ("text/html", "application/json"):
        r.headers["Cache-Control"] = "public, max-age=600"
    return r


# The page ships with its data inlined, which is what makes it a single
# self-contained file. Swapping that block on the way out keeps it that way
# while serving current prices; nothing in the page has to know.
DATA_BLOCK = re.compile(
    r'(<script id="nhl-data" type="application/json">)(.*?)(</script>)',
    re.S)


@app.route("/")
def index():
    if not os.path.exists(PAGE):
        return Response(
            "The page has not been built. Run ./daily.sh on the machine "
            "with the database, then commit tool/index.html.",
            status=503, mimetype="text/plain")
    with open(PAGE) as f:
        html = f.read()
    p = liveserve.payload()
    if p is not None:
        js = json.dumps(p, separators=(",", ":")).replace("</", "<\\/")
        html = DATA_BLOCK.sub(
            lambda m: m.group(1) + js + m.group(3), html, count=1)
    return Response(html, mimetype="text/html")


@app.route("/data.json")
def data():
    p = liveserve.payload()
    if p is None:
        return jsonify({"error": "no payload built"}), 503
    return Response(json.dumps(p, separators=(",", ":")),
                    mimetype="application/json")


@app.route("/healthz")
def healthz():
    """Freshness, without having to open the page and squint at it."""
    if not os.path.exists(PAYLOAD):
        return jsonify({"ok": False, "error": "no payload built"}), 503
    d = _payload()
    built = datetime.fromtimestamp(os.path.getmtime(PAYLOAD), timezone.utc)
    age_h = (datetime.now(timezone.utc) - built).total_seconds() / 3600
    return jsonify({
        "ok": True,
        "generated": d.get("generated"),
        "slate_starts": d.get("asof"),
        "payload_built_utc": built.isoformat(timespec="seconds"),
        "payload_age_hours": round(age_h, 1),
        # A page more than ~36 hours old is showing yesterday's slate.
        "stale": age_h > 36,
        "upcoming_games": len(d.get("games", [])),
        "posted_lines": len(d.get("lines", {})),
        "prices_fetched": d.get("lineAt"),
        "past_games": len(d.get("past", [])),
        "record": d.get("record"),
        "model": "nhl-v2-20261015 (frozen)",
        # What is live on this host, and what shipped with the build.
        "live": liveserve.status(),
        "static": ["model prices", "finished games", "shots and saves props"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)),
            debug=False)
