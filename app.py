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

  /          the page
  /data.json the payload the page is built from
  /healthz   when the data was generated, and how stale the prices are
"""
import json
import os
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, send_file

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


@app.route("/")
def index():
    if not os.path.exists(PAGE):
        return Response(
            "The page has not been built. Run ./daily.sh on the machine "
            "with the database, then commit tool/index.html.",
            status=503, mimetype="text/plain")
    return send_file(PAGE)


@app.route("/data.json")
def data():
    if not os.path.exists(PAYLOAD):
        return jsonify({"error": "no payload built"}), 503
    return send_file(PAYLOAD, mimetype="application/json")


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
        "model": "nhl-v2-20261014 (frozen)",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)),
            debug=False)
