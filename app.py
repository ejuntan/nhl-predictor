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

Both the page and the payload are sent with `no-cache` and an ETag, so a
reload always asks the server and gets whatever is current -- costing a
304 and no transfer when nothing has moved. Add `?refresh=1` to skip the
score timer for that request, or `?refresh=all` to refetch the prices too,
which takes about half a minute.
"""
import json
import os
import re
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_file

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
    # This page exists to show what is happening now, so a reload has to
    # reach the server. It used to say max-age=600 with no validator, which
    # let a browser answer a reload out of its own cache for ten minutes
    # without asking -- on top of the server's own score timer, so a reload
    # could show numbers a quarter of an hour old.
    #
    # `no-cache` does not mean "do not store": it means revalidate before
    # reusing. Paired with the ETag set on each route, an unchanged page
    # comes back as a 304 and costs nothing to transfer, while a changed
    # one is always delivered.
    if r.status_code in (200, 304) and r.mimetype in (
            "text/html", "application/json", None):
        r.headers["Cache-Control"] = "no-cache, must-revalidate"
    return r


# The page ships with its data inlined, which is what makes it a single
# self-contained file. Swapping that block on the way out keeps it that way
# while serving current prices; nothing in the page has to know.
DATA_BLOCK = re.compile(
    r'(<script id="nhl-data" type="application/json">)(.*?)(</script>)',
    re.S)


def _force():
    """What `?refresh=` asks for, as keyword arguments for the refresher.

    The timers exist so a burst of reloads does not become a burst of calls
    to someone else's API. `?refresh=1` steps past the score timer only --
    one call per date, about a second. `?refresh=all` also refetches the
    prices, which costs a lookup per game and takes half a minute, so it is
    not what a reload should do.
    """
    v = request.args.get("refresh")
    if v == "all":
        return {"force": True}
    return {"force_scores": v in ("1", "true", "yes")}


@app.route("/")
def index():
    if not os.path.exists(PAGE):
        return Response(
            "The page has not been built. Run ./daily.sh on the machine "
            "with the database, then commit tool/index.html.",
            status=503, mimetype="text/plain")
    with open(PAGE) as f:
        html = f.read()
    p = liveserve.payload(**_force())
    if p is not None:
        js = json.dumps(p, separators=(",", ":")).replace("</", "<\\/")
        html = DATA_BLOCK.sub(
            lambda m: m.group(1) + js + m.group(3), html, count=1)
    r = Response(html, mimetype="text/html")
    # The tag is over the finished page, data and all, so it changes exactly
    # when something a reader would see changes.
    r.add_etag()
    return r.make_conditional(request)


@app.route("/data.json")
def data():
    p = liveserve.payload(**_force())
    if p is None:
        return jsonify({"error": "no payload built"}), 503
    r = Response(json.dumps(p, separators=(",", ":")),
                 mimetype="application/json")
    r.add_etag()
    return r.make_conditional(request)


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
