"""Cache player names so predictions can show the probable goalie."""
import json
import os

import requests

CACHE = "data/player_names.json"
API = "https://api-web.nhle.com/v1/player/{}/landing"


def load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def resolve(ids):
    cache = load()
    missing = [str(int(i)) for i in ids if str(int(i)) not in cache]
    s = requests.Session()
    s.headers["User-Agent"] = "nhl-predictor/1.0"
    for pid in missing:
        try:
            d = s.get(API.format(pid), timeout=20).json()
            cache[pid] = (f"{d.get('firstName',{}).get('default','')} "
                          f"{d.get('lastName',{}).get('default','')}").strip()
        except Exception:
            cache[pid] = f"#{pid}"
    if missing:
        json.dump(cache, open(CACHE, "w"))
    return cache
