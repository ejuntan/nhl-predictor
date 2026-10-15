#!/bin/bash
# Rebuild the page from fresh data, then push it so Render redeploys.
#
# Render deploys from git, and the model's data cannot live in git, so the
# built page is what gets committed: ./daily.sh produces it here, this
# commits the two files that matter and pushes.
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "--no-refresh" ]; then
  ./daily.sh
fi

# Only the deploy artifacts. Anything else staged is the author's business.
git add -f tool/index.html out/tool_data.json
if git diff --cached --quiet; then
  echo "page unchanged; nothing to deploy"
  exit 0
fi

REC=$(./.venv/bin/python - <<'PY'
import json
d = json.load(open("out/tool_data.json"))
r = d.get("record", {})
print(f"{d.get('generated','?')}: {r.get('w',0)}-{r.get('l',0)}, "
      f"{len(d.get('games',[]))} upcoming, {len(d.get('lines',{}))} lines")
PY
)
git commit -q -m "page: $REC"
echo "committed — $REC"

if git remote get-url origin >/dev/null 2>&1; then
  git push -q origin HEAD
  echo "pushed; Render will redeploy within a minute or two"
else
  echo "no git remote named 'origin' — add one and push to deploy:"
  echo "  git remote add origin git@github.com:<you>/nhl-predictor.git"
  echo "  git push -u origin main"
fi
