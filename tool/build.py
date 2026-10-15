"""Assemble tool/index.html from its parts plus the exported data payload."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    head = open(os.path.join(HERE, "head.html")).read()
    body = open(os.path.join(HERE, "body.html")).read()
    app = open(os.path.join(HERE, "app.js")).read()
    data = open(os.path.join(ROOT, "out", "tool_data.json")).read()
    out = (head + body
           + '\n<script id="nhl-data" type="application/json">' + data
           + "</script>\n<script>\n" + app + "\n</script>\n")
    path = os.path.join(HERE, "index.html")
    open(path, "w").write(out)
    print(f"wrote {path}  {len(out) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
