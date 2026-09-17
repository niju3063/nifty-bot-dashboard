"""
Dashboard Builder (final step)
==================================

Stitches dashboard_template.html + dashboard_data.json + paper_trading_data.json
into the finished nifty-edge-terminal.html.

RUN ORDER (from the same folder as all the other .py files):
    python3 generate_dashboard_data.py
    python3 generate_paper_trading_data.py
    python3 build_dashboard.py

Then open nifty-edge-terminal.html directly in a browser, or publish it
wherever you're hosting it (e.g. re-upload to Claude and ask it to
publish the artifact again).

Must sit alongside: dashboard_template.html, dashboard_data.json,
paper_trading_data.json (the first two Python scripts produce the
latter two).
"""

import json
from pathlib import Path

TEMPLATE_PATH = "dashboard_template.html"
DATA_PATH = "dashboard_data.json"
PAPER_PATH = "paper_trading_data.json"
CONTEXT_PATH = "market_context_data.json"
OUTPUT_PATH = "nifty-edge-terminal.html"


def main():
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    with open(DATA_PATH) as f:
        data_json = f.read()
    with open(PAPER_PATH) as f:
        paper_json = f.read()

    # market_context_data.json is optional -- the dashboard still works
    # without it (the panel just shows no data), since it needs its own
    # separate NSE fetch and you may not have run it yet.
    if Path(CONTEXT_PATH).exists():
        with open(CONTEXT_PATH) as f:
            context_json = f.read()
    else:
        context_json = "{}"
        print(f"Note: {CONTEXT_PATH} not found -- Market Context panel will be empty. "
              f"Run generate_market_context_data.py to populate it.")

    # Sanity-check all three are valid JSON before embedding -- a broken
    # embed silently breaks the whole page, so fail loudly here instead.
    json.loads(data_json)
    json.loads(paper_json)
    json.loads(context_json)

    html = (template
            .replace("__DATA_JSON__", data_json)
            .replace("__PAPER_JSON__", paper_json)
            .replace("__CONTEXT_JSON__", context_json))

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
