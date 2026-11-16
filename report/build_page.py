#!/usr/bin/env python3
"""Inject the run's JSON into the front-end template and write the standalone page.

Kept as a build step rather than a hand-edited file so the page can never drift from the run that
produced it: re-run runner/run_gomoku.py, re-run this, and every number on the page moves
together.

    python report/build_page.py          # writes docs/index.html
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = "gomoku_template.html"
OUT = ROOT / "docs" / "index.html"
# (json, window variable) — each becomes a global the template reads
DATA = [("gomoku_flywire.json", "GOMOKU"), ("fly_parts.json", "FLY"),
        ("circuit_flywire.json", "CIRCUIT"), ("reward_flywire.json", "REWARD"),
        ("conditioning_flywire.json", "COND")]
# .js.part files inlined at the template's /*CODE*/ slots, in order
CODE = ["fly3d.js.part", "circuit.js.part"]


def main() -> int:
    tpl = (ROOT / "report" / TEMPLATE).read_text(encoding="utf-8")
    if "/*DATA*/" not in tpl:
        raise SystemExit(f"{TEMPLATE} has no /*DATA*/ placeholder")
    chunks = []
    for name, var in DATA:
        path = ROOT / "report" / name
        if not path.exists():
            raise SystemExit(f"{name} not generated yet")
        chunks.append(f"window.{var}=" + json.dumps(
            json.loads(path.read_text(encoding="utf-8")), separators=(",", ":")) + ";")
    out_txt = tpl.replace("/*DATA*/", "\n".join(chunks))
    for part in CODE:
        if "/*CODE*/" not in out_txt:
            raise SystemExit(f"{TEMPLATE}: more code parts than /*CODE*/ slots")
        # count=1: each part fills the next slot. Without it every slot gets the same part,
        # which inlines the first one twice and drops the rest.
        out_txt = out_txt.replace("/*CODE*/", (ROOT / "report" / part).read_text(encoding="utf-8"), 1)
    if "/*CODE*/" in out_txt:
        raise SystemExit(f"{TEMPLATE}: a /*CODE*/ slot was left unfilled")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(out_txt, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
