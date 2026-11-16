#!/usr/bin/env python3
"""The type-to-type wiring the demos actually run through, measured rather than drawn.

A schematic of a circuit is an assertion. This reads the signed synapse totals between the cell
types on the visual pathway straight out of the locked connectome, so every arrow on the page has
a number behind it and the feedback loops are either there or they are not.

    python report/circuit_matrix.py --dataset flywire

Writes report/circuit_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("model", "data", "controls", "runner"):
    sys.path.insert(0, str(ROOT / sub))
from graph import load_graph                                  # noqa: E402

# the pathway the pages use, in order, plus the partners that close loops onto it
LAYERS = [
    ("retina", ["R1-6", "R7", "R8"]),
    ("lamina", ["L1", "L2", "L3", "L4", "L5"]),
    ("medulla", ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm9", "Tm20", "TmY3", "C2", "C3"]),
    ("lobula", ["T2", "T4a", "T4b", "LC4", "LPLC2", "LC6"]),
    ("descending", ["DNp01", "DNa02"]),
]
MIN_SYN = 400          # an arrow below this is noise at this grain, and saying so is the point


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flywire")
    args = ap.parse_args()

    ann, ids, pre, post, w = load_graph(args.dataset, min_syn=5, mod_sign=1, sign_from="cell")
    ty = ann["type"].fillna("").astype(str).to_numpy()
    names = [t for _, ts in LAYERS for t in ts]
    idx = {t: i for i, t in enumerate(names)}
    slot = np.full(len(ids), -1, np.int64)
    counts = {}
    for t in names:
        m = ty == t
        slot[m] = idx[t]
        counts[t] = int(m.sum())

    a, b = slot[pre], slot[post]
    keep = (a >= 0) & (b >= 0)
    M = np.zeros((len(names), len(names)))
    np.add.at(M, (a[keep], b[keep]), w[keep])

    edges = []
    for i, s in enumerate(names):
        for j, d in enumerate(names):
            if i == j or abs(M[i, j]) < MIN_SYN:
                continue
            edges.append({"from": s, "to": d, "syn": int(round(M[i, j]))})
    present = [t for t in names if counts[t] > 0]
    print(f"{len(present)}/{len(names)} types present, {len(edges)} edges above {MIN_SYN} synapses")

    # loops back onto the lamina are the ones D-057 named as the candidate compressor
    lam = set(LAYERS[1][1])
    loops = []
    for e in edges:
        if e["to"] in lam and e["from"] not in lam and e["from"] not in set(LAYERS[0][1]):
            back = [f for f in edges if f["from"] == e["to"] and f["to"] == e["from"]]
            loops.append({**e, "forward_syn": back[0]["syn"] if back else 0})
    loops.sort(key=lambda e: -abs(e["syn"]))
    print("\nfeedback onto the lamina (the D-057 candidate):")
    for e in loops[:8]:
        sign = "+" if e["syn"] > 0 else "−"
        print(f"  {e['from']:6s} -> {e['to']:4s} {sign}{abs(e['syn']):7,}   "
              f"and {e['to']} -> {e['from']} {e['forward_syn']:+8,}")

    doc = {"dataset": args.dataset, "min_syn": MIN_SYN,
           "layers": [{"name": n, "types": [t for t in ts if counts[t] > 0]} for n, ts in LAYERS],
           "cells": {t: counts[t] for t in present},
           "edges": [e for e in edges if counts[e["from"]] and counts[e["to"]]],
           "loops": loops[:10]}
    dest = ROOT / "report" / f"circuit_{args.dataset}.json"
    dest.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
