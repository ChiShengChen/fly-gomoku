#!/usr/bin/env python3
"""Build the two tables the runner reads, from public FlyWire v783 downloads.

The published run read these tables from a lab database export. This script rebuilds the same
schema from the public release so nobody needs that database:

    data/raw/flywire_annotations.parquet   one row per neuron: type, side, transmitter
    data/raw/flywire_connections.parquet   one row per (pre, post, neuropil): synapse count

Inputs, all v783:

  * `Supplemental_file1_neuron_annotations.tsv` — Schlegel et al. 2024. Downloaded automatically
    to data/raw/flywire_annotations_public.tsv.
  * `connections.csv.gz` and `neurons.csv.gz` from https://codex.flywire.ai/api/download
    (Dorkenwald et al. 2024). These need a login, so fetch them by hand into data/raw/.
  * optionally `classification.csv.gz` from the same page: 14 of the 139,255 neurons are missing
    from the Schlegel table, and this fills in their class and side.

    python data/prepare_flywire.py

The annotation mapping below is the one the published run used (transmitter and side vocabulary
included), and was checked against that run's table. The connection half has NOT been run against
a real Codex download — the column names are Codex's documented ones — so compare the row counts
it prints with data/README.md before trusting a result built on it.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"
URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
       "supplemental_files/Supplemental_file1_neuron_annotations.tsv")
PUBLIC = RAW / "flywire_annotations_public.tsv"

# FlyWire spells neurotransmitters out; the sign table in runner/graph.py uses short codes.
# Mapped explicitly rather than upper-casing, so an unrecognised value stays unrecognised (it
# resolves to sign 0) instead of being silently coerced into something plausible.
NT_MAP = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT",
          "dopamine": "DA", "serotonin": "SER", "octopamine": "OCT",
          "histamine": "HISTAMINE", "unknown": "UNCLEAR"}
NT_SHORT = {"ACH": "ACH", "GABA": "GABA", "GLUT": "GLUT", "DA": "DA", "SER": "SER", "OCT": "OCT"}
# Left unmapped this is silent rather than loud: a side-split readout finds zero cells.
SIDE_MAP = {"left": "L", "right": "R", "center": "M", "na": None}


def annotations(ids: pd.Series, pub: pd.DataFrame, fallback: pd.DataFrame | None) -> pd.DataFrame:
    """One row per id in `ids`, labelled from the Schlegel table, then from `fallback`."""
    m = pd.DataFrame({"body_id": ids.astype("int64")}).merge(
        pub, left_on="body_id", right_on="root_id", how="left")
    if fallback is not None:
        fb = fallback.set_index("root_id").reindex(m.body_id)
        fb.index = m.index
    else:
        fb = pd.DataFrame(index=m.index, columns=["cell_type", "super_class", "side"])
    return pd.DataFrame({
        "body_id": m.body_id,
        "type": m.cell_type.fillna(fb.cell_type),
        "hemibrain_type": m.hemibrain_type,
        "super_class": m.super_class.fillna(fb.super_class),
        "cell_class": m.cell_class,
        "soma_side": m.side.fillna(fb.side).str.lower().map(SIDE_MAP),
        # `known_nt` is free text and often lists co-transmitters and negative results, so it is
        # used only when it maps cleanly on its own; otherwise the single-valued `top_nt`.
        "consensus_nt": (m.known_nt.str.lower().map(NT_MAP)
                         .fillna(m.top_nt.str.lower().map(NT_MAP)).fillna("UNCLEAR")),
        "nt_raw": m.top_nt,
        "nerve": m.nerve,
        "flow": m.flow,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the annotation table")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    if args.refresh or not PUBLIC.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, PUBLIC)
    for need in ("neurons.csv.gz", "connections.csv.gz"):
        if not (RAW / need).exists():
            raise SystemExit(f"data/raw/{need} is missing — download it from "
                             "https://codex.flywire.ai/api/download (v783) and put it there")

    pub = pd.read_csv(PUBLIC, sep="\t", low_memory=False,
                      usecols=["root_id", "cell_type", "hemibrain_type", "super_class",
                               "cell_class", "side", "top_nt", "known_nt", "nerve", "flow"])
    ids = pd.read_csv(RAW / "neurons.csv.gz", usecols=["root_id"]).root_id
    cls = RAW / "classification.csv.gz"
    fallback = (pd.read_csv(cls, usecols=["root_id", "cell_type", "super_class", "side"])
                if cls.exists() else None)
    ann = annotations(ids, pub, fallback)
    matched = int(ann.body_id.isin(pub.root_id).sum())
    print(f"annotations: {len(ann):,} neurons, {matched:,} found in the Schlegel table")
    if matched / len(ann) < 0.99:
        raise SystemExit("root_id overlap below 99 % — this is a different materialisation, "
                         "and joining it would mislabel cells")
    stray = set(ann.soma_side.dropna()) - {"L", "R", "M"}
    if stray:
        raise SystemExit(f"soma_side did not map cleanly to L/R/M; stray values: {stray}")
    ann.to_parquet(RAW / "flywire_annotations.parquet", index=False)

    con = pd.read_csv(RAW / "connections.csv.gz",
                      usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"])
    con = pd.DataFrame({
        "pre_id": con.pre_root_id.astype("int64"),
        "post_id": con.post_root_id.astype("int64"),
        "weight": con.syn_count.astype("int32"),
        # only read under --sign-from edge; the gomoku run takes the sign from the cell
        "nt_type": con.nt_type.astype(str).str.upper().map(NT_SHORT).astype("category"),
    })
    con.to_parquet(RAW / "flywire_connections.parquet", index=False)
    print(f"connections: {len(con):,} rows, {int(con.weight.sum()):,} synapses, "
          f"smallest row {int(con.weight.min())} synapses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
