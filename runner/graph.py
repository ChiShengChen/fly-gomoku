"""The signed graph and the retinal projection — the only two things the gomoku scripts need
from the rest of FlyBench, lifted out verbatim so this repository stands on its own.

`load_graph` is `runner/run_assay.py::load_graph` and `retinal_sheet` is
`runner/run_on_channel.py::retinal_sheet` in the parent project; neither has been edited.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PUBLIC = RAW / "flywire_annotations_public.tsv"
VOXEL_NM = np.array([4.0, 4.0, 40.0])
GRID = 64

# D-001 sign convention; D-018 for the two the plan did not name
# the same transmitter is spelled differently in the two annotation tables
NT_ALIAS = {"ACETYLCHOLINE": "ACH", "GLUTAMATE": "GLUT", "DOPAMINE": "DA",
            "SEROTONIN": "SER", "OCTOPAMINE": "OCT", "UNKNOWN": "UNCLEAR"}
NT_SIGN = {"ACH": +1, "GABA": -1, "GLUT": -1, "HISTAMINE": -1,
           "DA": +1, "OCT": +1, "SER": +1, "UNCLEAR": 0}


def load_graph(dataset: str, min_syn: int, mod_sign: int, sign_from: str = "cell"):
    """Signed graph. `sign_from` picks where an edge's sign comes from.

    `"cell"` (default) takes the presynaptic neuron's `consensus_nt`, which enforces Dale's law
    and uses an estimate pooled over all of that cell's synapses. `"edge"` is the original
    behaviour, reading each connection's own `nt_type`.

    The difference is nothing on MaleCNS, where the per-edge label is already derived from the
    cell: 0 of 160,755 presynaptic neurons emit both signs. On FlyWire it is not nothing —
    **20.8 %** of presynaptic cells emit both signs under the per-edge rule, and for the
    photoreceptors it is simply wrong: all 8,452 R1-6 are histaminergic at the cell level, and the
    per-edge table labels their synapses ACH/GLUT/GABA with not one histamine among them, turning
    the inhibitory input that drives the whole visual system into an excitatory one (D-050)."""
    ann = pd.read_parquet(RAW / f"{dataset}_annotations.parquet")
    con = pd.read_parquet(RAW / f"{dataset}_connections.parquet")
    # One row per (pre, post, ROI), not per neuron pair -- 1.15 rows per pair on MaleCNS and 1.43
    # on FlyWire, up to 26 for one pair. Thresholding the rows therefore thresholds each brain
    # region separately: a pair with 3 synapses in each of three regions has 9 and was being
    # dropped entirely. At min_syn=5 that discarded 476,803 MaleCNS pairs (7.6 %) and 6.0 M
    # synapses (6.7 %). Aggregate first, then threshold (D-064).
    con = (con.groupby(["pre_id", "post_id"], as_index=False, sort=False)
              .agg(weight=("weight", "sum"),
                   nt_type=("nt_type", "first") if "nt_type" in con.columns
                   else ("weight", "size")))
    # A copy can already be filtered below what is asked for, in which case `min_syn` looks like a
    # knob and is not one: the FlyWire copy loaded here has a floor of 5 synapses, so --min-syn 1
    # changes nothing on it while adding 19 M edges to MaleCNS. Comparing the two below 5 would put
    # a filtered graph against an unfiltered one, so the floor is reported rather than assumed.
    floor = int(con["weight"].min()) if len(con) else 0
    if min_syn < floor:
        print(f"  note: {dataset} is already filtered at >= {floor} synapses; "
              f"min_syn={min_syn} cannot go below that and has no effect")
    con = con[con["weight"] >= min_syn]
    ids = ann["body_id"].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    pre = pos.reindex(con["pre_id"]).to_numpy()
    post = pos.reindex(con["post_id"]).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)

    def norm(v):
        v = pd.Series(v).fillna("").astype(str).str.upper()
        return v.replace(NT_ALIAS).to_numpy()

    if sign_from == "cell":
        cell_nt = norm(ann["consensus_nt"].to_numpy())
        nt = cell_nt[pre]
    elif sign_from == "edge":
        nt = norm(con["nt_type"].to_numpy()[ok])
    else:
        raise ValueError(f"sign_from must be 'cell' or 'edge', not {sign_from!r}")
    sign = np.array([mod_sign if NT_SIGN.get(x, 0) == +1 and x in ("DA", "OCT", "SER")
                     else NT_SIGN.get(x, 0) for x in nt], dtype=np.float32)
    w = con["weight"].to_numpy()[ok].astype(np.float32) * sign
    keep = w != 0
    return ann, ids, pre[keep], post[keep], w[keep]


def retinal_sheet(xyz: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Project cells onto the plane the reference population defines, scaled to [0,1]."""
    c = ref - ref.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    xy = (xyz - ref.mean(0)) @ vt[:2].T
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / np.where(hi - lo > 0, hi - lo, 1)
