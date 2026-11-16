"""Does the gomoku pipeline's activity ever reach the reward machinery?

D-048 measured the mushroom body at roughly five synapses from the eye and D-057 measured the
signal losing about fivefold per synapse, which predicts that nothing survives the trip. This
checks the prediction against the recorded runs instead of arguing it: for every move of every
head-to-head game, how many Kenyon cells, MBONs and dopaminergic neurons actually fired.

The subsample is reproduced from the runner's seed rather than stored, so this cannot drift away
from the cells the simulation really recorded.

    python report/reward_reach.py            # prints the table, writes report/reward_flywire.json
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("model", "data", "controls", "runner"):
    sys.path.insert(0, str(ROOT / sub))

from graph import load_graph  # noqa: E402

SEED = 1          # must match runner/run_gomoku.py --seed
SUBSAMPLE = 28000  # must match its --cloud-cells
READOUT = ("LC4", "LPLC2")

# Ordered from the reward machinery outwards, so the table reads as a distance from the eye.
POPULATIONS = [
    ("KC", "Kenyon cells", "prefix"),
    ("MBON", "mushroom body output", "prefix"),
    ("PAM", "dopamine, reward", "prefix"),
    ("PPL1", "dopamine, punishment", "prefix"),
    ("LC4|LPLC2", "the readout itself", "exact"),
    # Kept deliberately: under D-053 the optic lobe is graded, so these units never spike and are
    # invisible to this channel. Reporting the resulting zero as silence would be a false negative,
    # so the row is flagged rather than dropped — dropping it invites someone to re-derive it.
    ("Mi1|Tm1", "medulla — graded, never spikes", "exact"),
]


def main() -> int:
    ann, ids, pre, post, _ = load_graph("flywire", min_syn=5, mod_sign=1, sign_from="cell")
    # dtype=str, not object: np.char.startswith has no loop for object arrays.
    ty = ann["type"].fillna("").astype(str).to_numpy(dtype=str)
    side = ann["soma_side"].astype(str).to_numpy(dtype=str)
    sup = ann["super_class"].astype(str).to_numpy(dtype=str)

    # The same graded set the runner builds. A graded unit never crosses threshold, so it can never
    # appear in a spike record: a zero for such a population means "not measured here", not "quiet".
    is_graded = np.zeros(len(ty), bool)
    is_graded[np.union1d(np.flatnonzero(sup == "optic"),
                         np.flatnonzero(np.isin(ty, ("R1-6", "R7", "R8"))))] = True

    # Reproduce exactly the cells run_gomoku kept: the readout, plus a seeded random subsample.
    readout = np.concatenate([np.flatnonzero((ty == t) & (side == "R")) for t in READOUT])
    rng = np.random.default_rng(SEED + 31)
    keep = np.sort(np.union1d(
        readout, rng.choice(len(ids), min(SUBSAMPLE, len(ids)), replace=False)))

    rows = []
    for pat, label, kind in POPULATIONS:
        names = pat.split("|")
        if kind == "prefix":
            mask = np.zeros(len(ty), bool)
            for n in names:
                mask |= np.char.startswith(ty, n)
        else:
            mask = np.isin(ty, names)
        idx = np.flatnonzero(mask)
        sampled = idx[np.isin(idx, keep)]
        rows.append({
            "name": pat.replace("|", " + "), "note": label,
            "in_volume": int(mask.sum()),
            "sampled": int(len(sampled)),
            "graded": bool(len(idx) and is_graded[idx].all()),
            "pos": set(np.searchsorted(keep, sampled).tolist()),
        })

    games = json.loads((ROOT / "report/gomoku_flywire.json").read_text())["h2h"]
    moves = {1: 0, 2: 0}
    for g in games:
        for i, who in enumerate(g["whose"]):
            fired = set(np.frombuffer(base64.b64decode(g["active"][i]), np.uint32).tolist())
            moves[who] += 1
            for r in rows:
                r.setdefault("hit", {1: 0, 2: 0})[who] += len(fired & r["pos"])

    for r in rows:
        # None, not 0.0: a graded population is unmeasurable in a spike record, and the page must
        # not be able to render that as a zero.
        r["real"] = None if r["graded"] else round(r["hit"][1] / max(moves[1], 1), 2)
        r["scrambled"] = None if r["graded"] else round(r["hit"][2] / max(moves[2], 1), 2)
        del r["pos"], r["hit"]

    out = {"moves_real": moves[1], "moves_scrambled": moves[2], "rows": rows}
    (ROOT / "report/reward_flywire.json").write_text(json.dumps(out, indent=1))

    print(f"{'population':22s} {'note':32s} {'in vol':>8s} {'sampled':>8s} "
          f"{'real':>7s} {'shuffled':>9s}")
    for r in rows:
        fmt = (lambda v: "    n/a" if v is None else f"{v:7.2f}")
        print(f"{r['name']:22s} {r['note']:32s} {r['in_volume']:8,} {r['sampled']:8,} "
              f"{fmt(r['real'])} {fmt(r['scrambled']):>9s}")
    print(f"\ncells firing per move, over {moves[1]} real and {moves[2]} shuffled moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
