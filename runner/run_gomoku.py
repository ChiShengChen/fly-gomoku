#!/usr/bin/env python3
"""A fly plays gomoku, and a shuffled copy of it plays the same games.

The point is not that a connectome can play a board game. It is that the claim "a fly learned to
play X" is cheap to make and cheap to check, and the check is the same one this project uses
everywhere: run the identical pipeline through a shuffled connectome and see whether the real
wiring did anything.

Nothing is faked in between. The board is rendered as an image, the image reaches the
photoreceptors through the retinotopic map, the hybrid kernel propagates it (D-053), and the move
is read from **where the lobula is most active** — which is a real computation that circuit does,
since LC4 and LPLC2 are retinotopic and report where on the retina something salient is.

Receptive-field positions are derived from the wiring rather than assumed: a cell's retinal
coordinate is the weight-weighted mean of its presynaptic partners' coordinates, seeded at the
photoreceptors and propagated forward.

    python runner/run_gomoku.py --games 5

Writes report/gomoku_<dataset>.json — every board, every move, both arms.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for sub in ("model", "data", "controls", "runner"):
    sys.path.insert(0, str(ROOT / sub))
from lif_torch import LIFParams, TorchLIF                        # noqa: E402
from generators import degree_preserving_shuffle                 # noqa: E402
from graph import load_graph, retinal_sheet, VOXEL_NM, PUBLIC, GRID  # noqa: E402

N = 9                       # board size; 5 in a row wins
READOUT = ["LC4", "LPLC2"]  # retinotopic lobula columnar cells: where, not what
R_MAX, DRIVE_MV = 200.0, 24.0e-3
FRAMES, FRAME_MS = 8, 25.0


def wins(b, p):
    for y in range(N):
        for x in range(N):
            if b[y, x] != p:
                continue
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                n = 0
                while (0 <= y + dy * n < N and 0 <= x + dx * n < N
                       and b[y + dy * n, x + dx * n] == p):
                    n += 1
                if n >= 5:
                    return True
    return False


def board_image(b, size=GRID):
    """Stones as bright discs on a dark board. What the eye is shown, nothing more."""
    img = np.zeros((size, size), np.float32)
    cell = size / N
    yy, xx = np.mgrid[0:size, 0:size]
    for y in range(N):
        for x in range(N):
            if b[y, x] == 0:
                continue
            cy, cx = (y + .5) * cell, (x + .5) * cell
            m = (yy - cy) ** 2 + (xx - cx) ** 2 < (cell * .36) ** 2
            img[m] = 1.0 if b[y, x] == 1 else 0.45      # fly's stones bright, opponent's dim
    return img


def propagate_positions(pre, post, w, seed_idx, seed_uv, n, rounds=3):
    """Give every reachable cell a retinal coordinate: the weight-weighted mean of its inputs'.

    Seeded at the photoreceptors, which have one by construction, and pushed forward. This is how
    a receptive-field centre is estimated from wiring rather than assumed from a soma position —
    a lobula cell sits nowhere near the part of the retina it watches.
    """
    uv = np.full((n, 2), np.nan, np.float32)
    uv[seed_idx] = seed_uv
    aw = np.abs(w)
    for _ in range(rounds):
        known = np.isfinite(uv).all(1)
        m = known[pre]
        if not m.any():
            break
        num = np.zeros((n, 2), np.float64)
        den = np.zeros(n, np.float64)
        np.add.at(num, post[m], uv[pre[m]] * aw[m, None])
        np.add.at(den, post[m], aw[m])
        got = (den > 0) & ~known
        uv[got] = (num[got] / den[got, None]).astype(np.float32)
    return uv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flywire")
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--h2h", type=int, default=8,
                    help="real against shuffled, head to head, alternating who opens")
    ap.add_argument("--boards", type=int, default=40,
                    help="identical boards shown to both brains; one data point each")
    ap.add_argument("--side", default="R")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ann, ids, pre, post, w = load_graph(args.dataset, min_syn=5, mod_sign=1, sign_from="cell")
    ty = ann["type"].fillna("").astype(str).to_numpy()
    side = ann["soma_side"].astype(str).to_numpy()
    sup = ann["super_class"].fillna("").to_numpy()

    pub = pd.read_csv(PUBLIC, sep="\t", low_memory=False,
                      usecols=["root_id", "pos_x", "pos_y", "pos_z"])
    xyz = (pd.DataFrame({"body_id": ids}).merge(pub, left_on="body_id", right_on="root_id",
                                                how="left")
           [["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64) * VOXEL_NM)
    have = np.isfinite(xyz).all(1)
    xyz[~have] = np.nanmean(xyz[have], axis=0)     # the handful without coordinates are kept,
    print(f"{int((~have).sum())} of {len(ids):,} cells have no coordinate; "     # not dropped
          f"placed at the centroid rather than left NaN")

    photo = np.flatnonzero(np.isin(ty, ["R1-6", "R7", "R8"]))
    graded = np.union1d(np.flatnonzero(sup == "optic"), photo)
    drive = np.intersect1d(photo, np.flatnonzero(side == args.side))
    drive = drive[np.isfinite(xyz[drive]).all(1)]
    ref = xyz[(ty == "L2") & (side == args.side)]
    uv_drive = retinal_sheet(xyz[drive], ref[np.isfinite(ref).all(1)])

    uv = propagate_positions(pre, post, w, drive, uv_drive, len(ids))
    read_idx, read_uv = [], []
    for t in READOUT:
        ii = np.flatnonzero((ty == t) & (side == args.side))
        ok = ii[np.isfinite(uv[ii]).all(1)]
        read_idx.append(ok); read_uv.append(uv[ok])
        print(f"{t}: {len(ok)}/{len(ii)} on the {args.side} side got a receptive-field centre "
              f"from the wiring")
    rd = np.concatenate(read_idx); rduv = np.vstack(read_uv)
    # each readout cell votes for the board square its receptive-field centre falls in
    vote = (np.clip((rduv[:, 1] * N).astype(int), 0, N - 1) * N
            + np.clip((rduv[:, 0] * N).astype(int), 0, N - 1))
    cover = len(np.unique(vote))
    print(f"{len(rd)} readout cells cover {cover}/{N*N} board squares")

    params = LIFParams(t_run=FRAMES * FRAME_MS / 1000.0, r_max_hz=R_MAX)
    frame_steps = int(round(FRAME_MS / 1000.0 / params.dt))
    col = np.clip((uv_drive[:, 0] * (GRID - 1)).round().astype(int), 0, GRID - 1)
    row = np.clip(((1 - uv_drive[:, 1]) * (GRID - 1)).round().astype(int), 0, GRID - 1)

    def b64(a):
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")

    def fly_move(net, board, rng, cells=None):
        img = board_image(board)
        lum = np.repeat(img[row, col][None, :], FRAMES, axis=0)
        lum[0] = 0.0                                       # a step on, so there is a transient
        out = net.run(np.array([], dtype=np.int64), seed=int(rng.integers(1 << 30)),
                      frame_steps=frame_steps, graded_stim=drive,
                      graded_schedule=(DRIVE_MV * lum).astype(np.float32),
                      record_pop=[rd], record_cells=cells)
        _c = out[0]
        if cells is not None:
            # which of the sampled cells fired at all during this move -- sparse, because the
            # network is nearly silent and a dense frame would be 99 % zeros
            fired = cells[out[2].sum(axis=1) > 0]
            fly_move.last_active = fired.astype(np.uint32)
        # per-cell would be ideal; the population trace gives the timing, and the spatial vote
        # comes from which cells actually fired, so record per cell via counts
        counts = _c[rd].astype(np.float64)
        score = np.zeros(N * N)
        np.add.at(score, vote, counts)
        free = (board.reshape(-1) == 0)
        score[~free] = -1
        if score.max() <= 0:                               # silent circuit: say so, do not pretend
            legal = np.flatnonzero(free)
            return int(rng.choice(legal)), False, score
        return int(score.argmax()), True, score
    fly_move.last_active = np.array([], np.uint32)

    t0 = time.time()
    out = {"meta": {"dataset": args.dataset, "board": N, "games": args.games,
                    "readout": READOUT, "n_readout": int(len(rd)),
                    "squares_covered": int(cover), "frames": FRAMES,
                    "rf_uv": [[round(float(a), 4), round(float(b), 4)] for a, b in rduv],
                    "rf_vote": [int(v) for v in vote]},
           "games": []}

    # --- the measurement that does not need games -------------------------------------------
    # A game is one data point and mostly luck. Showing both brains the *same* board gives one
    # per board, and adds the question games cannot answer: does the circuit do anything the
    # image does not already say?
    def peak_luminance_move(board):
        img = board_image(board)
        cell = GRID / N
        sc = np.zeros(N * N)
        for y in range(N):
            for x in range(N):
                if board[y, x]:
                    continue
                blk = img[int(y*cell):int((y+1)*cell), int(x*cell):int((x+1)*cell)]
                sc[y * N + x] = blk.mean()
        return int(sc.argmax())

    brng = np.random.default_rng(args.seed + 7)
    nets = {k: TorchLIF(len(ids), *v, params, graded_idx=graded)
            for k, v in (("real", (pre, post, w)),
                         ("scrambled", degree_preserving_shuffle(pre, post, w, args.seed + 100)))}
    agree = 0
    lum_match = {"real": 0, "scrambled": 0}
    circuit = {"real": 0, "scrambled": 0}
    picks = {"real": [], "scrambled": [], "luminance": []}
    for i in range(args.boards):
        b = np.zeros((N, N), np.int8)
        k = brng.integers(6, 22)
        cells = brng.choice(N * N, k, replace=False)
        for j, c in enumerate(cells):
            b[c // N, c % N] = 1 + (j % 2)
        mv = {}
        for arm, net in nets.items():
            m, used, _sc = fly_move(net, b, brng)
            mv[arm] = m; circuit[arm] += int(used); picks[arm].append(m)
        lm = peak_luminance_move(b)
        picks["luminance"].append(lm)
        agree += int(mv["real"] == mv["scrambled"])
        for arm in nets:
            lum_match[arm] += int(mv[arm] == lm)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.boards} boards ({time.time()-t0:.0f}s)", flush=True)
    free_mean = float(np.mean([int((np.zeros(N*N)==0).sum())]))
    out["boards"] = {
        "n": args.boards, "real_equals_scrambled": agree,
        "chance_same_square": round(1.0 / max(cover, 1), 4),
        "moves_from_circuit": circuit,
        "equals_peak_luminance": lum_match,
        "picks": picks}
    print(f"\nsame board, both brains: they chose the same square "
          f"{agree}/{args.boards} times")
    for arm in nets:
        print(f"  {arm:10s} matched the plain peak-luminance heuristic "
              f"{lum_match[arm]}/{args.boards}, circuit decided {circuit[arm]}/{args.boards}")

    rng = np.random.default_rng(args.seed)
    arms = {"real": (pre, post, w),
            "scrambled": degree_preserving_shuffle(pre, post, w, args.seed + 100)}
    for arm, (p_, q_, w_) in arms.items():
        net = TorchLIF(len(ids), p_, q_, w_, params, graded_idx=graded)
        won = drew = 0
        decided = silent = 0
        for g in range(args.games):
            b = np.zeros((N, N), np.int8)
            moves, scores, turn = [], [], 1
            gr = np.random.default_rng(1000 + g)           # the opponent is the same both arms
            while (b == 0).any():
                if turn == 1:
                    mv, real_choice, sc = fly_move(net, b, rng)
                    scores.append([round(float(v), 2) for v in sc])
                    decided += int(real_choice); silent += int(not real_choice)
                else:
                    mv = int(gr.choice(np.flatnonzero(b.reshape(-1) == 0)))
                b[mv // N, mv % N] = turn
                moves.append(int(mv))
                if wins(b, turn):
                    break
                turn = 3 - turn
            res = "fly" if wins(b, 1) else ("random" if wins(b, 2) else "draw")
            won += res == "fly"; drew += res == "draw"
            out["games"].append({"arm": arm, "game": g, "moves": moves,
                                 "scores": scores, "result": res})
            print(f"  {arm:10s} game {g}: {res:6s} in {len(moves)} moves "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out.setdefault("summary", {})[arm] = {
            "wins": won, "draws": drew, "losses": args.games - won - drew,
            "moves_from_circuit": decided, "moves_random_because_silent": silent}
        print(f"{arm}: {won}/{args.games} wins, {silent} moves fell back to random "
              f"because the circuit was silent")

    # --- head to head -------------------------------------------------------------------------
    # Both arms beating a random opponent says little; playing each other is the direct question,
    # and it is the one a "two flies at a board" picture actually depicts.
    if args.h2h:
        nr = TorchLIF(len(ids), pre, post, w, params, graded_idx=graded)
        ns = TorchLIF(len(ids), *degree_preserving_shuffle(pre, post, w, args.seed + 100),
                      params, graded_idx=graded)
        # a display subsample of the whole brain: every cell of the readout populations, plus a
        # fixed sample of the rest, so the cloud is a brain rather than a handful of bright dots
        srng = np.random.default_rng(args.seed + 31)
        keep = np.union1d(rd, srng.choice(len(ids), min(28000, len(ids)), replace=False))
        keep = np.sort(keep).astype(np.int64)
        pos_lo, pos_hi = xyz.min(0), xyz.max(0)
        pspan = float((pos_hi - pos_lo).max())
        q = (((xyz[keep] - (pos_lo + pos_hi) / 2) / pspan + 0.5) * 65535)
        out["brain"] = {
            "n": int(len(keep)),
            "pos": b64(q.clip(0, 65535).astype(np.uint16)),
            "readout": b64(np.searchsorted(keep, rd).astype(np.uint32)),
            "extent_um": [int(v / 1000) for v in (pos_hi - pos_lo)],
        }
        print(f"brain cloud: {len(keep):,} of {len(ids):,} cells "
              f"(all {len(rd)} readouts plus a sample)")

        h2h = []
        score = {"real": 0, "scrambled": 0, "draw": 0}
        for g in range(args.h2h):
            b = np.zeros((N, N), np.int8)
            hr = np.random.default_rng(5000 + g)
            moves, scores, whose, fell, act = [], [], [], [], []
            turn = 1 if g % 2 == 0 else 2          # alternate who opens, so first-move advantage
            while (b == 0).any():                  # cannot be mistaken for a wiring effect
                net = nr if turn == 1 else ns
                m, used, sc = fly_move(net, b, hr, cells=keep)
                act.append(b64(np.searchsorted(keep, fly_move.last_active).astype(np.uint32)))
                b[m // N, m % N] = turn
                moves.append(int(m)); whose.append(int(turn)); fell.append(int(not used))
                scores.append([round(float(v), 2) for v in sc])
                if wins(b, turn):
                    break
                turn = 3 - turn
            res = ("real" if wins(b, 1) else "scrambled" if wins(b, 2) else "draw")
            score[res] += 1
            h2h.append({"game": g, "moves": moves, "whose": whose, "scores": scores,
                        "fell_back": fell, "active": act, "result": res,
                        "opened": "real" if g % 2 == 0 else "scrambled"})
            print(f"  head-to-head game {g}: {res} won in {len(moves)} moves "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out["h2h"] = h2h
        out["h2h_score"] = score
        print(f"head to head over {args.h2h} games: real {score['real']}, "
              f"scrambled {score['scrambled']}, draws {score['draw']}")

    dest = ROOT / "report" / f"gomoku_{args.dataset}.json"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {dest.name} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
