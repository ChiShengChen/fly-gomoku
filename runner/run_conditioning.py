"""Appetitive conditioning in the mushroom body, under the pre-registered rule of D-073.

An odour is a combination of glomeruli, delivered by driving the olfactory receptor neurons of
those types. The unconditioned stimulus is delivered by driving the PAM cells directly, which is
what optogenetic substitution for sugar does at the bench. Between trials, KC→MBON synapses are
depressed wherever Kenyon-cell activity coincided with dopamine in the same compartment.

Three arms, all of which run whether or not the first one shows anything:

    paired      CS+ presented with the US
    unpaired    identical odours and identical US, never at the same time
    shuffled    the same protocol on a degree-preserving shuffle of KC→MBON only

    python runner/run_conditioning.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("model", "data", "controls", "runner"):
    sys.path.insert(0, str(ROOT / sub))

from generators import degree_preserving_shuffle  # noqa: E402
from lif_torch import LIFParams, TorchLIF  # noqa: E402
from plasticity import KCOutputDepression  # noqa: E402
from graph import load_graph  # noqa: E402

T_RUN = 0.3        # s per trial
N_GLOM = 12        # glomeruli per odour
N_TRAIN = 6        # paired training trials
ETA = 1.0          # depression rate, D-073
KC_R_REF = 50.0    # Hz, the rate at which a Kenyon cell counts as fully active


def rates(counts, t):
    c = counts.cpu().numpy() if hasattr(counts, "cpu") else np.asarray(counts)
    return c.astype(np.float64) / t


def shuffle_subset(pre, post, w, edge, seed):
    """Degree-preserving shuffle of one edge subset, the rest of the brain untouched."""
    p2, q2, w2 = pre.copy(), post.copy(), w.copy()
    sp, sq, sw = degree_preserving_shuffle(pre[edge], post[edge], w[edge], seed)
    # The shuffle may drop self-loops, so it can return fewer edges than it was given; the
    # remainder are dropped rather than back-filled, which keeps the null conservative.
    keep = np.ones(len(pre), bool)
    keep[edge] = False
    return (np.concatenate([p2[keep], sp]), np.concatenate([q2[keep], sq]),
            np.concatenate([w2[keep], sw]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flywire")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--min-syn", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    ann, ids, pre, post, w = load_graph(args.dataset, min_syn=args.min_syn, mod_sign=1,
                                        sign_from="cell")
    ty = ann["type"].fillna("").astype(str).to_numpy(dtype=str)
    cc = ann["cell_class"].fillna("").astype(str).to_numpy(dtype=str)
    orn = np.flatnonzero(cc == "olfactory")
    kc = np.flatnonzero(cc == "Kenyon_Cell")
    mbon = np.flatnonzero(np.char.startswith(ty, "MBON"))
    pam = np.flatnonzero(np.char.startswith(ty, "PAM"))
    dan = np.flatnonzero(cc == "DAN")
    glom = np.unique(ty[orn])
    glom = glom[glom != ""]
    params = LIFParams(t_run=T_RUN)

    print(f"ORN {len(orn)} in {len(glom)} glomeruli | KC {len(kc)} | MBON {len(mbon)} | "
          f"PAM {len(pam)} | DAN {len(dan)}")

    def trial(net, stim, seed):
        r = rates(net.run(np.asarray(stim, np.int64), seed=seed), T_RUN)
        return r, float(r[mbon].sum())

    def one(arm, seed):
        """One animal: naive test, training, test. Returns the two fractional drops."""
        rng = np.random.default_rng(seed)
        g = rng.choice(glom, 2 * N_GLOM, replace=False)
        cs_p = orn[np.isin(ty[orn], g[:N_GLOM])]
        cs_m = orn[np.isin(ty[orn], g[N_GLOM:])]

        p2, q2, w2 = pre, post, w
        if arm == "shuffled":
            e = np.flatnonzero(np.isin(pre, kc) & np.isin(post, mbon))
            p2, q2, w2 = shuffle_subset(pre, post, w, e, seed + 900)
        plast = KCOutputDepression(p2, q2, w2, kc, mbon, dan, eta=ETA, r_max_hz=KC_R_REF)

        net = TorchLIF(len(ids), p2, q2, plast.w.astype(np.float32), params)
        # one reward-alone trial fixes what "a full reward's worth of dopamine" means, so an
        # unrewarded trial's residual DAN firing stays small instead of renormalising to 1
        r_ref, _ = trial(net, pam, seed + 500)
        plast.calibrate(r_ref)
        (_, base_p), (_, base_m) = trial(net, cs_p, seed), trial(net, cs_m, seed + 1)

        removed = 0.0
        for t in range(N_TRAIN):
            # paired: odour and US together. unpaired: the same two events, never coincident.
            if arm == "unpaired":
                r_odour, _ = trial(net, cs_p, seed + 10 * t + 2)
                r_us, _ = trial(net, pam, seed + 10 * t + 3)
                # the rule sees each event's own dopamine, which is the whole point of the control
                removed = plast.step(r_odour)
                removed = plast.step(r_us)
            else:
                r, _ = trial(net, np.union1d(cs_p, pam), seed + 10 * t + 2)
                removed = plast.step(r)
            r_m, _ = trial(net, cs_m, seed + 10 * t + 4)      # CS- always unreinforced
            removed = plast.step(r_m)
            net = TorchLIF(len(ids), p2, q2, plast.w.astype(np.float32), params)

        (_, end_p), (_, end_m) = trial(net, cs_p, seed), trial(net, cs_m, seed + 1)
        d_p = (base_p - end_p) / base_p if base_p > 0 else 0.0
        d_m = (base_m - end_m) / base_m if base_m > 0 else 0.0
        return {"d_plus": d_p, "d_minus": d_m, "removed": removed,
                "base_plus": base_p, "base_minus": base_m,
                "end_plus": end_p, "end_minus": end_m}

    out = {"meta": {"dataset": args.dataset, "seeds": args.seeds, "t_run": T_RUN,
                    "n_train": N_TRAIN, "eta": ETA, "n_glom": N_GLOM,
                    "n_kc": int(len(kc)), "n_mbon": int(len(mbon)), "n_orn": int(len(orn))},
           "arms": {}}
    for arm in ("paired", "unpaired", "shuffled"):
        runs = []
        for s in range(args.seeds):
            r = one(arm, 1000 + s)
            runs.append(r)
            print(f"  {arm:9s} seed {s}: d+ {r['d_plus']:+.3f}  d- {r['d_minus']:+.3f}  "
                  f"weight removed {100*r['removed']:.0f}%  [{time.time()-t0:.0f}s]", flush=True)
        dp = np.array([r["d_plus"] for r in runs])
        dm = np.array([r["d_minus"] for r in runs])
        tot = dp + dm
        # S is undefined when nothing moved at all; a seed that did not learn cannot report a
        # specificity, and averaging a 0/0 as zero would read as "learned but did not discriminate".
        ok = np.abs(tot) > 1e-9
        S = (dp[ok] - dm[ok]) / tot[ok]
        out["arms"][arm] = {
            "runs": runs,
            "d_plus_mean": float(dp.mean()), "d_plus_sd": float(dp.std(ddof=1)),
            "d_minus_mean": float(dm.mean()), "d_minus_sd": float(dm.std(ddof=1)),
            "S_mean": float(S.mean()) if len(S) else None,
            "S_sd": float(S.std(ddof=1)) if len(S) > 1 else None,
            "S_n": int(len(S)),
        }
        a = out["arms"][arm]
        print(f"{arm:9s}  d+ {a['d_plus_mean']:+.3f}±{a['d_plus_sd']:.3f}  "
              f"d- {a['d_minus_mean']:+.3f}±{a['d_minus_sd']:.3f}  "
              f"S {a['S_mean'] if a['S_mean'] is None else round(a['S_mean'], 3)}"
              f" (n={a['S_n']})\n", flush=True)

    # the two pre-registered thresholds, applied exactly as frozen in D-073
    P, U = out["arms"]["paired"], out["arms"]["unpaired"]
    learned = P["d_plus_mean"] >= 0.10 and U["d_plus_mean"] < 0.03
    disc = (P["S_mean"] is not None and U["S_mean"] is not None and P["S_sd"] is not None
            and P["S_mean"] >= 0.20
            and P["S_mean"] - U["S_mean"] >= 3 * P["S_sd"])
    out["verdict"] = {"learning": bool(learned), "discrimination": bool(disc)}
    print(f"D-073 thresholds -> learning: {'PASS' if learned else 'FAIL'} | "
          f"discrimination: {'PASS' if disc else 'FAIL'}")

    dest = ROOT / "report" / f"conditioning_{args.dataset}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.name} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
