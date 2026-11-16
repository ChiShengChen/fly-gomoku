#!/usr/bin/env python3
"""Controls (PLAN.md §5.2). An assay that passes on the real connectome has shown something
only if it fails on these.

  degree_preserving_shuffle  rewire while keeping every neuron's in- and out-degree, so a
                             result that survives is about *this* wiring rather than about the
                             degree sequence. Weights and signs travel with the edge.
  sign_shuffle               permute the excitatory/inhibitory assignment across edges, keeping
                             the topology exactly. Separates "the graph does it" from "the
                             excitation/inhibition pattern does it".
  off_target_stim            a size-matched, unrelated stimulus population. Answers the dullest
                             and most necessary question: would anything have driven this?

Every generator takes and returns the same edge arrays and records its seed, so any control run
is reproducible from the manifest alone.
"""
from __future__ import annotations

import numpy as np


def degree_preserving_shuffle(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                              seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute the postsynaptic column. Out-degree is exact per neuron; in-degree is exact in
    distribution (a permutation of the same multiset of targets). Self-loops created by the
    permutation are dropped, which is why the returned arrays can be marginally shorter — the
    count is reported by the runner rather than hidden."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(post))
    post_s = post[perm]
    keep = post_s != pre
    return pre[keep], post_s[keep], w[keep]


def in_preserving_shuffle(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                          seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute the *pre*synaptic column: every target keeps its exact incoming degree and its
    multiset of incoming weights, and only the identity of the sources is randomised.

    The mirror of `degree_preserving_shuffle`, and the pair is the point (D-035). The optic-lobe
    edges are unusually strong — L2→Tm1 runs a median of 128 synapses — so permuting targets
    drops such an edge on a cell that would never receive one, while permuting sources leaves
    who-receives-what intact. If a result survives both, it is about the shape of the graph; if
    only the target-permuting version produces it, it is about strong edges landing wrongly."""
    rng = np.random.default_rng(seed)
    pre_s = pre[rng.permutation(len(pre))]
    keep = pre_s != post
    return pre_s[keep], post[keep], w[keep]


def ei_preserving_shuffle(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                          seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute sources *within each sign class* — the benchmark's null from D-039 onward.

    Each target keeps its in-degree, its incoming weight multiset and its excitatory/inhibitory
    composition exactly, and every source still emits a single sign, so Dale's law survives. The
    only thing randomised is which particular cells supply the inhibition — which is precisely
    what D-038 identified as the property the real connectome is using and none of the earlier
    shuffles preserved.

    Out-degree per source is no longer exact; under a permutation the two sides cannot both hold,
    and for "is this cell driven for a reason?" the input side is the one that matters."""
    rng = np.random.default_rng(seed)
    pre_s = pre.copy()
    for s in (+1, -1):
        m = np.flatnonzero(np.sign(w) == s)
        if len(m) < 2:
            continue
        pre_s[m] = pre[m][rng.permutation(len(m))]
        # Self-loops are *repaired*, not dropped. Dropping them is the usual shortcut and it
        # quietly breaks the exact preservation this null is for: the edges lost after permuting
        # are a different set from the ones the real graph loses, so in-degrees stop matching.
        # Swapping the source of a self-loop against another edge in the same sign class keeps
        # every count intact.
        for _ in range(20):
            bad = m[pre_s[m] == post[m]]
            if not len(bad):
                break
            partner = m[rng.integers(0, len(m), len(bad))]
            ok = (pre_s[partner] != post[bad]) & (pre_s[bad] != post[partner]) & (partner != bad)
            b, p = bad[ok], partner[ok]
            pre_s[b], pre_s[p] = pre_s[p], pre_s[b].copy()
    keep = pre_s != post                      # only edges no swap could rescue
    return pre_s[keep], post[keep], w[keep]


def sign_shuffle(pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                 seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep |w| on each edge, permute the signs across edges. Topology and weight magnitudes
    are untouched, so only the excitation/inhibition assignment is randomised."""
    rng = np.random.default_rng(seed)
    signs = np.sign(w)
    return pre, post, np.abs(w) * signs[rng.permutation(len(w))]


def off_target_population(all_idx: np.ndarray, exclude: np.ndarray, n: int,
                          seed: int) -> np.ndarray:
    """n neurons drawn at random from `all_idx`, excluding the real stimulus and its immediate
    partners. Size-matched to the real stimulus so the comparison is about identity, not count."""
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(all_idx, exclude, assume_unique=False)
    if len(pool) < n:
        raise ValueError(f"off-target pool too small: {len(pool)} < {n}")
    return rng.choice(pool, size=n, replace=False)
