"""Dopamine-gated depression of Kenyon-cell output synapses (D-073).

The one learning rule in this project, and it is deliberately the narrowest one that the
experimental literature supports: coincidence of Kenyon-cell activity with dopamine in the
same compartment depresses that KC→MBON synapse. No potentiation, no sign changes, nothing
applied anywhere else in the brain.

What is *not* assumed here is which MBONs a reward reaches. The compartment coupling is read
out of the same connectome as everything else — the measured DAN→MBON synapse matrix — so a
wrong compartment table cannot quietly become the result.
"""
from __future__ import annotations

import numpy as np


class KCOutputDepression:
    """Holds the plastic subset of the weight vector and applies the rule between trials."""

    def __init__(self, pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                 kc: np.ndarray, mbon: np.ndarray, dan: np.ndarray,
                 eta: float, r_max_hz: float, floor: float = 0.0):
        self.w = w.astype(np.float64).copy()
        self.eta = float(eta)
        self.r_max = float(r_max_hz)
        self.floor = float(floor)

        # the plastic edges: Kenyon cell -> mushroom body output neuron
        self.edge = np.flatnonzero(np.isin(pre, kc) & np.isin(post, mbon))
        if not len(self.edge):
            raise ValueError("no KC->MBON edges: the rule would be a no-op")
        self.w0 = self.w[self.edge].copy()
        self.e_pre, self.e_post = pre[self.edge], post[self.edge]

        # DAN -> MBON synapse counts, i.e. which compartment a given dopaminergic cell writes to.
        # Read from the graph; magnitude only, because a DAN's sign in the connectome describes
        # its fast transmitter and the modulatory action is what is being modelled here.
        dan_of = {int(d): k for k, d in enumerate(dan)}
        mb_of = {int(m): k for k, m in enumerate(mbon)}
        self.C = np.zeros((len(mbon), len(dan)), np.float64)
        sel = np.isin(pre, dan) & np.isin(post, mbon)
        for a, b, v in zip(pre[sel], post[sel], np.abs(w[sel])):
            self.C[mb_of[int(b)], dan_of[int(a)]] += float(v)
        self.mbon, self.dan = mbon, dan
        self.da_scale: float | None = None
        self.n_compartment_edges = int(sel.sum())

    def calibrate(self, us_rate_hz: np.ndarray) -> float:
        """Fix the dopamine scale from one reward-alone trial. Must be called before `step`.

        Normalising each trial by its *own* maximum instead would make an unrewarded trial's
        residual dopamine saturate to 1 and depress as hard as a reward — which is to say it would
        abolish the difference conditioning is made of. The unpaired control caught exactly that,
        so the reference is measured once and held."""
        self.da_scale = float((self.C @ np.asarray(us_rate_hz, np.float64)[self.dan]).max())
        if self.da_scale <= 0:
            raise ValueError("the reward trial produced no dopamine: nothing to calibrate against")
        return self.da_scale

    def dopamine(self, rate_hz: np.ndarray) -> np.ndarray:
        """Per-MBON dopamine, in units of what a full reward delivers. Not re-normalised per trial.

        `rate_hz` is indexed by **global neuron id**, like everything else in the kernel. Taking a
        pre-subset vector here would index silently into the wrong cells."""
        if self.da_scale is None:
            raise RuntimeError("calibrate() must run on a reward-alone trial first")
        return (self.C @ np.asarray(rate_hz, np.float64)[self.dan]) / self.da_scale

    def step(self, rate_hz: np.ndarray) -> float:
        """Apply one trial's depression from a full-length rate vector.

        Returns the fraction of plastic weight removed so far."""
        rate_hz = np.asarray(rate_hz, np.float64)
        if len(rate_hz) <= int(max(self.e_pre.max(), self.dan.max())):
            raise ValueError("step() wants a rate vector indexed by global neuron id")
        da = self.dopamine(rate_hz)
        r = np.clip(rate_hz[self.e_pre] / self.r_max, 0.0, 1.0)
        da = np.clip(da, 0.0, 1.0)
        mb_of = np.searchsorted(self.mbon, self.e_post)
        d = self.eta * r * da[mb_of] * np.abs(self.w0)
        # Depression towards zero from either sign, clipped at the floor. Using w0 rather than the
        # running weight keeps the step size from collapsing as the weight shrinks, which would
        # make the rule stall rather than saturate.
        new = np.sign(self.w0) * np.maximum(np.abs(self.w[self.edge]) - d,
                                            self.floor * np.abs(self.w0))
        self.w[self.edge] = new
        return float(1.0 - np.abs(new).sum() / np.abs(self.w0).sum())

    def reset(self) -> None:
        self.w[self.edge] = self.w0.copy()
