#!/usr/bin/env python3
"""Torch LIF kernel — the working implementation, written to match the Brian2 reference.

Semantics are copied from `philshiu/Drosophila_brain_model@main/model.py`, not approximated:

  state      dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
             dg/dt = -g / tau                     (unless refractory)
  spike      v > v_th   ->   v = v_rst, g = 0, refractory for t_rfc
  synapse    on_pre: g += w,  w = (signed synapse count) * w_syn,  delay t_dly
  stimulus   PoissonInput onto **v** (not g) with weight w_syn * f_poi, and the stimulated
             neurons have their refractory period set to zero

Three choices worth stating, because they are where a "close enough" kernel would drift:

* **Exact integration, not Euler.** Brian2 uses `method='linear'`, which solves the 2-variable
  linear system exactly per step. The same closed form is used here, so the two kernels are not
  merely both "an LIF" — they integrate identically.
* **Event-driven propagation.** Only ~3 neurons spike per 0.1 ms step out of ~130k, so the
  network is advanced by gathering the outgoing edges of the spiking rows (CSR by presynaptic)
  rather than by a dense 14.7M-nnz mat-vec. Same arithmetic, ~4 orders of magnitude less of it;
  this is what lets parity run on CPU while a GPU is unavailable.
* **Refractory drops synaptic input entirely.** This one was found the hard way: the first
  version assumed Brian2 freezes v and g while letting `on_pre` still accumulate, which made
  synaptically-driven neurons fire ~28 % too fast. Brian2 guards writes to a variable declared
  `(unless refractory)`, so input arriving during the refractory window is *lost*, not banked.
  A hand-designed arrival train now reproduces Brian2's spike steps exactly
  (`model/refractory_probe.py`).

Two step-level offsets come from Brian2's scheduling and are reproduced rather than rounded
away: `on_pre` takes effect after the state update, so the effective delay is one step longer
than `t_dly / dt`; and `not_refractory` is `t - lastspike > t_rfc`, which makes the refractory
window one step longer than `t_rfc / dt`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LIFParams:
    """Defaults are the reference's own (DECISIONS D-014); nothing here is from memory."""
    v_0: float = -52.0e-3      # V, resting
    v_rst: float = -52.0e-3    # V, reset
    v_th: float = -45.0e-3     # V, threshold
    t_mbr: float = 20.0e-3     # s, membrane time constant
    tau: float = 5.0e-3        # s, synaptic time constant
    t_rfc: float = 2.2e-3      # s, refractory period
    t_dly: float = 1.8e-3      # s, synaptic delay
    w_syn: float = 0.275e-3    # V, per synapse
    r_poi: float = 150.0       # Hz
    f_poi: float = 250.0       # Poisson weight scaling
    dt: float = 0.1e-3         # s
    t_run: float = 1.0         # s
    # --- graded units (D-053); unused unless TorchLIF is given a `graded_idx` ---
    r_max_hz: float = 100.0    # full release drives targets like a spiking cell at this rate
    v_half: float | None = None    # half-activation; None -> v_0, i.e. half-open at rest
    k_release: float = 5.0e-3  # V, release slope


class TorchLIF:
    def __init__(self, n_neurons: int, pre: np.ndarray, post: np.ndarray,
                 weight_counts: np.ndarray, params: LIFParams | None = None,
                 device: str = "cpu", dtype=torch.float32,
                 graded_idx: np.ndarray | None = None):
        """weight_counts: the signed synapse count per edge ('Excitatory x Connectivity').

        `graded_idx` names neurons to simulate as **graded** units instead of spiking ones
        (D-053): same passive membrane, no threshold, no reset, no refractory, and a continuous
        release that is half-open at rest so that inhibition has an ongoing output to reduce.
        Leaving it `None` is not merely the default — it must reproduce the spiking kernel
        exactly, and `tests/test_hybrid_graded.py` asserts that element-wise."""
        self.p = params or LIFParams()
        self.n = n_neurons
        self.device = torch.device(device)
        self.dtype = dtype

        order = np.argsort(pre, kind="stable")          # CSR over presynaptic neurons
        self.tgt = torch.as_tensor(post[order].astype(np.int64), device=self.device)
        self.w = torch.as_tensor((weight_counts[order] * self.p.w_syn).astype(np.float32),
                                 device=self.device, dtype=dtype)
        counts = np.bincount(pre.astype(np.int64), minlength=n_neurons)
        self.indptr = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                                      device=self.device)

        p = self.p
        self.a = p.dt / p.t_mbr                          # decay exponents
        self.ev = math.exp(-p.dt / p.t_mbr)
        self.eg = math.exp(-p.dt / p.tau)
        # exact solution of dv/dt = (v0 - v + g)/t_mbr with g decaying at 1/tau
        ia, ib = 1.0 / p.t_mbr, 1.0 / p.tau
        self.cross = ia * (self.eg - self.ev) / (ia - ib)
        # +1 on each: Brian2 applies on_pre after the state update, and its refractory test
        # is strictly-greater, so both windows are one step longer than the naive division.
        self.delay_steps = int(round(p.t_dly / p.dt)) + 1
        self.rfc_steps = int(round(p.t_rfc / p.dt)) + 1
        self.n_steps = int(round(p.t_run / p.dt))

        # --- graded partition -------------------------------------------------------------
        self.graded = None
        if graded_idx is not None and len(graded_idx):
            gi = np.unique(np.asarray(graded_idx, dtype=np.int64))
            self.graded = torch.as_tensor(gi, device=self.device)
            self.is_graded = torch.zeros(n_neurons, dtype=torch.bool, device=self.device)
            self.is_graded[self.graded] = True
            # Release is dense: every graded cell drives its targets every step, where a spiking
            # cell only does so when it fires. So the graded fan-out is one sparse mat-vec per
            # step, W_g @ r, built once -- not the per-spike gather the spiking path uses.
            col = np.full(n_neurons, -1, np.int64)
            col[gi] = np.arange(len(gi))
            sel = col[pre.astype(np.int64)] >= 0
            rows = post.astype(np.int64)[sel]
            cols = col[pre.astype(np.int64)[sel]]
            vals = (weight_counts[sel] * p.w_syn).astype(np.float32)
            o = np.argsort(rows, kind="stable")
            crow = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_neurons))])
            self.Wg = torch.sparse_csr_tensor(
                torch.as_tensor(crow, device=self.device),
                torch.as_tensor(cols[o], device=self.device),
                torch.as_tensor(vals[o], device=self.device, dtype=dtype),
                size=(n_neurons, len(gi)))
            self.r_scale = p.r_max_hz * p.dt
            self.v_half = p.v_0 if p.v_half is None else p.v_half
            self.k_rel = p.k_release

    def run(self, stim_idx: np.ndarray, seed: int, rate_schedule: np.ndarray | None = None,
            frame_steps: int | None = None, record_pop: np.ndarray | None = None,
            record_cells: np.ndarray | None = None,
            graded_stim: np.ndarray | None = None,
            graded_schedule: np.ndarray | None = None,
            record_graded: list | None = None,
            record_graded_cells: bool = False):
        """One trial. Returns spike counts per neuron (length n).

        `rate_schedule` drives a video: an (n_frames, n_stim) array of per-cell Poisson rates in
        Hz, held for `frame_steps` steps each. Without it every stimulated cell gets the constant
        `params.r_poi`, which is what the fixed-population assays use.

        `record_pop` is an index array, or a list of them; when given, the method also returns
        per-step spike counts within each population, which is what a video readout or a
        field-potential trace needs — the assays only ever wanted totals, so this stays
        opt-in.

        `record_graded` is a list of index arrays of *graded* cells, returning each group's mean
        release per frame **in spike-equivalent Hz** — a graded unit has no spikes to count, and
        reporting release in the same units as everything else is what makes the two populations
        comparable on one axis.

        `record_graded_cells` returns the release of **every** graded unit per frame, which a
        visualisation needs and a summary does not: the group means above cannot show a wave
        moving across a population.

        `record_cells` is an index array returning per-*frame* counts for each of those cells,
        shape (len(record_cells), n_frames). Per-step would be 250x larger for no gain — a
        visualisation samples at the frame rate — and per-frame keeps a whole-brain recording
        small enough to be worth taking."""
        p, dev = self.p, self.device
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        v = torch.full((self.n,), p.v_0, device=dev, dtype=self.dtype)
        g = torch.zeros(self.n, device=dev, dtype=self.dtype)
        refr = torch.zeros(self.n, dtype=torch.int32, device=dev)
        counts = torch.zeros(self.n, dtype=torch.int32, device=dev)

        ring = torch.zeros((self.delay_steps + 1, self.n), device=dev, dtype=self.dtype)
        stim = torch.as_tensor(np.asarray(stim_idx, dtype=np.int64), device=dev)
        # PoissonInput(N=1, rate) -> Bernoulli(rate*dt) events per step, weight added to v
        p_event = p.r_poi * p.dt
        poi_w = p.w_syn * p.f_poi
        sched = None
        if rate_schedule is not None:
            sched = torch.as_tensor(np.asarray(rate_schedule, dtype=np.float32) * p.dt,
                                    device=dev)
            if sched.shape[1] != len(stim_idx):
                raise ValueError(f"rate_schedule has {sched.shape[1]} columns for "
                                 f"{len(stim_idx)} stimulated cells")
            frame_steps = frame_steps or max(1, self.n_steps // sched.shape[0])
        groups = None
        if record_pop is not None:
            raw = record_pop if isinstance(record_pop, (list, tuple)) else [record_pop]
            # NB: not `for g in raw` — `g` is the conductance state variable, and shadowing it
            # here leaves it holding the last readout group instead of the network state.
            groups = [torch.as_tensor(np.asarray(grp, dtype=np.int64), device=dev) for grp in raw]
        pop_trace = (torch.zeros((len(groups), self.n_steps), dtype=torch.int32, device=dev)
                     if groups else None)
        cells = cell_trace = None
        if record_cells is not None:
            cells = torch.as_tensor(np.asarray(record_cells, dtype=np.int64), device=dev)
            n_fr = (sched.shape[0] if sched is not None
                    else max(1, self.n_steps // (frame_steps or self.n_steps)))
            cell_trace = torch.zeros((len(cells), n_fr), dtype=torch.int16, device=dev)
        rfc_of = torch.full((self.n,), self.rfc_steps, dtype=torch.int32, device=dev)
        rfc_of[stim] = 0                                  # reference sets rfc=0 for Poisson targets

        gtrace = gcols = None
        if record_graded is not None and self.graded is not None:
            col = torch.full((self.n,), -1, dtype=torch.long, device=dev)
            col[self.graded] = torch.arange(len(self.graded), device=dev)
            # One sparse averaging matrix rather than a Python loop per group per step: with a
            # few hundred readout types that loop is millions of tiny tensor ops and dominates
            # the run.
            rows_a, cols_a, vals_a = [], [], []
            for gi_, grp in enumerate(record_graded):
                c = col[torch.as_tensor(np.asarray(grp, dtype=np.int64), device=dev)]
                if (c < 0).any():
                    raise ValueError("record_graded was given a cell that is not graded")
                rows_a.append(torch.full((len(c),), gi_, dtype=torch.long, device=dev))
                cols_a.append(c)
                vals_a.append(torch.full((len(c),), 1.0 / len(c), device=dev,
                                         dtype=self.dtype))
            gcols = torch.sparse_coo_tensor(
                torch.stack([torch.cat(rows_a), torch.cat(cols_a)]), torch.cat(vals_a),
                (len(record_graded), len(self.graded))).coalesce().to_sparse_csr()
            n_fr = (sched.shape[0] if sched is not None
                    else max(1, self.n_steps // (frame_steps or self.n_steps)))
            gtrace = torch.zeros((len(record_graded), n_fr), dtype=torch.float32, device=dev)

        gcell = None
        if record_graded_cells and self.graded is not None:
            n_fr = (sched.shape[0] if sched is not None
                    else max(1, self.n_steps // (frame_steps or self.n_steps)))
            gcell = torch.zeros((len(self.graded), n_fr), dtype=torch.float32, device=dev)

        gstim = gsched = None
        if graded_stim is not None and len(graded_stim):
            # A graded cell has no spikes to receive, so its drive is a sustained conductance in
            # the same mV units as a synapse -- injecting Poisson events instead would reintroduce
            # the private noise source that decoupled the network in D-052.
            gstim = torch.as_tensor(np.asarray(graded_stim, dtype=np.int64), device=dev)
            gsched = torch.as_tensor(np.asarray(graded_schedule, dtype=np.float32), device=dev)
            if gsched.shape[1] != len(gstim):
                raise ValueError(f"graded_schedule has {gsched.shape[1]} columns for "
                                 f"{len(gstim)} driven cells")

        for step in range(self.n_steps):
            slot = step % (self.delay_steps + 1)
            active = refr <= 0
            # input to a refractory neuron is dropped, not banked (see module docstring)
            g = g + ring[slot] * active
            ring[slot].zero_()

            v_new = p.v_0 + (v - p.v_0) * self.ev + self.cross * g
            g_new = g * self.eg
            v = torch.where(active, v_new, v)             # frozen while refractory
            g = torch.where(active, g_new, g)

            if stim.numel():
                if sched is not None:
                    f = min(step // frame_steps, sched.shape[0] - 1)
                    thresh = sched[f]
                else:
                    thresh = p_event
                fired = (torch.rand(stim.numel(), generator=gen).to(dev) < thresh)
                if fired.any():
                    v[stim[fired]] += poi_w

            if self.graded is not None:
                # continuous release, half-open at rest, delivered down the same delay ring so
                # graded and spiking transmission are not given different latencies by accident
                r = self.r_scale * torch.sigmoid((v[self.graded] - self.v_half) / self.k_rel)
                arrive_g = (step + self.delay_steps) % (self.delay_steps + 1)
                ring[arrive_g] += torch.mv(self.Wg, r)
                if gtrace is not None or gcell is not None:
                    fr = min(step // (frame_steps or self.n_steps),
                             (gtrace if gtrace is not None else gcell).shape[1] - 1)
                    if gtrace is not None:
                        gtrace[:, fr] += torch.mv(gcols, r)
                    if gcell is not None:
                        gcell[:, fr] += r
                if gstim is not None:
                    f = min(step // frame_steps, gsched.shape[0] - 1) if gsched.ndim > 1 else 0
                    g[gstim] += gsched[f] * (1.0 - self.eg)   # steady state -> the declared mV

            spikes = (v > p.v_th) & active
            if self.graded is not None:
                spikes = spikes & ~self.is_graded          # graded units never fire
            if spikes.any():
                idx = spikes.nonzero(as_tuple=True)[0]
                v[idx] = p.v_rst
                g[idx] = 0.0
                refr[idx] = rfc_of[idx]
                counts[idx] += 1
                arrive = (step + self.delay_steps) % (self.delay_steps + 1)
                for s in idx.tolist():                    # CSR gather of this row's edges
                    lo, hi = int(self.indptr[s]), int(self.indptr[s + 1])
                    if hi > lo:
                        ring[arrive].index_add_(0, self.tgt[lo:hi], self.w[lo:hi])

            if pop_trace is not None and spikes.any():
                for gi, grp in enumerate(groups):
                    pop_trace[gi, step] = int(spikes[grp].sum())

            if cell_trace is not None and spikes.any():
                fr = min(step // (frame_steps or self.n_steps), cell_trace.shape[1] - 1)
                cell_trace[:, fr] += spikes[cells].to(torch.int16)

            refr = torch.clamp(refr - 1, min=0)

        out = [counts.cpu().numpy()]
        if pop_trace is not None:
            out.append(pop_trace.cpu().numpy())
        if cell_trace is not None:
            out.append(cell_trace.cpu().numpy())
        fs = frame_steps or self.n_steps
        if gtrace is not None:
            # summed over the steps in a frame, so divide back out, then r -> Hz-equivalent
            out.append((gtrace / fs / self.r_scale * self.p.r_max_hz).cpu().numpy())
        if gcell is not None:
            out.append((gcell / fs / self.r_scale * self.p.r_max_hz).cpu().numpy())
        return tuple(out) if len(out) > 1 else out[0]

    def run_many(self, stim_idx, n_run: int, seed0: int = 0) -> np.ndarray:
        """n_run trials -> (n_run, n_neurons) spike counts, one seed per trial."""
        return np.stack([self.run(stim_idx, seed0 + i) for i in range(n_run)])
