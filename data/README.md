# Data

Nothing in this folder is needed to **look at** the result — `docs/index.html` and the JSON files
in `report/` are the published run. The connectome is only needed to run it again, and it is
downloaded rather than stored here.

## What the runner reads

All three files go in `data/raw/` (git-ignored).

| file | rows | what |
|---|---:|---|
| `flywire_annotations.parquet` | 139,255 | one row per neuron: `body_id`, `type`, `super_class`, `soma_side` (L/R/M), `consensus_nt`, … |
| `flywire_connections.parquet` | 3,869,878 | one row per (pre, post, neuropil): `pre_id`, `post_id`, `weight` (synapses), `nt_type` |
| `flywire_annotations_public.tsv` | 139,241 | the Schlegel et al. table as released; read for the `pos_x/y/z` coordinates |

The published run's connection table holds 34,153,566 synapses, and no neuron **pair** in it has
fewer than 5 (single neuropil rows go down to 1).

## Getting them

```
python data/prepare_flywire.py
```

downloads the annotation table and builds both parquet files, once you have put these in
`data/raw/` by hand (they sit behind a free login):

* `neurons.csv.gz`, `connections.csv.gz`, and ideally `classification.csv.gz` — FlyWire Codex,
  snapshot **783**, https://codex.flywire.ai/api/download

**How far this has been checked.** The published run read its tables from a lab database holding
the same FlyWire v783 release, not from these downloads. The annotation half of
`prepare_flywire.py` was checked against the table that run used: identical in every column when
`classification.csv.gz` is supplied. Without it, 644 neurons lose their cell type (mostly `L1-3`
and ocellar photoreceptors; none of the types this experiment drives or reads) and 14 lose their
class and side. The connection half has not been run against a real Codex download — compare the
counts it prints with the table above before relying on it.

## Sources

* Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature 634:124–138 — the
  FlyWire connectome.
* Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of Drosophila*,
  Nature 634:139–152 — cell types, sides, transmitters
  ([flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)).
* Eckstein et al. 2024, *Neurotransmitter classification from electron microscopy images at
  synaptic sites in Drosophila melanogaster*, Cell 187:2574–2594 — transmitter predictions.
* Lobato-Rios et al. 2022, *NeuroMechFly*, Nature Methods 19:620–627 — the fly body mesh in the
  3D scene (Apache-2.0; `data/fetch_fly_mesh.py` rebuilds `report/fly_parts.json` from it).
* Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*,
  Nature 634:210–219 — the leaky integrate-and-fire model `model/lif_torch.py` follows.
