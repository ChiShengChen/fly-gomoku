# fly-gomoku

A fruit-fly connectome plays gomoku. A shuffled copy of the same connectome plays the same games,
and wins.

"Someone taught a fly brain to play X" is a cheap claim to make and a cheap one to check. This
repository is the check: run the identical pipeline through a rewired copy of the connectome and
see whether the real wiring did anything. Here it did not — the shuffled copy plays better.

**Open [`index.html`](index.html) in a browser to see the result**: every game, move by move, with
both brains lighting up beside the board. The page is self-contained (about 3.5 MB, with the
recorded simulation embedded) and fetches nothing but its fonts.

## Result

One run, FlyWire v783, 139,255 neurons, seed 1 (2026-09-13).

| | real connectome | shuffled copy |
|---|---:|---:|
| head to head, 8 games, alternating who opens | **0** | **8** |
| games won against a random opponent | 11 / 12 | 12 / 12 |
| moves the circuit decided (the rest fell back to random because it was silent) | 112 / 142 | 159 / 164 |
| neurons firing per move, median, of 28,126 recorded | 686 | 3,087 |
| picked the square a plain brightest-region rule would pick, 60 boards | 0 / 60 | 0 / 60 |

* **Win rate against a random opponent cannot tell a connectome from a shuffle of it.** It is the
  number a demo would quote, and both arms score almost perfectly on it.
* **The real wiring is the quiet one.** It falls silent six times as often, plays at random when it
  does, and loses every head-to-head game to a copy with its wiring scrambled.
* **The circuit is computing something, and it is not gomoku.** Neither arm matched the
  brightest-region rule on a single board, and shown identical boards the two brains chose the
  same square 5 times in 60.
* **The readout reaches only 30 of the 81 squares.** LC4 and LPLC2 do not tile the visual field
  evenly, so 51 squares can never be chosen. That is the cell types' sampling, not a choice in the
  code.
* **Nothing reaches the reward system.** Over 97 moves, 0 of 1,042 sampled Kenyon cells fired, so
  there is no signal a learning rule could act on. The plastic version was therefore built on
  olfaction instead; it learns, does not tell two odours apart, and does no better than its own
  shuffle.
* **The L1/Mi1 feedback loop is real, and it is not why the signal fades.** The visual signal
  loses roughly fivefold per stage. Cutting both return arrows of the loop makes L1 respond
  *less* to light (4.11 % modulation down to 3.48 %, against 3.81 % for a matched random
  deletion). What is left is the model's own gain: the photoreceptors move L1's membrane by about
  one millivolt.

[`docs/NOTES.md`](docs/NOTES.md) is the lab notebook behind these numbers, mistakes included.

## How a move is made

Nothing is trained. There is no reward, no policy and no search.

1. The board is drawn as a 64×64 image: the fly's stones bright, the opponent's dim.
2. The image drives the right eye's photoreceptors (R1-6, R7, R8) through their retinal positions.
3. A leaky integrate-and-fire network of the whole brain runs for 200 ms, carrying the image
   through lamina, medulla and lobula. The optic lobe uses graded (non-spiking) units; everything
   else spikes.
4. Each LC4 and LPLC2 neuron — lobula cells that report *where* something salient is — votes for
   the board square under its receptive field. The receptive-field centres are derived from the
   wiring, not assumed from where the cell body sits.
5. The free square with the most spikes is the move. If no readout cell fired, the move is random
   and is counted as such.

The shuffled copy keeps every neuron's number of inputs and outputs and rewires who connects to
whom. A plain brightest-region rule, with no neurons at all, is shown the same boards as a second
baseline.

## Layout

```
index.html                 the result page (built; open it directly)
docs/NOTES.md              lab notes for this experiment
runner/run_gomoku.py       the experiment: boards, games vs random, head to head
runner/graph.py            signed connectome loader, retinal projection
runner/run_conditioning.py the olfactory plasticity follow-up shown on the page
runner/gomoku.log          console output of the published run
model/lif_torch.py         the network kernel
model/plasticity.py        dopamine-gated depression rule (conditioning only)
controls/generators.py     degree-preserving shuffle and other nulls
report/*.json              the published run's data, one file per page section
report/build_page.py       template + JSON -> index.html
report/reward_reach.py     how far activity gets towards the mushroom body
report/circuit_matrix.py   synapse counts between the cell types on the pathway
data/                      how to obtain the connectome (not stored here)
```

## Running it again

```
pip install -r requirements.txt
python data/prepare_flywire.py        # see data/README.md — needs three files from FlyWire Codex
python runner/run_gomoku.py --games 12 --h2h 8 --boards 60
python report/reward_reach.py
python report/build_page.py
```

The published run took 23 minutes on CPU; no GPU is used. `run_gomoku.py` overwrites
`report/gomoku_flywire.json`.

`data/prepare_flywire.py` rebuilds the input tables from public downloads, but the published run
read them from a lab database export of the same release, and the connection half of that script
has not been checked against a real download. See [`data/README.md`](data/README.md).

## What this is not

Not an emulation of a living fly, and not evidence that a connectome can or cannot play a board
game. The dynamics, the synapse signs and the way the stimulus enters are model assumptions. The
3D flies, the board and the reaching are animation; the bodies are the
[NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly) mesh (Apache-2.0).

The task the readout cells are built for is looming detection: LC4 and LPLC2 converge 311 cells
onto the 2 giant-fibre escape neurons, and in the parent project that pathway answers looming at
280 Hz while beating its shuffle control. A dodging task would test the animal's actual computation. Gomoku
tests whether a wiring diagram can be made to look like it is playing, which is easier and means
less.

## Provenance and credit

This is one experiment lifted out of FlyBench, a behavioural validation suite for
connectome-constrained *Drosophila* models. `runner/graph.py` holds the two functions it needed
from the rest of that project, unedited.

Connectome: FlyWire (Dorkenwald et al. 2024, *Nature*). Cell types and transmitters: Schlegel et
al. 2024, *Nature*; Eckstein et al. 2024, *Cell*. Network model: the kernel follows the
leaky integrate-and-fire brain model of Shiu et al. 2024, *Nature*
([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)). Body
mesh: Lobato-Rios et al. 2022, *Nature Methods*. Full references are in
[`data/README.md`](data/README.md).

Author: Chisheng (Michael) Chen
