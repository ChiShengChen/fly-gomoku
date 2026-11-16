# Lab notes

These are the entries from the parent project's decision log that cover this experiment, copied
as they were written on the day — including the mistakes, which are the useful part. The parent
project (FlyBench, a behavioural validation suite for connectome-constrained models) keeps an
append-only log in which every entry is numbered `D-nnn`; the numbers are kept here so the
cross-references between entries still work.

Entries referred to but not reproduced, in one line each:

| | |
|---|---|
| D-027 | no single (weight, threshold) setting gives both sparse Kenyon-cell coding and motor output |
| D-034, D-038 | a shuffled connectome is not a weaker network but a more active one; the real wiring is the quiet one |
| D-046 | rule: screenshot a page before publishing it |
| D-048 | the mushroom body is about five synapses from the eye |
| D-052, D-053 | a spiking unit gives zero stimulus modulation in early vision; the optic lobe is modelled with graded units instead |
| D-057 | the visual signal loses roughly fivefold per synapse on the way to the lobula |
| D-064, D-065 | synapse threshold applied per neuron pair instead of per brain region; results re-run on the corrected graph |
| D-078, D-080 | deleting the L1/Mi1 feedback loop, and then every unmodulated input, does not restore the signal — neither is the cause of the loss |

Scripts and tests named below that are not in this repository (`run_kc_code.py`,
`tests/test_plasticity.py`) live in the parent project.

---

## D-066 · The fly "wins" at gomoku and so does its shuffled copy, more often
**2026-09-13**

The viral claim is "someone taught a fly to play X". Built the honest version of it: the board is
rendered as an image, reaches the photoreceptors through the retinotopic map, the hybrid kernel
propagates it, and the move is the square the **lobula** is most active over. Nothing is trained;
there is no reward and no policy. A shuffled copy of the connectome plays the same games.

| | real | shuffled |
|---|---:|---:|
| games won against a random opponent | 11/12 | **12/12** |
| moves the circuit actually decided | 112/142 (79 %) | **159/164 (97 %)** |
| matched a plain peak-luminance heuristic | **0/60** | **0/60** |
| the two brains chose the same square, identical boards | **5/60 (8 %)** | chance ≈ 3 % |

**The shuffled copy wins more and goes silent less** — 5 fallback moves against the real
connectome's 30 — which is D-038 again in a new setting: the real wiring is the quiet one. On
win rate alone the demo "works", and it works *better* without the fly.

**Two things worth keeping.**

* **Neither arm matched the luminance heuristic on a single board.** So the circuit is not simply
  reporting the brightest region; it is computing *something*. That something is not gomoku.
* **The lobula reaches only 30 of 81 squares.** Receptive-field centres are derived from wiring —
  each cell placed at the weighted mean of its inputs' retinal positions, seeded at the
  photoreceptors and pushed forward — and LC4/LPLC2 do not tile the visual field evenly. Moves to
  the other 51 squares are unreachable by this readout. That is the cell types' sampling density,
  not a choice in the code, and it is drawn on the page.

**What the page is for.** Win rate against a random opponent is the number a demo would quote, and
it is the one number that cannot distinguish a connectome from a shuffle of it. The check costs one
extra simulation. Published as an artifact with the shuffled arm beside the real one and the
fallback-to-random count on screen.

The honest next demo is the one the circuit is built for: LC4 and LPLC2 converge 311 cells onto 2
giant-fibre neurons and that pathway answers looming at 280 Hz while beating its null (A1). A
dodging task would test the animal's actual computation. Gomoku tests whether a wiring diagram can
be made to look like it is playing, which is easier and means less.

---

## D-068 · Head to head, a shuffled copy of the connectome beats the connectome 8–0
**2026-09-13**

D-066 had both arms beating a random opponent, which says little. Playing them against each other
is the direct question, and it is what a "two flies at a board" picture actually depicts. Eight
games, **alternating who opens** so a first-move advantage cannot be mistaken for a wiring effect:

| | real connectome | shuffled copy |
|---|---:|---:|
| games won | **0** | **8** |
| moves the circuit decided (vs random opponent) | 112/142 (79 %) | 159/164 (97 %) |

**Not close, and not an opening effect.** The reason is in the fallback counts: the real wiring
falls silent far more often, so it plays at random far more often, and random loses to anything
with a consistent bias. This is D-038 in its bluntest form — the real connectome is the quiet one,
and on any task scored by *doing something*, quiet loses.

**A randomised copy of a fly brain beats the fly brain.** That is the number every demo of this
shape should be asked for, and it costs one extra run.

### The 3D scene, and what is real in it

The bodies are **NeuroMechFly** (NeLy-EPFL, Apache-2.0; Lobato-Rios et al. 2022, *Nature Methods*):
65 parts, 334,525 triangles reduced to 81,866 by vertex clustering, 0.98 MB. The board, the
reaching and the wingbeat are animation. The page says so.

**Getting the mesh assembled took three wrong assumptions, each of which rendered something
plausible or nothing at all:**

1. **No transform.** The parts are not all in one frame: the abdominal segments and all 24 leg
   parts collapse onto the same point.
2. **Composing the joint chain.** `world(parent) @ jointPose @ linkPose` — in SDF 1.6 a
   `<link><pose>` is *already* relative to the model frame, so the chain double-counts and blows
   the body apart. The check that settles it: link poses alone span [2.30, 0, 0.28] and a fly is
   about 3 mm long; the composed chain spans [5.30, 0, 3.92].
3. **Ignoring `<scale>`.** The SDF carries `1000 1000 1000` on every mesh. Without it each part
   came out **one ten-thousandth** of its spacing and **91 % of triangles collapsed to zero area**
   under quantisation — so the buffers, indices, matrices and shaders were all provably correct
   and nothing appeared.

That third one cost the most, because every diagnostic pointed at the renderer: no GL error, buffer
sizes matching byte for byte, indices in range, positions finite and sensible when read back. The
thing that found it was **measuring the geometry in Python instead of asking the browser** — one
line comparing each part's own extent against the spread between parts. A renderer that draws
nothing is not evidence about the renderer.

---

## D-069 · The published page was blank on a phone, and I had not re-tested it
**2026-09-13**

The gomoku artifact opened black on mobile. One error, and it was mine in the most ordinary way:

    Uncaught ReferenceError: Cannot access 'H2H' before initialization

The head-to-head verdict paragraph reads `H2H`, and I inserted it next to the results table —
**above** the `const H2H` that declares it. That is a temporal dead zone error, it throws at script
evaluation, and everything after it never runs, which is why the 2D boards were blank as well as
the 3D scene. The canvases still carried their default 300×380 buffers, which is the tell: `render()`
had never been called.

**It did not appear on desktop because the desktop screenshots predate that paragraph.** I added
the prose and published without rendering again. The rule that D-046 established — screenshot
before publishing — was followed for every earlier change and skipped for the one that broke it,
because it was "just text".

A second, genuine bug the mobile check found: a canvas with only `width` set in CSS keeps its
**height attribute** as its CSS height, so at `devicePixelRatio` 3 every 2D canvas became three
times too tall. The 3D canvas already set `style.height`; the others did not. And the five-column
results table had no scroll container of its own, so it widened the page and clipped the prose at
the right edge.

**What changes.** Publishing checks now run at a phone viewport with a device pixel ratio, not only
at desktop:

    --force-device-scale-factor=3 --window-size=1170,2600

and the check asserts `document.body.scrollWidth === clientWidth` rather than relying on the
screenshot looking right — a narrow screenshot crops, which looks exactly like overflow and is not.

---

## D-070 · The live clouds make the silence visible: 687 cells against 3,079
**2026-09-13**

Each player in the head-to-head scene now carries its own brain: **28,126** of the 139,255 cells at
their FlyWire coordinates — every readout neuron plus a fixed sample of the rest — lighting on the
move during which they fired. It is not a separate visualisation. The per-move simulation that
chooses the stone is the one being drawn, and the activity is stored as sparse indices because the
network is nearly silent and a dense frame would be 99 % zeros.

**Cells firing per move, over all eight games:**

| | median | max |
|---|---:|---:|
| real connectome | **686** | 694 |
| shuffled copy | **3,087** | 3,103 |

**Four and a half times more of the shuffled brain lights up**, and on screen that is the whole
difference: a small dense cloud against a large sprawling one. D-038's finding — the real
connectome is the quiet one — stops being a table and becomes the picture, next to the 8–0 result
that quiet produces.

Both clouds are the same neurons at the same coordinates. Only the wiring between them differs.

**A near-miss worth recording.** `run_gomoku.py` did not fill the 14 cells that have no coordinate,
so the position array carried NaN. `astype(np.uint16)` on NaN does not raise — it emits a
`RuntimeWarning` and writes garbage. The run only failed later, at `int(v / 1000)` for the extent,
and **if I had not been computing that extent the page would have shipped with a handful of
randomly-placed neurons**, which would have looked like noise in a point cloud rather than like an
error. The cells are now placed at the centroid and the count is printed.

Rendering notes, both found by screenshot rather than by reading: 28,126 additively-blended points
saturate to white and stop looking like a brain, so per-point alpha is 0.022 rising to 0.18; and
the clouds were first placed between each fly and the board, where they hid the player they belong
to — they sit behind their fly now.

---

## D-071 · The feedback loop D-057 named is real: L1 ⊣ Mi1 at 120,383 synapses, Mi1 → L1 at 23,444
**2026-09-13**

D-057 measured the visual signal losing about fivefold per synapse and named negative feedback onto
the lamina as the candidate, with the test left explicit so it could not be tuned after the fact.
`report/circuit_matrix.py` reads the signed synapse totals between the 27 cell types on the pathway
straight out of the locked connectome. The loops are there:

| forward | synapses | return | synapses | the loop |
|---|---:|---|---:|---|
| **L1 → Mi1** | **−120,383** | **Mi1 → L1** | **+23,444** | **negative** |
| L2 → C3 | +1,033 | C3 → L2 | −78,936 | negative |
| L1 → C2 | −37,901 | C2 → L1 | −26,418 | positive (two inhibitions) |
| L5 → C2 | +15,216 | C2 → L5 | −23,706 | negative |
| Tm2 → L5 | +30,055 | L5 → Tm2 | 0 | one-way |

**The largest closed loop in the early pathway is L1 ⊣ Mi1 → L1**, sitting on the second synapse of
vision, and it is negative: L1 suppresses Mi1, Mi1 excites L1 back. That is the shape that
compresses a modulation, and it is now a measurement rather than a hypothesis. What it does *not*
yet establish is that this loop causes the attenuation — the deletion test D-057 froze (remove
Mi1→L1 and L5→L1, re-run, see whether L1's modulation grows) has still not been run, and the loop
existing is necessary but not sufficient for that claim.

**The diagram on the page is drawn from this table, not from a textbook figure.** Arrow thickness
is the signed synapse total, colour is the sign, every connection above 400 synapses is drawn.
Which is why the two descending neurons hang off almost nothing: **LC4 reaches DNp01 across 1,601
synapses against the 120,383 L1 spends on Mi1.** That thinness is the result. A hand-drawn
schematic would have given both arrows the same weight and hidden it.

### Two build bugs, the same shape as the one that shipped a blank page

* `str.replace("/*CODE*/", src)` **with no count** put the first script part into *every* slot, so
  the fly-scene code was inlined twice — once before its own declarations — and the circuit code
  was never inlined at all. The build now fills one slot per part and fails loudly if the counts
  disagree either way.
* Adding the second slot with `replace(..., 1)` matched the **recursive** `requestAnimationFrame(tick)`
  inside `tick` rather than the starter at the end of the script, so the parts landed in the wrong
  order. Slots are now placed deliberately, and the built file's declaration order is checked.

Both produced `Cannot access X before initialization` — the same failure as D-069. Three times now
the cause has been code order rather than logic, and every time the page went entirely blank rather
than partly wrong, which is at least a loud way to fail.

---

## D-072 · The board never reaches the reward machinery: 0 of 1,042 Kenyon cells, over 97 moves
**2026-09-13**

Asked whether the gomoku pipeline connects to a dopaminergic reward system. It does not, and
`report/reward_reach.py` measures how far the activity gets rather than asserting it. The
subsample is reproduced from the runner's seed instead of stored, so it cannot drift from the
cells the simulation actually recorded.

| population | in volume | sampled | real, per move | shuffled |
|---|---:|---:|---:|---:|
| KC (Kenyon cells) | 5,177 | 1,042 | **0.00** | 110.95 |
| MBON | 96 | 20 | 0.14 | 16.00 |
| PAM (reward dopamine) | 307 | 69 | **0.00** | 10.75 |
| PPL1 (punishment dopamine) | 16 | 1 | 0.00 | 1.00 |
| LC4 + LPLC2 (the readout) | 314 | 182 | 56.00 | 98.54 |

Across 97 real moves **not one sampled Kenyon cell fired**, against 111 per move in the shuffled
copy — the same pattern as D-034/D-068/D-070, where scrambling removes the attenuation and lets
noise arrive where signal could not. Two independent things are therefore missing before anything
here could be called learning: the signal does not survive the ~5 synapses D-048 measured to the
mushroom body at D-057's ~5× per-synapse loss, and the weights are a fixed anatomical measurement
with no plasticity rule that a dopamine signal could act through. Listed separately so nobody has
to guess which one is the blocker — both are.

### A zero that was an artifact, caught before it shipped

The first version of the table carried a **Mi1 + Tm1** row reading 0.00 in both arms, which reads
as "the medulla is silent two synapses in". It is not a measurement. Under D-053 the entire optic
lobe is graded, and a graded unit never crosses threshold, so **all 3,139 of those cells are
invisible to a spike record by construction** — verified by counting them against the runner's own
`graded` set rather than assumed. The row now carries `None`, prints and renders as `n/a`, and the
page says why; the check is in the script so the zero cannot come back. A `0` and a "not measured"
look identical in a table and mean opposite things, which is exactly the confusion this project
exists to avoid, and it nearly went out as a headline.

`PPL1` is kept for completeness but rests on **one** sampled cell; the page marks the count in the
warning colour. The Kenyon-cell row at 1,042 of 5,177 is the one carrying the claim.

### Also fixed

`report/gomoku_template.html` had **no viewport meta** — the artifact host supplies one, but
`report/gomoku.html` opened directly laid out at 980 px and shrank to fit, which is the likeliest
reason the page has read badly on a phone twice now. Measured at 980/980 before and 390/390 after.

---

## D-073 · PRE-REGISTRATION · A plastic version, in the modality the mushroom body actually has
**2026-09-13 · written before any conditioning run**

Asked for a version with plasticity. Bolting a learning rule onto the gomoku pipeline would be
theatre: D-072 measured **zero** Kenyon-cell spikes there, and a rule multiplying by zero activity
produces a page that looks like learning and is not. The mushroom body does have an input that
arrives — olfaction — so the plastic version is built there, on the paradigm the animal is actually
known for.

### What the probe already established (before any rule exists)

| | |
|---|---|
| ORNs / glomerular types | 2,281 in 53 |
| KC firing to an odour | **26.7–28.6 %** of 5,177 |
| KC→MBON edges available to a rule | 21,438 edges, 169,931 synapses |
| **pairwise KC code overlap (Jaccard, 4 odours)** | **0.666** (0.629–0.734) |
| APL → KC | 5,166 edges, 98,592 inhibitory synapses = **94 % of all inhibition onto KC** |

Two things are already visible. A real fly's KC code is sparse, around a few per cent; this one
is **27 %**, and an odour driving 350 ORNs lights the same number of KCs as one driving 598 — that
is saturation, not encoding. And **APL is already in the simulation**, since the kernel runs the
whole graph; the canonical sparsening feedback is present and connected and the code is dense
anyway. Whatever is wrong is the operating point, not a missing circuit — which is D-027's finding
arriving from a second direction.

### The rule

Dopamine-gated depression of Kenyon-cell output, which is the direction the experimental
literature reports for appetitive conditioning:

    Δw_ij = −η · (r_i / r_max) · (DA_j / DA_max) · w_ij(0),   clipped to [0, w_ij(0)]

for every KC i → MBON j edge. No potentiation, no sign flips. **DA_j is not assumed**: it is
`Σ_d C_jd · rate_d` where `C` is the measured DAN→MBON synapse matrix out of the same connectome,
so which MBONs a reward touches is a property of the wiring rather than a compartment table typed
in by hand. The US is delivered by driving the PAM cells directly, which is what the optogenetic
substitution for sugar does in the bench version of this experiment.

### Protocol and the three controls

Naive test of CS+ and CS−, six paired training trials (CS+ with PAM driven) interleaved with
unpaired CS− trials, then re-test. Ten seeds. Reported against:

1. **Unpaired** — identical odours and identical DA, delivered on trials with no odour. Coincidence
   is the only difference, so any learning that survives this is not learning.
2. **Shuffled KC→MBON** — the degree-preserving null, asking whether the measured wiring does
   anything a random wiring of the same shape does not.
3. **Naive** — the pre-training response, so a drift is not read as a result.

### Frozen thresholds, and the prediction

With Δ± the fractional drop in each odour's MBON response and specificity
**S = (Δ+ − Δ−)/(Δ+ + Δ−)**:

* **learning** if Δ+ ≥ 0.10 and the unpaired control's Δ+ is below 0.03;
* **discrimination** if S ≥ 0.20 and S exceeds the unpaired control's by 3 SD over the ten seeds.

**Prediction, recorded now: it will learn and fail to discriminate.** At Jaccard 0.666 with codes
of ~1,400 cells, 1,119 of them — **80 %** — are shared between any two odours, so depression driven
by CS+ lands on four fifths of CS−'s pathway too. That puts S near **(1−0.80)/(1+0.80) ≈ 0.11**,
below the 0.20 threshold. If S comes out above 0.20 the prediction was wrong and the reason needs
finding; if it comes out near 0.11 then the plastic version works exactly as well as its input
code allows, and the thing to fix is the operating point, not the learning rule.

A7 (D-060) already measured this overlap at 0.814 by a different route. That two independent
measurements of KC code separability agree is the reason this prediction is worth writing down
rather than discovering afterwards.

---

## D-074 · The plastic version learns, does not discriminate, and the connectome adds nothing to it
**2026-09-13 · result of D-073, ten seeds, thresholds unchanged**

| arm | CS+ drop | CS− drop | specificity S |
|---|---:|---:|---:|
| **paired** | **0.225 ± 0.039** | 0.197 ± 0.022 | **+0.059 ± 0.102** |
| unpaired | 0.023 ± 0.031 | 0.013 ± 0.032 | undefined |
| shuffled KC→MBON | 0.239 ± 0.026 | 0.206 ± 0.027 | **+0.074 ± 0.103** |

**Learning: PASS.** 0.225 against the unpaired control's 0.023, which is below the 0.03 the
threshold demanded. Removing only the simultaneity removes the learning, so this is a coincidence
detector and not a decay — the single most important thing for that control to establish.

**Discrimination: FAIL**, and by more than predicted. S = 0.059 against a threshold of 0.20 and
against **0.11 written down beforehand**. The direction of the prediction was right and the
magnitude was optimistic: the measured shared fraction is 0.794, almost exactly the 0.80 the
prediction assumed, so the shortfall is not in that estimate but in the linear reasoning from it.

**The result that was not predicted, and that matters more.** The degree-preserving shuffle of
KC→MBON reaches **S = 0.074**, which is 0.32 standard errors *above* the real wiring. The measured
connectome is indistinguishable from a random rewiring of itself at this task. Whatever separates
the two odours at all — and it is barely more than nothing — the anatomy is not what is doing it.

An earlier two-seed run showed 0.132 against a shuffled 0.062 and I reported it as the connectome
contributing something. At ten seeds that reverses. **It was a two-seed artifact, reported before
the pre-registered n was reached**, and the pre-registration is the only reason the correction is
cheap rather than embarrassing. Do not report an arm before its seeds are in.

### Why, and what it does not license

`runner/run_kc_code.py` measures the input the rule has to work with: **27.7 % ± 1.4** of Kenyon
cells active per odour, where a real fly runs a few per cent, and **79.4 %** of one odour's code
carried by the other. APL is present and connected — 94 % of all inhibition onto the Kenyon cells,
98,592 synapses — and the code is dense anyway. No depression rule can separate two
representations that overlap by four fifths, so the failure sits upstream of the learning rule
entirely.

This does **not** license "the connectome cannot support learning". It licenses exactly one
sentence: *at this operating point, with this kernel, the measured wiring produces a Kenyon-cell
code too dense for a dopamine-gated depression rule to discriminate on, and performs no better
than its own shuffle.* D-027 already found no operating point at which the benchmark works, and
this is that finding arriving through a third door. The test that would separate "the wiring is
wrong" from "the operating point is wrong" is a sweep, and the sweep still has not been re-run on
the corrected graph (D-065).

### One more bug the controls caught

The first implementation normalised dopamine by **each trial's own maximum**, so a trial with
almost no dopaminergic firing still delivered DA = 1.0 to its most-innervated compartment. The
unpaired control depressed 24 % — as hard as the paired arm — which is how it surfaced. The scale
is now measured once from a reward-alone trial and held fixed (`calibrate()`), and
`tests/test_plasticity.py` asserts that a faint reward stays faint, along with: only KC→MBON edges
change, magnitudes only shrink, no weight crosses zero, no dopamine means no change, no
presynaptic activity means no change, and a subset-indexed rate vector is rejected rather than
silently indexing the wrong cells.

Without the unpaired arm this would have shipped as a working learning rule. It was in the
pre-registration, so it ran whether or not the headline needed it.
