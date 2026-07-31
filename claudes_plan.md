# Comparative Analysis Plan — Reviewer Response

Plan for the comparative/validation work needed to address the four reviews in
`todo-list.txt` for *"Open-Source SAR Simulator in PyTorch Via Interposummation."*

Written against:
- `shapenet_sar_sim/` — our simulator
- `Civilian_Vehicle_Data_Domes/` + `/workspace/data/Civilian_Vehicle_Data_Domes/` — CVDomes phase histories
- `/workspace/data/cv_domes_cad_models_ojb_mtl_blend/` — the *same* CAD models as `.obj`
- `RaySAR_Python/` — Python port of RaySAR image formation

---

## 0. What the reviewers actually want

| Reviewer | Core demand | Addressed by |
|---|---|---|
| R1.1 | Coherent validation: point scatterer, plate, trihedral, known phase history | **A** |
| R1.2 | Metrics (peak location, resolution, sidelobes) vs RaySAR/CVDomes conditions | **A**, **B2** |
| R1.3 | Units, coordinate system, scale, reproducibility | **§1**, **F** |
| R2.1 | Position interposummation vs classical matched filtering | **E** |
| R2.2 | Quantitative/qualitative comparison to established simulators | **B**, **C**, **D** |
| R2.3 | Explain why complex-valued imaging performs poorly | **§2**, **A1** |
| R2.4 | Colorbars/units on Fig. 1(c),(d) | **F** |
| R3 | More discussion of coherent imaging | **§2**, **A1** |
| R4 | *Any* verification of correctness; benchmarks | **A**, **B6**, **D** |

Two reviewers (R1, R4) will likely reject again without **A**. R2 will likely
reject again without **B** and **E**.

---

## 1. Prerequisite: fix the unit convention

Reviewer 1's point 3 is correct and currently unanswerable, because the repo uses
three inconsistent scale conventions:

- `paper_figures.py:23-24` — `spatial_bw = spatial_fs = 3650/50` (implies 3650 mm scene / 50 mm resolution)
- `render_cvdomes.py:18,25-26` — `spatial_bw = 3680/RESOLUTION_MM` with `RESOLUTION_MM = 100`
- `render_cvdomes.py:45` — `mesh_scale = 0.05`, while the CVDomes `Camry_06212012.obj` has extent
  `12.08 × 3.66 × 5.05` in its native units
- `wavelength = 0.5` in normalized scene units → ~1.8 m physical. Not X-band, not any band a
  reviewer will recognize.

**Action.** Adopt CVDomes' physical convention throughout: **meters, X-band**. Every comparison
below then becomes direct, and the parameter table in the paper becomes reproducible.

Publish, in the paper, an explicit table of: scene unit ↔ meters, wavelength in meters and GHz,
`spatial_bw`/`spatial_fs` in m⁻¹ *and* the temporal `B_t`/`F_t` in Hz they derive from
(`B_s = 2B_t/c`), ray-grid spacing in meters, and the world coordinate system
(x/y/z axes, azimuth and elevation sign conventions, sensor side of the image).

---

## 2. Three code issues that will corrupt any comparison

Fix before generating comparison numbers, or the numbers describe artifacts.

### 2.1 Per-pulse normalization by return count — `signal_simulation.py:333`

```python
ray_normalized_signal = signal / R      # R = scatter_z.shape[-1]
```

`render_images.py:123-131` passes `all_ranges[t][p]`, which contains **only the hit rays**, so `R`
is the number of returns *for that pulse* and varies pulse to pulse as the aspect changes.

Effect: an artificial, aspect-dependent amplitude taper across the synthetic aperture. It degrades
coherent integration and is a direct confound for the RCS-vs-aspect comparison (**B3**).

Fix: normalize by a constant — ray-grid area, or total transmitted ray count `n_ray_width *
n_ray_height` — so results are invariant to how many rays happened to hit.

### 2.2 Phasors weighted by energy instead of amplitude — `accumulate_scatters.py:318-324`

```python
scatter_energies[t][p] = scatter_energies[t][p] * torch.exp(
    1j * 2 * np.pi / wavelength * scatter_ranges[t][p]
)
```

`energy_b` (`accumulate_scatters.py:240-244`) is a **power** fraction by construction — it comes
from the `R + S + A = 1` / `D + I = 1` energy-conservation derivation in the paper. A coherent
field sum must be over field **amplitudes**, i.e. weight ∝ `sqrt(E)`. Summing power-weighted
phasors produces the wrong interference structure.

This is a plausible contributor to R2.3 ("complex signals perform poorly"). **A1/A2 will confirm
or rule it out immediately** — do that before investing in CVDomes infrastructure.

Sign check (this part is fine): simulation applies `exp(+j·2π·z_2way/λ)`; `imaging_algorithms.py:52`
compensates with `exp(-4jπ·sample_z/λ)` on the one-way `sample_z`. Consistent.

### 2.3 Unstated phase-sampling Nyquist condition on the ray grid

Adjacent rays must differ in two-way range by much less than λ/2:

```
δ = 2 · Δ_ray · sin θ  <  λ/2      ⇒      Δ_ray < λ / (4 sin θ)
```

so, worst case (grazing),  **Δ_ray < λ/4**.

Current defaults: `grid_width = 1.2`, `n_ray_width = 128` → `Δ_ray = 0.0094`, `λ = 0.5`. Margin ≈ 13×,
so the baseline is safe. **The wavelength sweep (Fig. 8) crosses the limit.** The "interference
patterns from constructive and destructive summation across nearby rays" described in the paper is
**phase aliasing**, not physics.

This is the real answer to R2.3 and R3 — and it is a genuine contribution once stated as a design
rule rather than an artifact.

**Consequence for the CVDomes comparison, state it honestly:** at λ = 3.12 cm on a 4.5 m vehicle,
Δ_ray < 7.8 mm → **≈ 600² rays minimum**, realistically **1024²–2048²**. The current 128² is nowhere
near coherent-fidelity requirements at true X-band. Budget compute accordingly; report the
requirement rather than letting a reviewer find it.

### 2.4 Minor, worth a sentence in the paper

`imaging_algorithms.py:169` — `CBP_2D` returns `sqrt(re² + im²)`, so the imaging output is always
non-negative real. Fine, but say so; a reviewer asking about coherence will look here.

---

## 3. Reference parameters extracted from CVDomes

Measured from `/workspace/data/Civilian_Vehicle_Data_Domes/Domes/Camry/Camry_el32.0000.mat`:

```
512 frequency bins, 6.9226 → 12.2775 GHz,  Δf = 10.479 MHz
B  = 5.355 GHz          fc = 9.600 GHz          λ = 3.12 cm
range resolution        c/2B  = 2.80 cm
unambiguous range       c/2Δf = 14.31 m
azimuth  5760 pulses / 360°  →  0.0625° spacing
elevation 30°–60°,  polarizations HH / VV / HV  (each 512 × 5760 complex128)
```

Derived aperture parameters for a matched comparison:

| Aperture | Pulses | Cross-range res `λ/2Δθ` |
|---|---|---|
| 10° | 160 | 8.9 cm |
| 30° | 480 | **2.98 cm** (≈ matched to 2.80 cm range res) |
| 90° | 1440 | 0.99 cm |

**Use the 30° aperture as the primary comparison point** — range and cross-range resolution are
balanced, so image structure is isotropic and metrics are interpretable.

CAD: `Camry_06212012.obj`, 24609 verts, native extent `12.08 × 3.66 × 5.05` (Y is up, `min_y ≈ 0.025`).
Pin down the native unit before setting `mesh_scale` (see §1).

---

## Tier 1 — required

### A. Canonical-target validation against closed form

Cheapest, highest value. Needs no second simulator. Answers R1.1, R3, R4 simultaneously, and is
the fastest way to confirm/refute §2.2.

| # | Target | Reference | Metrics to report |
|---|---|---|---|
| A1 | Single ideal point scatterer at known (x,y,z) | analytic `exp(-j4πR(θ)/λ)` | **Phase residual across the aperture** (the direct evidence R1 demands); peak location error (target < 0.1 px); mainlobe width vs `c/2B` and `λ/2Δθ`; PSLR vs −13.26 dB; ISLR |
| A2 | Flat plate | PO: `σ = 4πA²cos²θ/λ² · sinc²(·)` | Specular glint mainlobe width = λ/L; RCS vs aspect curve overlay |
| A3 | **Dihedral and triangular trihedral** | `σ_dih = 8πa²b²/λ²`, `σ_tri = 4πL⁴/3λ²` | **Validates the 2-bounce path — the headline claim.** Also: double-bounce return must land at the correct phase-center range (the corner line) |
| A4 | Two points separated by exactly one resolution cell | Rayleigh criterion | Resolution is real, not nominal |
| A5 | Sphere (sweep already exists, Fig. 13) | `σ = πa²`, aspect-independent | Make the existing figure quantitative: measured RCS vs `πa²` across the five scales |

A1 and A3 are the minimum. A1 is the single most important experiment in the revision — it is the
literal thing R1 asked for and the thing R4 says is entirely missing.

### B. CVDomes head-to-head

The strongest card: **same CAD models and ground-truth complex phase histories.** No other
open-source SAR simulator paper can do this.

- **B1 — Phase-history level.** Emit our signal on the same K×Np grid (512 bins × N pulses over the
  band in §3). Compare per pulse: complex correlation `|⟨s_ours, s_cv⟩| / (‖s_ours‖‖s_cv‖)`,
  range-profile magnitude, and aperture-domain phase.
  *This is the figure that answers "coherently accurate phase histories, not just visually
  plausible magnitude images" head-on.*

- **B2 — Image level, through the *same* backprojector.** Critical: run both phase histories
  through **one** imaging path — port `Civilian_Vehicle_Data_Domes/code/bpBasicFarField.m`, or feed
  the CVDomes history into our CBP. Otherwise the comparison measures imagers, not simulators.
  Metrics: NCC on dB images, SSIM, and a scatterer-location metric (pick top-N peaks in each, report
  mean nearest-neighbor distance in resolution cells). Match CVDomes' display convention:
  70 dB dynamic range, 10 m × 10 m scene.

- **B3 — Aspect sweep.** 0–360° in 10° steps at fixed elevation. Plot integrated image energy /
  RCS vs aspect for both. Real vehicles show large broadside glints near 0/90/180/270°.
  Directly answers R1.2's "aspect dependence, specular glints, diffuse scattering" — and it is
  informative whether or not we reproduce the glints.

- **B4 — Elevation sweep.** 30°–60° available. Compare layover and shadow behavior.

- **B5 — Polarization.** We have none. Compare against VV only and **state the limitation
  explicitly.** Better said by us than found by a reviewer.

- **B6 — Multi-bounce ablation scored against CVDomes.** Run `num_bounce = 1` vs `2`, score both
  with the B1/B2 metrics.
  **If two bounces measurably improves the match to a physics-based reference, that is quantitative
  validation of the paper's central novelty.** Probably the highest-value single experiment in the
  whole revision.

- **B7 — Material calibration** (the `todo-list.txt` "find hyperparameters that best match CVDomes"
  item). Fit `(R,S,A,I,D)` on 2 vehicles, then **evaluate on held-out vehicles and aspects**.
  Reframes "we tuned hyperparameters" as "we calibrated, and it generalizes." Report fitted values
  and a sensitivity analysis.

Ten vehicles are available in both `.obj` and dome form: Camry, HondaCivic4dr, Jeep93, Jeep99,
Maxima, MazdaMPV, Mitsubishi, Sentra, ToyotaAvalon, ToyotaTacoma. Use ~2 for calibration, the rest
for evaluation.

### D. Computational trade-offs

R2 asks for this explicitly, and it is the one axis where we clearly win — so make it prominent.

Table: seconds per phase history, GPU vs CPU, scaling in ray count and pulse count, peak memory —
against POV-Ray/RaySAR wall time and Dungan's reported CVDomes compute cost
(`Civilian_Vehicle_Data_Domes/Dungan_SPIE10a.pdf`, `CVDomes.pdf`).

Pair this with the §2.3 ray-density requirement so the cost of coherent fidelity is explicit.

### E. Interposummation positioning

R2.1 is essentially right. Conceding cleanly helps far more than defending.

Show numerical agreement (to stated tolerance) between interposummation and:
1. Range-bin gridding + FFT-based sinc filtering
2. Explicit LFM transmit + matched filter — the machinery already exists in
   `signal_simulation.py:_matched_filter_window` / `make_lfm_window`
3. Direct frequency-domain synthesis `Σ E_k exp(-j4πf z_k/c)` over the CVDomes band, then IFFT

Then revise the paper to state plainly: *interposummation is a gridless, band-limited matched-filter
evaluation; the contribution is the O(N·Z) gridless GPU implementation with no interpolation loss
or gridding ringing — not a new filter.* Quantify the gridding loss avoided (straddle loss vs
nearest-bin assignment) to justify the implementation claim.

---

## Tier 2 — if time allows

### C. RaySAR comparison

Weaker, because the Python port is not a coherent simulator — it sums contribution amplitudes with
an ad-hoc "system response" (`RaySAR_Python/RaySAR_git/para.json`: `responseTh`, `responseDecey`).
The honest comparison is **geometric**, not radiometric.

- **C1 — Geometric structure.** Same scene and geometry; compare shadow length, layover extent, and
  the vehicle–ground dihedral line position. Quote errors in resolution cells.
- **C2 — Compare the scatter clouds, not the images.** RaySAR's `*_Contributions.txt` files are
  (azimuth, range, amplitude, bounce-level) — structurally identical to our energy-range scatter
  plus a bounce index. Comparing that intermediate is far more informative than comparing final
  images, and it isolates ray tracing from image formation.
- **Blocker to check first:** generating new contribution files requires the modified POV-Ray,
  which is **not** in `RaySAR_Python/` (only `application.py`, `main.py`, and pre-computed
  contribution files for car / t-62 / airplane are present). If POV-Ray can't be built, fall back
  to reconstructing those three shipped scenes.

### F. Presentation fixes

- Colorbars, units, and scales on Fig. 1(c) "range per ray" and 1(d) "energy per ray" (R2.4).
- Units column in Table I; add the coordinate-system figure (§1).
- Open-source preparedness (R1.3): LICENSE file, minimal sample script per figure, and a
  Reproducibility section giving the exact config for every headline figure.
- Typo pass (R3).

---

## Recommended minimum scope

If only a subset is feasible:

> **A1 + A3 + B1/B2 + B6 + D + E + the units table (§1)**

That covers every reviewer's central objection: coherent correctness (A1, A3), comparison to an
established simulator (B1, B2), validation of the 2-bounce novelty (B6), computational trade-offs
(D), interposummation framing (E), and reproducibility (§1, F).

## Suggested order of execution

1. **§2.2 check via A1/A2** — build the point-scatterer + plate harness first. Self-contained, no
   CVDomes I/O, and it confirms or refutes the energy-vs-amplitude issue before any large
   investment. *(start here)*
2. §2.1 and §2.3 fixes; ray-density convergence study → becomes the coherent-imaging figure.
3. §1 unit convention → unblocks everything in B.
4. A3 (dihedral/trihedral), A4, A5.
5. B2 (image-level, shared backprojector) → then B1 (phase-history level).
6. B6 multi-bounce ablation.
7. E, D.
8. B3, B7, then C and F.

## Open risks

- **Compute.** B1/B2 at true X-band with 1024²–2048² rays over 480 pulses × 10 vehicles is a real
  workload. Scope the vehicle/aspect count to what fits.
- **RaySAR POV-Ray dependency** may be unbuildable (see C).
- **B3/B7 may show poor agreement.** That is a publishable result if reported honestly with the
  §2.3 ray-density analysis explaining the regime of validity — far better than omitting it.
