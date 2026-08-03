# Comparative Analysis Plan — Reviewer Response

Plan for the comparative/validation work needed to address the four reviews in
`todo-list.txt` for *"Open-Source SAR Simulator in PyTorch Via Interposummation."*

Written against:
- `shapenet_sar_sim/` — our simulator
- `Civilian_Vehicle_Data_Domes/` + `/workspace/data/Civilian_Vehicle_Data_Domes/` — CVDomes phase histories
- `/workspace/data/cv_domes_cad_models_ojb_mtl_blend/` — the *same* CAD models as `.obj`
- `RaySAR_Python/` — Python port of RaySAR image formation

---

## 0. Coverage tracker

Each reviewer demand maps to a **specific artifact in the revised paper**, not just to a plan
section. A reviewer checks whether the figure/table exists and says what they asked for; this table
is the checklist for exactly that. Keep the Status column current — it is the revision tracker.

Status: ☐ not started · ◐ in progress · ☑ done · — deferred (with stated reason)

| # | Reviewer demand | Deliverable in the revised paper | Plan § | Status |
|---|---|---|---|---|
| R1.1 | Coherent validation on canonical targets with known phase history | **New Fig. `fig:point-validation`** — aperture-domain phase residual vs analytic `exp(-j4πR(θ)/λ)` for an ideal point scatterer, plus impulse-response cut. **New Fig. `fig:plate-validation`** — measured glint pattern overlaid on PO (A2b). **New Table `tab:canonical`** — plate / dihedral / trihedral / sphere: measured vs closed-form RCS, % error | A1–A3, A5 | ◐ A1 + A2 done — `validation_point_target.py`, `validation_plate.py`; phase residual **3.0e-6 deg RMS**; plate glint matches PO to **0.02%** in −3 dB width. A3/A5 rows of `tab:canonical` outstanding |
| R1.2 | Quantitative metrics (peak location, resolution, sidelobes) under RaySAR/CVDomes conditions | Columns in **`tab:canonical`**: peak-location error (px), −3 dB mainlobe width vs `c/2B` and `λ/2Δθ`, PSLR vs −13.26 dB, ISLR. Measured at the CVDomes band of §3, not in normalized units | A1, A4, B2 | ◐ all five metrics implemented and measured at baseline; plate glint PSLR **−13.42 dB** vs −13.26 reference; still to redo at the CVDomes band |
| R1.3 | Units, coordinate system, scale, reproducibility | **Units column added to `tab:params`** (line 449); **new Table `tab:units`** — scene unit ↔ m, λ in m and GHz, `spatial_bw`/`spatial_fs` in m⁻¹ with the `B_t`/`F_t` in Hz they derive from; **new coordinate-system panel in `fig:terminology`** (line 173); **new Reproducibility section** before Conclusion | §1, F | ☐ |
| R2.1 | Position interposummation against classical matched filtering | **Rewritten claim in §Method** (interposummation = gridless band-limited matched-filter evaluation); **new Fig. `fig:interposum-equiv`** — agreement vs range-bin+FFT, vs explicit LFM matched filter, vs direct frequency-domain synthesis; **straddle-loss plot** quantifying what gridding costs | E | ☐ |
| R2.2 | Quantitative *and* qualitative comparison to an established simulator | **New Fig. `fig:cvdomes-image`** — our image ‖ CVDomes image, same backprojector, 70 dB, 10 m × 10 m; **new Table `tab:cvdomes-metrics`** — NCC(dB), SSIM, top-N peak displacement in resolution cells, per vehicle; **new Fig. `fig:cvdomes-phase`** — per-pulse complex correlation | B1, B2, C, D | ☐ |
| R2.3 | *Explain why* complex-valued imaging performs poorly | **New §Coherent fidelity** stating the ray-grid Nyquist rule `Δ_ray < λ/(4 sin θ)`; **new Fig. `fig:ray-density`** — RCS error vs rays-per-phase-cycle crossing the limit against PO, showing the Fig. 8 "interference" as aliasing (A2e(b), generated); plus the §2.2 radiometric-scaling note. **Also new Fig. `fig:demod-order`** (A1d, generated) | §2.1–2.3, **2.5**, A1, A2 | ◐ **causes measured** — §2.5 demod order, §2.3 ray aliasing (+14.6 dB past the limit); §2.1 fixed; §2.2 quantified as a units/interpretation point. Needs writing up |
| R2.4 | Colorbars/units on Fig. 1(c),(d) | **`fig:main-fig`** (line 97) regenerated with colorbars + units on the range-per-ray and energy-per-ray panels | F | ☐ |
| R3 | More discussion of coherent imaging | Same §Coherent fidelity section as R2.3; plus one sentence at `imaging_algorithms.py:169` behaviour (CBP returns magnitude) | §2, A1 | ◐ material now exists (§2.5–2.7); needs writing up |
| R4 | *Any* verification of correctness; benchmarks | Everything under R1.1/R1.2, **plus new Fig. `fig:bounce-ablation`** (1-bounce vs 2-bounce scored against CVDomes) and **new Table `tab:compute`** (s/phase-history, GPU vs CPU, scaling, peak memory, vs RaySAR/CVDomes) | A, B6, D | ◐ A1 + A2 are the first verification of correctness in the repo, and A2 is the first against an *exact closed form through the full pipeline* |

**Reading of the reviews.** R1 and R4 will likely reject again without the R1.1 row — it is the
single load-bearing deliverable. R2 will likely reject again without the R2.2 and R2.1 rows.
R2.3/R3 are cheap by comparison and are already half-answered by the §2 code work.

**Gap check.** Every row above produces at least one new numbered figure or table. If a row's
Status is ☑ but no artifact in `main.tex` carries that label, the row is not actually done.

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

### 2.1 Per-pulse normalization by return count — ☑ FIXED

Was `signal / R` in `interpolate_signal`, where `R = scatter_z.shape[-1]`. Because
`render_images.py:123-131` passes `all_ranges[t][p]` — **only the hit rays** — `R` was the number of
returns *for that pulse* and varied pulse to pulse with aspect, imposing an artificial
aspect-dependent amplitude taper across the synthetic aperture.

The division has been removed; `interpolate_signal` now returns `signal, sample_z` unmodified
(`signal_simulation.py:330-333`). Aperture taper is gone and the RCS-vs-aspect comparison (**B3**)
is no longer confounded.

**Remaining caveat — ray-density scaling.** With no normalization at all, signal magnitude scales
with the *number of hit rays*, i.e. roughly quadratically in `n_ray_width`. Two consequences:

- The §2.3 ray-density convergence study will measure amplitude growth, not convergence, unless
  normalized. Divide by the **constant** total transmitted ray count `n_ray_width * n_ray_height`
  (or equivalently by per-ray solid angle) — constant in the pulse index, so aspect-invariance is
  preserved, while also making results invariant to ray density.
- Absolute scale is now arbitrary, so any comparison to CVDomes must either be scale-invariant
  (NCC, SSIM, peak locations — as specified in **B2**) or carry an explicitly fitted calibration
  constant. State which, and fit it once globally rather than per image.

### 2.2 Radiometric scaling of the returned energy — MEASURED (A2d)

A1 could not probe this (one scatterer makes any monotone reweighting a pure scale factor); A2 can.
Two measurements of the current model against closed form, both through the full ray-traced
pipeline:

**(a) Scaling law.** `s` is a fraction of incident *power* (from the `R+S+A = 1` budget), so a
linear-in-`s` RCS is what the material parameterization implies. Measured log-log slope of `σ̂`
vs `s` for a flat plate: **2.0000**.

**(b) Contrast.** Two plates at different ranges, `s = 1.0` and `s = 0.1`, i.e. `10·log10(s₁/s₂)`
= 10.00 dB apart. Measured range-profile contrast: **20.00 dB**. Contrasts come out **doubled in
dB** — a scene with 30 dB of separation between materials renders with 60 dB.

**The model is self-consistent**; what this fixes is the *interpretation* of `(R,S,A,I,D)`. They
behave as field-amplitude fractions, not power fractions, so `s = 0.1` is a −20 dB material and
not a −10 dB one. Two places this has to be carried through:

- **B7 calibration.** Fit and report the values on that understanding, or the fitted materials
  will not mean what the `R+S+A = 1` derivation in the paper says they mean.
- **B2 metrics.** Any NCC/SSIM on dB images compares against a CVDomes image whose dB axis is not
  stretched the same way. Either state the convention, or compare on a scale where it cancels.

The energy equation itself is the author's; this section records what it measures, not a proposed
change to it.

§2.5 is a *separate and larger* effect on coherent imaging; this one scales amplitude, that one
destroys cross-range resolution.

Sign check (this part is fine): simulation applies `exp(+j·2π·z_2way/λ)`; `imaging_algorithms.py:56`
compensates with `exp(-4jπ·sample_z/λ)` on the one-way `sample_z`. Consistent.

### 2.3 Unstated phase-sampling Nyquist condition on the ray grid — MEASURED (A2e)

Adjacent rays must differ in two-way range by much less than λ/2:

```
δ = 2 · Δ_ray · sin θ  <  λ/2      ⇒      Δ_ray < λ / (4 sin θ)
```

so, worst case (grazing),  **Δ_ray < λ/4**. Equivalently: **more than 2 rays per phase cycle**
across the object, since rays/cycle = `λ / (2 Δ_ray sin θ)`.

**A2e(b) crosses the limit against an exact reference.** Aspect held at 4°, wavelength shrunk so
the cycle count across a flat plate grows while the ray grid does not, each λ chosen to land on a
PO sidelobe peak (away from nulls). Error is the simulator's normalized RCS minus physical optics:

| rays / phase cycle | 101 | 33.9 | 11.3 | 5.98 | 4.06 | 2.48 | 2.07 | 1.67 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|
| error vs PO (dB) | 0.04 | 0.04 | −0.00 | −0.12 | −0.33 | −1.19 | −2.01 | −4.83 | **+14.63** |

The sign flip at the end is the signature of aliasing: below one ray per cycle, energy folds back
and the simulator reports a return **14.6 dB too large** at an aspect where PO says there is almost
nothing. The paper's "interference patterns from constructive and destructive summation across
nearby rays" is **phase aliasing**, not physics.

**Refined design rule (worth stating in the paper, since it is what the measurement supports):**
`Δ_ray < λ/(4 sin θ)` is the *aliasing threshold*, where error is already ≈1 dB. For RCS accurate to
0.5 dB use twice the density, **Δ_ray < λ/(8 sin θ)** (≈4 rays/cycle); at 11 rays/cycle the
agreement with PO is exact to 0.00 dB.

Current defaults: `grid_width = 1.2`, `n_ray_width = 128` → `Δ_ray = 0.0094`, `λ = 0.5`. Margin ≈ 13×,
so the baseline is safe. **The wavelength sweep (Fig. 8) crosses the limit.**

This is a real answer to R2.3 and R3 — and a genuine contribution once stated as a design rule with
a measured error curve rather than as an artifact.

**Consequence for the CVDomes comparison, state it honestly:** at λ = 3.12 cm on a 4.5 m vehicle,
Δ_ray < 7.8 mm → **≈ 600² rays minimum**, realistically **1024²–2048²**. The current 128² is nowhere
near coherent-fidelity requirements at true X-band. Budget compute accordingly; report the
requirement rather than letting a reviewer find it.

### 2.5 `projected_CBP` demodulates before interpolating — MEASURED, this is the R2.3 answer

`imaging_algorithms.py:52` multiplies the **sampled** signal by `exp(-j4π·sample_z/λ)`,
then `CBP_2D` sinc-interpolates it. Demodulation shifts the spectrum from baseband to a band of
width `B_s` centred at `2/λ`, so band-limited reconstruction at rate `F_s` requires

```
2/λ + B_s/2 < F_s/2      ⇒      F_s > B_s + 4/λ
```

With the paper default `F_s = B_s` this **can never be satisfied** — the carrier is always partly
aliased. Measured consequence (`validation_point_target.py`, A1d): `projected_CBP` cross-range
resolution is **0.02687 for every λ from 0.02 to 0.5** — a 50× sweep with no response at all. It
gains nothing from the carrier and is effectively a non-coherent tomographic reconstruction.

Two controls confirm the cause:

| Imager | λ=0.02 cross-range | carrier limit λ/(4 sin(Δθ/2)) |
|---|---|---|
| reference BP, demod **after** interpolation | **0.00673** | 0.00707 ✓ |
| reference BP, demod **before** interpolation | 0.03378 | — |
| `projected_CBP`, `F_s/B_s` = 1 / 2 / 4 | 0.02687 / 0.01241 / **0.00646** | recovers once `F_s > B_s + 4/λ` = 273 |

Two fixes, both verified: demodulate **after** interpolation using the pixel's own range — which
`strip_map_imaging` already does correctly (`imaging_algorithms.py:244-256`) — or keep the order
and set `F_s ≥ B_s + 4/λ`. The first is free; the second costs samples.

This supersedes §2.3 as the primary explanation for the paper's wavelength sweep (Fig. 8): the
degradation at short λ is carrier aliasing in the **imager**, and only possibly also ray-grid
aliasing in the simulator. Both conditions should be stated.

### 2.6 The whole pipeline is far-field — CORRECTED by A2f; a limitation, not a displacement bug

`imaging_algorithms.py:63` computes `sample_r = sample_z - |trajectory|`, a far-field projection.
A1b fed that imager a *spherical* phase history and measured a peak displacement following
`ρ²/(2D)`, reaching **4.9 resolution cells at ρ = 0.45**. That number describes the imager in
isolation and **overstates the end-to-end consequence**, because the simulator is far-field too:
`accumulate_scatters.py:260-265` sets `total_range = 2·dot(hit − trajectory, forward)`, a
plane-wavefront range, and `projected_CBP` inverts exactly that.

A2f runs a small diffuse patch through the *real* pipeline — ray tracer through image — at
increasing radius:

| radius ρ | 0.00 | 0.15 | 0.30 | 0.45 |
|---|---|---|---|---|
| peak error (range cells) | 0.062 | 0.019 | 0.025 | **0.068** |
| A1b spherical-history prediction | 0.000 | 0.547 | 2.188 | 4.924 |

Residuals are at the 0.5-pixel quantization floor of the 512² image at every radius. **The forward
and inverse models agree, so ray-traced targets land where they belong.**

What remains true, and is what the paper should say: the simulator is a **far-field simulator
throughout**, so `ρ²/(2D)` is its deviation from true spherical propagation — a stated modelling
assumption with validity bound `ρ < sqrt(2D/(B_s cos el))` ≈ 0.20, not an internal inconsistency.
This is also the right framing for **B2**: CVDomes' own reference imager is
`bpBasicFarField.m`, far-field by name and construction, so the comparison is like for like.

### 2.7 The `|r|` ramp filter costs ~9 dB of range PSLR — MEASURED

`CBP_2D` multiplies the spectrum by `|r|` (Radon inversion). Measured at the origin (A1c):

| | range PSLR | range ISLR | range −3 dB (×predicted) |
|---|---|---|---|
| with `\|r\|` ramp (current) | −5.09 dB | −2.35 dB | 0.01257 (×0.795) |
| ramp disabled | **−14.28 dB** | −12.71 dB | 0.015611 (**×0.987**) |
| sinc reference | −13.26 dB | | |

Without the ramp the impulse response matches closed form almost exactly. The ramp is the correct
inversion for a full 360° tomographic aperture; for a limited-angle spotlight aperture whose range
profile is already matched-filtered it is over-sharpening — narrower mainlobe bought with 9 dB of
sidelobes. Either drop it, apodize it (Shepp-Logan/Hann), or justify it explicitly.

Separately, using the range coordinate `r` as a stand-in for the frequency axis is only valid because
both are uniform ramps symmetric about zero — true for the odd-length `sample_z` grid
`interpolate_signal` builds, but silently wrong for an even-length grid. Now commented in-place.

### 2.8 Geometric-optics RCS has no λ⁻² factor — MEASURED (A2c), a limitation to declare

At specular, every ray on a plate normal to the line of sight returns at the same range, so the
coherent sum is wavelength-independent *by construction*. Physical optics says
`σ = 4πA²/λ²`. Measured log-log slope of `σ̂` vs λ over a 16× sweep:

| | slope vs λ |
|---|---|
| simulator | **0.0000** |
| physical optics | −2.0000 |

The missing factor is aperture diffraction, which geometric optics does not carry. The *shape* of
the response is right (§A2b below: the pattern matches PO to 0.02%), only the absolute level is
λ-dependent in a way the simulator does not reproduce.

Consequences, all of which are cheaper to state than to have found:

- Any absolute RCS calibration constant is **λ-dependent** — it must carry a `1/λ²`. Fit it that
  way in **B7**, or the fitted materials will not transfer across bands.
- Cross-wavelength comparisons of *absolute* image brightness are meaningless as-is; normalized or
  scale-invariant metrics (NCC, SSIM, peak locations — as **B2** already specifies) are unaffected.
- This is a property of every GO/ray-based simulator, RaySAR included. Say so; it positions the
  limitation as a known modelling class rather than a bug.

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
| A1 ☑ | Single ideal point scatterer at known (x,y,z) | analytic `exp(-j4πR(θ)/λ)` | **Phase residual across the aperture** (the direct evidence R1 demands); peak location error (target < 0.1 px); mainlobe width vs `c/2B` and `λ/2Δθ`; PSLR vs −13.26 dB; ISLR |
| A2 ☑ | Flat plate | PO: `σ = 4πA²cos²θ/λ² · sinc²(·)` | Specular glint mainlobe width = λ/L; RCS vs aspect curve overlay |
| A3 | **Dihedral and triangular trihedral** | `σ_dih = 8πa²b²/λ²`, `σ_tri = 4πL⁴/3λ²` | **Validates the 2-bounce path — the headline claim.** Also: double-bounce return must land at the correct phase-center range (the corner line) |
| A4 | Two points separated by exactly one resolution cell | Rayleigh criterion | Resolution is real, not nominal |
| A5 | Sphere (sweep already exists, Fig. 13) | `σ = πa²`, aspect-independent | Make the existing figure quantitative: measured RCS vs `πa²` across the five scales |

A1 and A3 are the minimum. A1 is the single most important experiment in the revision — it is the
literal thing R1 asked for and the thing R4 says is entirely missing.

**A1 and A2 are done** (`validation_point_target.py`, `validation_plate.py`). What they *verified as
correct* is as load-bearing for the rebuttal as the defects they found, so keep it together:

| Claim now backed by measurement | Result |
|---|---|
| Interposummation reconstructs the analytic phase history | phase residual **3.0e-6 deg RMS**, amplitude ratio 0.99922 (A1) |
| Coherent ray sum reproduces the PO glint pattern | −3 dB width **3.1705° vs 3.1712°** at L/λ=8; **1.5883° vs 1.5861°** at L/λ=16 (A2b) |
| …including its sidelobe structure | measured angular PSLR **−13.42 / −13.30 dB** vs the sinc reference −13.26 dB, tracking PO past 5 sidelobes to −30 dB (A2b) |
| Plate RCS scales as area² | log-log slope **1.978** vs PO's 2 (A2a) |
| Ray-count normalization is density-invariant | `σ̂` flat to ≈2% from 256² to 1024² rays; no systematic trend (A2e(a)) |
| Ray-traced targets land at the right pixel | ≤ **0.068 range cells** out to ρ = 0.45, at the image quantization floor (A2f) |

That A2b overlay is the strongest single validation artifact in the revision — an *exact* closed-form
reference matched to 0.02% in mainlobe width and to within 0.2 dB in sidelobe level, from the full
ray-traced pipeline. It should be a figure (`fig:plate-validation`), not just a table row.

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

1. ☑ **§2.2 check via A1/A2** — point-scatterer and plate harnesses built
   (`validation_point_target.py`, `validation_plate.py`). §2.2 confirmed, §2.3 measured, §2.6
   corrected, §2.8 found; A2b/A2d/A2e figures generated.
2. ◐ §2.1 fixed. §2.3 measured (A2e(b)) — the ray-density figure exists; the remaining piece is
   rerunning it on a *vehicle* so the paper's Fig. 8 regime is covered, not just a plate.
   **Decision needed before B:** whether to change the §2.5 demod order, which changes every
   existing figure. The fix is verified but not enabled.
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
