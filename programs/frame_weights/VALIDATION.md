# estimate_frame_weights: validation

Checked 2026-11-24 against the study the estimator was ported from. That study
measured per-frame B-factors at refined particle poses with a null of 1,024 random
orientations per particle.

## Data

- One micrograph of the yeast ribosome large subunit (LSU) in situ, at 300 kV.
- 54 frames over 54.5 e/Å², 0.936 Å/px.
- The frames were already aligned; the run used `aligned_movie_path`.
- 407 particles refined with Leopard-EM, at their refined poses and defocus.
- The search template: ttsim3d's G&G dose filter over 0–50 e/Å², with `rel_diff`.
- Settings: `reference_frame: 5` (as the study), 200 bootstraps over particles, and
  defaults otherwise.

## Correlation pieces

- **Against the study.** The pieces (per particle, frame, position and bin) equal the
  study's to 2e-5 relative (float32).
- **Against Leopard-EM.** Summed over frames and bins, they reproduce Leopard-EM's own
  correlation of the summed box (`do_batched_orientation_cross_correlate`) to
  2–3e-7.
- **Image boxes.** The frames' summed box correlates 0.99998 with Leopard-EM's
  whitened particle box.

## The analytic null against random orientations

`null_model: analytic` against `null_model: random` (1,024 random orientations per
particle, the study's null):

| | analytic | random |
|---|---|---|
| per-frame null variance, ratio to random (12 particles, median per bin, 72–2.3 Å) | 0.96–1.08 | 1 |
| random-orientation null mean (per bin, in its sd) | — | ≤ 0.03 |
| median null sd | 0.338 | 0.346 |
| last frame's noise, relative to the mean frame | 1.828 | 1.833 |
| largest B difference, in bootstrap SDs | 0.18 | — |
| largest scale difference, in bootstrap SDs | 0.31 | — |
| run time (one GPU, shared) | 127 s | 1,072 s |

## Per-frame B and scale against the study

The analytic null, against the study's `bfactor_<set>.json`. The study's values are
in brackets.

| Frame | B (Å²) | scale |
|---|---|---|
| 0 | −66.2 ± 5.8 (−65.9 ± 5.9) | 0.936 ± 0.024 (0.940) |
| 1 | −20.5 ± 3.6 (−20.4 ± 3.4) | 0.997 ± 0.018 (1.000) |
| 2 | −4.0 ± 2.5 (−3.5 ± 2.5) | 1.000 ± 0.015 (1.000) |
| 5 | −3.2 ± 2.5 (−3.3 ± 2.7) | 0.971 ± 0.023 (0.980) |
| 7 | −7.9 ± 3.1 (−7.7 ± 3.0) | 0.945 ± 0.022 (0.945) |
| 10 | −12.6 ± 3.4 (−12.3 ± 3.3) | 0.914 ± 0.024 (0.915) |
| 20 | −69.8 ± 6.7 (−69.7 ± 6.4) | 0.821 ± 0.027 (0.820) |
| 30 | −195.0 ± 20.5 (−194.3 ± 21.3) | 0.777 ± 0.033 (0.779) |
| 40 | −372.2 ± 69.9 (−373.7 ± 68.5) | 0.679 ± 0.050 (0.685) |
| 53 | −450.5 ± 126.3 (−462.5 ± 120.8) | 0.444 ± 0.050 (0.449) |

- **Every frame agrees.** All 54 frames' B are within 0.20 of the study's bootstrap SD
  (within 0.16 with the random null), and every scale within 0.37 SD.
- **The fit's inputs agree too.**
  - Bins fitted: the same, 33.7–2.89 Å.
  - χ²/dof: 1.035, against the study's 1.040.
  - Bootstrap SDs: the same (median ratio 0.997).

## Template envelope

- **Matched envelope.** Relative to the search template it equals the study's to
  0.1 % over 50–2.9 Å.
- **The 5-knot correction.** It has the study's shape: a dip near 20 Å (0.91), a peak
  at 4–5 Å (1.81 relative to 40 Å), and a fall towards 3 Å. It agrees within 8 %.
  - The study fitted the curve to Leopard-EM's normalised z, on a sum at G&G's
    per-bin power.
  - Here it is fitted to the analytic z on the table-weighted sum, with that sum's
    own whitening.
- **In-sample mean z of the particles on the table-weighted sum:**

  | Template | mean z |
  |---|---|
  | as is | 11.04 |
  | matched | 11.34 (+2.7 %) |
  | matched + correction | 11.41 (+3.3 %) |

## Synthetic data

`tests/test_estimate_frame_weights.py` pastes particles into a movie with Leopard-EM's
own projections, each frame damaged by a known B and C. The estimator recovers both
within their bootstrap errors. For example, frame 0 gives −19.2 ± 5.1 Å² (true −20)
and the last frame −180.2 ± 12.3 Å² (true −180).

- **A limitation, shared with the study's null.** Each frame's null variance comes
  from the particle box's own power.
- In real frames that power is almost all noise: on the data above, the frames'
  relative noise is 0.97–0.99 apart from the last frame.
- **If one frame's box were signal-dominated** in the fitted bins (very large particles
  at low resolution), the frames with the most signal would be down-weighted, and
  the fitted scales compressed.
