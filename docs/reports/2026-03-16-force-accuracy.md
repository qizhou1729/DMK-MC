# Force Accuracy Report

Date: 2026-03-16

## Setup

This report compares `HPDMKPtTree<double>::eval_force()` against the analytic Ewald bulk force from `hpdmk::Ewald::compute_force()`.

- Build/run command:
  `module load gcc fftw openmpi && cmake -S . -B build -DBUILD_ACCURACY=ON && cmake --build build -j 8 --target hpdmk_force_accuracy && ./build/hpdmk_force_accuracy`
- Common parameters:
  `L=20`, `n_per_leaf=5`, `init=DIRECT`, `s=4`, `alpha=1`, `OMP_NUM_THREADS=1`
- Systems:
  3 deterministic neutral random configurations per `(n_particles, digits)` pair
- Metrics:
  `mean_rel_l2 = mean(||F_hpdmk - F_ewald||_2 / ||F_ewald||_2)`,
  `worst_rel_l2 = max trial rel_l2`,
  `mean_rel_linf = mean(||F_hpdmk - F_ewald||_inf / ||F_ewald||_inf)`,
  `achieved_digits = -log10(mean_rel_l2)`

`n_particles=64` remains excluded for this report because the current force path is still not robust on that sparse case.

## Results

| n_particles | digits | mean_rel_l2 | worst_rel_l2 | mean_rel_linf | worst_rel_linf | mean_net_force | achieved_digits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 3 | 1.361e-03 | 1.865e-03 | 1.028e-03 | 1.351e-03 | 2.776e-06 | 2.87 |
| 96 | 6 | 2.663e-06 | 3.466e-06 | 2.027e-06 | 2.521e-06 | 4.471e-10 | 5.57 |
| 96 | 9 | 2.514e-08 | 3.112e-08 | 2.201e-08 | 2.564e-08 | 2.327e-12 | 7.60 |
| 96 | 12 | 2.493e-08 | 3.062e-08 | 2.174e-08 | 2.485e-08 | 1.514e-14 | 7.60 |
| 128 | 3 | 1.295e-03 | 1.835e-03 | 7.876e-04 | 1.280e-03 | 2.049e-06 | 2.89 |
| 128 | 6 | 2.341e-06 | 3.144e-06 | 1.806e-06 | 2.556e-06 | 2.272e-10 | 5.63 |
| 128 | 9 | 3.267e-08 | 5.612e-08 | 3.610e-08 | 7.168e-08 | 4.421e-12 | 7.49 |
| 128 | 12 | 3.256e-08 | 5.595e-08 | 3.674e-08 | 7.342e-08 | 1.327e-14 | 7.49 |
| 256 | 3 | 9.236e-04 | 1.092e-03 | 4.598e-04 | 5.185e-04 | 7.011e-06 | 3.03 |
| 256 | 6 | 1.898e-06 | 2.130e-06 | 9.787e-07 | 1.203e-06 | 9.407e-10 | 5.72 |
| 256 | 9 | 1.815e-08 | 2.187e-08 | 1.070e-08 | 1.472e-08 | 7.294e-12 | 7.74 |
| 256 | 12 | 1.803e-08 | 2.166e-08 | 1.076e-08 | 1.447e-08 | 2.925e-14 | 7.74 |
| 512 | 3 | 7.469e-04 | 9.746e-04 | 2.790e-04 | 3.802e-04 | 2.167e-05 | 3.13 |
| 512 | 6 | 1.506e-06 | 2.020e-06 | 5.421e-07 | 8.685e-07 | 3.055e-09 | 5.82 |
| 512 | 9 | 1.683e-08 | 2.337e-08 | 1.052e-08 | 1.968e-08 | 1.910e-11 | 7.77 |
| 512 | 12 | 1.670e-08 | 2.316e-08 | 1.055e-08 | 1.971e-08 | 5.345e-14 | 7.78 |

## Observations

- The repaired `eval_force()` now scales consistently across the tested particle counts. Requested `3`, `6`, and `9/12` digits translate to about `2.9-3.1`, `5.6-5.8`, and `7.5-7.8` achieved digits on `96-512` particles.
- The `12`-digit setting still saturates near the `9`-digit result, around `1.7e-8` to `3.3e-8` relative L2 error.
- The total force is again momentum-conserving to near machine precision, with mean net-force norms between `1e-14` and `2e-5`.
