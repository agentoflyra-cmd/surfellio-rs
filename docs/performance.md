# Per-frame performance instrumentation

Replay now reports per-stage wall-clock timings (`--map-stats`) and the IEKF
iteration count actually executed (the `min_delta_norm` early exit makes the
`max_iterations` cap a rare event).

## Spires 2024-03-18-christ-church-01 (7958 frames, 10 Hz Ouster, 64 lines)

| stage | mean | p99 |
|---|---:|---:|
| predict | 0.11 ms | 0.12 ms |
| deskew | 2.10 ms | 2.87 ms |
| preprocess | 1.44 ms | 1.99 ms |
| IEKF update | 65.31 ms | 123.88 ms |
| — association | **60.21 ms** | **117.63 ms** |
| — linear solve | 5.10 ms | 8.25 ms |
| map insert | 16.93 ms | 28.13 ms |
| **frame total** | **86.89 ms** | **157.16 ms** |

IEKF iterations: mean 1.83, p50 2, p99 3, max 10 (early exit works).

Conclusion: the frame budget is dominated by scan-to-map **association**
(~60 ms, 69% of the frame) and map insertion (~17 ms). Iterating the filter or
the linear solve is not the bottleneck; association sampling and map-size /
neighbour-search cost are.

## Insert-path micro-optimisation

`SurfelMap::insert` used a per-point `HashSet<SurfelID>` to de-duplicate
resolved roots. A diagnostic counter showed **0 duplicate roots** on both
corridor_04 (1396 frames) and Spires (7958 frames), even with tombstones
present, because `search_radius == 0` means a bucket never yields two ids with
the same root in practice. Removing the set (re-scoring a duplicate root would
not change which root wins anyway) gives:

| | map insert mean | map insert p99 | frame mean |
|---|---:|---:|---:|
| corridor_04 before | 0.92 ms | — | 6.68 ms |
| corridor_04 after | **0.63 ms** | 1.70 ms | 6.46 ms |
| Spires before | 16.93 ms | 28.13 ms | 86.89 ms |
| Spires after | **11.23 ms** | 17.83 ms | **81.38 ms** |

Remaining frame cost is dominated by association (~60 ms on Spires).

## Cross-iteration association reuse

The first IEKF iteration caches the associated `SurfelObservation` for every
point. Later iterations keep re-linearising (new pose -> new residual and
Jacobian) but reuse the cached correspondence; a point is re-queried only when
its plane residual exceeds 0.10 m, and cached misses are always re-queried so
recall cannot drop. Controlled by `FASTLIO_ASSOC_REUSE` (**opt-in for now**:
the four-bag regression showed it degrades KA e2e 3.54 -> 4.59 m and
upstairs/yuanqu z_rmse, so the re-association threshold needs tuning before it
can be enabled by default).

Measured correspondence-change rate (first vs final iteration, diagnostic mode):
corridor_04 2.44%, Spires 3.65%.

| Spires (7958 frames) | baseline | reuse |
|---|---:|---:|
| IEKF association | 59.74 ms | **34.72 ms** |
| IEKF update | 64.80 ms | 39.79 ms |
| frame total | 80.65 ms | **56.03 ms** |
| IEKF iterations | 1.83 | 1.84 |
| surfels | 1,276,645 | 1,278,179 |
| ATE vs hba reference | 0.966 m | **0.947 m** |
| loop gap (horizontal) | 1.963 m | 1.995 m |

Trajectory difference baseline vs reuse: median 5.1 cm / p90 7.8 cm / max 9.2 cm.

### Regression gate

With `FASTLIO_ASSOC_REUSE` enabled the four-bag check fails on KA.e2e (4.59 vs
3.54 m), upstairs.ate/z_rmse and yuanqu.ate/z_rmse, so the feature stays opt-in
until the re-query threshold / iteration interaction is understood. The
correspondence-reuse code path, diagnostics and timing numbers above remain
valid for A/B experiments.

### Correspondence stability is dataset-dependent

Change rate of the per-point association between the first and the final IEKF
iteration (`FASTLIO_ASSOC_CHANGE_DIAG`):

| dataset | overall | ground | wall |
|---|---:|---:|---:|
| corridor_04 | 2.44% | — | — |
| Spires christ-church-01 | 3.65% | — | — |
| upstairs | 6.62% | 5.74% | 6.97% |
| yuanqu | **16.02%** | 14.17% | 17.80% |

Ground correspondences are not less stable than wall ones; the whole reuse risk
scales with the dataset's overall instability (Spires 3.65% -> works, yuanqu
16% -> does not).

### Ground/wall reuse A/B (reuse opt-in, mode selectable)

`FASTLIO_ASSOC_REUSE_MODE=walls|ground` restricts reuse to one category;
`FASTLIO_REASSOC_M` sets the re-query plane residual (default 0.10 m).

yuanqu (baseline ate 0.1518, z_rmse 0.1143, latency p50 41.7 ms):

| mode | ate | z_rmse | latency p50 |
|---|---:|---:|---:|
| walls, 0.10 m | 0.1520 | 0.1348 (+18%) | 23.6 ms |
| ground, 0.10 m | 0.1549 | 0.1230 (+7.6%) | 29.5 ms |
| ground, 0.05 m | 0.1567 | 0.1263 (+10.5%) | 31.9 ms |
| ground, 0.02 m | 0.1602 | 0.1276 (+11.6%) | 36.4 ms |

upstairs ground/0.05: ate 0.0916 (+11.7% vs 0.0820), z_rmse 0.0488 (+70%).

Conclusion: neither category nor a tighter re-query threshold removes the
degradation on the short indoor bags; reuse stays opt-in and is only known-safe
where the measured correspondence change rate is low (Spires). A future adaptive
gate would have to estimate stability cheaply (e.g. iteration pose delta).
