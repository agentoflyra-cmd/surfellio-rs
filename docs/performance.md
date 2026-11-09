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
recall cannot drop. Controlled by `FASTLIO_ASSOC_REUSE` (on by default, `=0`
disables).

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
