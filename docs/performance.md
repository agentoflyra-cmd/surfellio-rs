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
