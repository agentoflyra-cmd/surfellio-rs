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
