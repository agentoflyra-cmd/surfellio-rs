# Regression check

`scripts/regression_check.py` runs the pinned development baseline and compares
trajectory accuracy plus map/latency statistics against
`scripts/regression_baseline.json`.

```bash
# refresh the baseline (after an intentional, reviewed change)
python3 scripts/regression_check.py --write-baseline

# check the current working tree against the committed baseline
python3 scripts/regression_check.py

# subset
python3 scripts/regression_check.py --bags KA,FR
```

Bags and configs are pinned in the script:

| bag | config | metric |
|---|---|---|
| KA_Urban_East | mid360_FR_IOSB_Short.yaml | ATE / e2e (LiLi-OM + GNSS reference) |
| FR_IOSB_Short | mid360_FR_IOSB_Short.yaml | ATE / e2e (in-bag GNSS) |
| rosbag2_upstairs | mid360.yaml | 3D ATE / z RMSE (FAST-LIO-SAM reference) |
| yuanqu_with_odom | mid360.yaml | 3D ATE / z RMSE (FAST-LIO-SAM reference) |

Map/latency statistics are parsed from `--map-stats`: duplicate coplanar pairs,
merged/compacted totals, frame latency mean/p50/p99.

Tolerances: accuracy metrics +5%; duplicate pairs +15%; p50/p99 latency +20%.
Exit code is non-zero when any metric regresses.

Reference trajectories, raw bags and per-run outputs live under `output/` and
`rosbags/` (both gitignored); the script assumes they are present on the machine.
Per-bag replay stdout is saved to `output/regression/<bag>.log` for diagnosis.
