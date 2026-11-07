# Revisit diagnostic packages

`scripts/revisit_packages.py` generates a self-contained visualization package
for each revisit candidate sample, so a loop measurement can be judged against
the historical map without re-running the whole pipeline.

```bash
python3 scripts/revisit_packages.py --dataset corridor
python3 scripts/revisit_packages.py --dataset ka --samples 3
```

Packages are written to `output/revisit_packages/<dataset>/sample_XX_i<i>_j<j>/`
with:

| file | content |
|---|---|
| `topdown_xy.png` | historical map + current scan at the reference alignment (green) and at the GICP alignment (red) |
| `slices_xz.png` | three x-z cross sections around the overlap |
| `virtual_view_ref.png` | real range / virtual range / range error / free-space violations at the reference pose |
| `virtual_view_gicp.png` | same at the GICP pose |
| `metrics.json` | point-to-plane RMSE, valid fraction, overlap, range agreement, free-space violation, local height difference (mean/p90), GICP fit |
| `summary.md` | automatic conclusion: which alignment is more self-consistent under the historical-map constraint |

`output/revisit_packages/<dataset>/ALL_SUMMARIES.md` collects all conclusions.

Datasets are configured in `DATASETS` inside the script:

- `ka`: keyframes/surfels from the KA diagnostics dumps, reference alignment from
  the LiLi reference trajectory (metre-level weak reference);
- `corridor`: reference alignment from the virtual-view-verified pose-graph
  corrected trajectory (`corridor_system_poses_vview.npz`), the positive-control
  dataset.

Dependencies: numpy and matplotlib (same environment as the other offline
diagnostics).
