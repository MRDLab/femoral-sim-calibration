# 2026-10-21: the live press on Femoral V5 (the cast leg)

Same loop and settings as [`../../V4_results/appendix/video_2026-09-30/`](../../V4_results/appendix/video_2026-09-30/), on V5: no pubic ramus, leg 81 mm long.
`frame_press_19mm.png` is the paper's Fig 3 (`figures/fig_pipeline_press_19mm.png`).

## Same subject, asserted

| check | value |
|---|---|
| finite-element mesh | V5 leg mesh `d14afb0ae307`: 129,783 tetrahedra, s = 20 mm station |
| B-mode scene | `f4dcd823e93b`, 12 bodies (V4: `69c48ab8b52b`, 13 with the ramus); rest set `out/live_render_rest_v5` |
| run | `~/i4h-phantom/out/live_bmode/legV5_s2_i35.{json,log,frames.npz}`; every frame saved (664 of 664) |
| setting | 2 time steps per frame (8.3 ms), 35 iterations, CUDA graph on, ±45° sweep, then the seven-depth press |
| code | phantom `live_press_bmode.py` `abd8bf588c3d` (main checkout), composer `analysis/make_live_video_plain.py` on branch `femoral-v5` |

## Readings of this run (single run: not quotable as a result)

| depth (mm) | 3 | 7 | 11 | 15 | 19 | 23 | 25.4 |
|---|---|---|---|---|---|---|---|
| vein % | 96.5 | 87.8 | 75.3 | 61.9 | 47.7 | 33.8 | 28.3 |
| artery % | 100.8 | 100.9 | 99.7 | 92.9 | 94.0 | 90.1 | 83.4 |

- As run: 17.6 frames per second mean, 56.4 ms median interval (V4 run: 17.6 fps). One run, no repeats.
- ⚠️ At 19 mm the V4 run read vein 39 %, artery 84 %. Both differ, but run-to-run spread is not known for this run type.
  The paper's V5 numbers come from the repeats in `~/i4h-phantom/out/paper_v5/`, not from this picture run.

## What the picture shows, and what it does not

- The dark columns under both vessels are present in the V4 frames too. They are the renderer's behaviour below a lumen, not V5.
- No bone echo appears behind the vessels: V5 has no ramus. The femur lies far deeper than the 48 mm frame.
- The side panel's 14 to 20 % floor is computed (0.8 dx × cut width over this mesh's 76.44 mm² rest area), not measured.
