# Video clips

All clips are the live run of the chosen solver setting (8.33 ms time step, 35 iterations per step, CUDA Graphs on) on the
V5 leg mesh (129,783 tetrahedra, digest `d14afb0ae307`), run 2026-10-21: a ±45° sweep of the probe about the limb axis,
then the seven-depth press (3, 7, 11, 15, 19, 23, 25.4 mm). Left: Isaac Sim view; right: the ray-traced B-mode image.

| file | what | as run | notes |
|---|---|---|---|
| `legV5_s2_i35_plain_asrun_18fps.mp4` | the paper's run, every frame at the rate it was computed | 17.6 frames/s mean, 56.4 ms median frame interval | the speed quoted in the paper (one run) |
| `legV5_s2_i35_cutsup_plain_asrun_14fps.mp4` | cut-away view, element colours | 14 frames/s | illustration run; slower because of the cut-away rendering; speed not quoted |
| `legV5_s2_i35_cuttex_plain_asrun_13fps.mp4` | cut-away view, smooth tissue colours (the paper's Fig. 2 frame is frame 408 of this run) | 13 frames/s, 75.6 ms median | illustration run; speed not quoted |

The paper's quantitative results come from the five-run repeats, not from these single picture runs.

**How it was made.** `i4h-phantom/run_scripts/paper_v5_pictures_20261021.sh` ran the three runs; `i4h-phantom/analysis/make_live_video_plain.py` encoded them. `i4h-phantom/run_scripts/example_video.sh` reproduces the plain run (`i4h-phantom/README.md`, section 5).
