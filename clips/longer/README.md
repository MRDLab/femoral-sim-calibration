# Video clips: longer setting

The same live run as `../final_run/` (V5 leg mesh, digest `d14afb0ae307`): a ±45° sweep of the probe about the limb
axis, then the seven-depth press (3, 7, 11, 15, 19, 23, 25.4 mm). Here the solver uses a slower setting from Table S4:
**2.08 ms time step, 140 iterations per step**. Run 2026-11-05, one run per view.

| file | what |
|---|---|
| `comparison/final_vs_longer_plain.mp4` | chosen setting (top) against this setting (bottom), on one clock of computer time |
| `comparison/final_vs_longer_cuttex.mp4` | the same, cut-away view, smooth tissue colours |
| `comparison/final_vs_longer_cutsup.mp4` | the same, cut-away view, element colours |
| `alone/legV5_s8_i140_plain_asrun_2fps.mp4`, `…_cuttex_plain_asrun_2fps.mp4`, `…_cutsup_plain_asrun_2fps.mp4` | this setting by itself, each frame at the rate it was computed (2 frames/s) |

**How the comparison clips are timed.** The video plays at the computer's own pace, at normal speed. Each panel shows
only the newest 3-D view and ultrasound image the computer had finished at that moment. Nothing is resampled or
interpolated. ⚠️ Neither row is real time; each row prints its measured speed.

| setting | median frame interval | simulated motion | last image | speed |
|---|---|---|---|---|
| chosen (8.33 ms, 35 iterations) | 56.3 ms | 11.20 s | 38.1 s | 0.29× real time |
| longer (2.08 ms, 140 iterations) | 462.0 ms | 8.98 s | 252.4 s | 0.036× real time |

- Each depth is held until it settles, so simulated motion differs between settings. Don't compare run lengths.
- ⚠️ No clip here meets the ISMR video rule. The comparison clips run 252.3–264.9 s at 26.4–29.4 MB.
  The submission video is in `../submission/`; these are for talks and the supplement.
- The chosen setting was re-run the same day for the top row: 56.3 ms per frame, against 56.4 ms in the paper.
- These are illustration runs. The paper's quantitative results come from the five-run repeats.
- The B-mode uses the simulator's default acoustic materials and has not been compared with clinical images.

**Reproducing.** Composer: `~/i4h-phantom/analysis/make_comparison_video.py --clock compute`, called by
`~/i4h-phantom/run_scripts/clips_compose_computer_time_20261105.sh`. ⚠️ The saved frames were deleted on 2026-11-05
after review, so the clips cannot be re-composed without re-running the simulation. The per-frame timing records
(`legV5_cmp_*.json`, `.frames.csv`, `.log` in `~/i4h-phantom/out/live_bmode/`) are kept.
