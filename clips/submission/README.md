# Submission video (ISMR 2027)

**Upload `ismr2027_submission_video.mp4`.** 117 s, 1280 × 720, 30 frames/s, 9.66 MB, no audio. ISMR's limits are
180 s, 10 MB, 480 px and 20 frames/s.

It tells the paper's story in the paper's order:

| time | section | on screen |
|---|---|---|
| 0–5 s | title | authors, affiliations, MRD and University of Minnesota logos |
| 5–25 s | problem | live run of the probe sweeping the leg; Fig. S1, the causal chain from robot action to robot observation |
| 25–37 s | method | cut-away live run, vessels labelled |
| 37–56 s | benchtop phantom | Fig. 1 and Fig. 2, with call-outs |
| 56–61 s | slab | element size, time step and iterations set on the slab (Supplement, Fig. S4) |
| 61–83 s | speed | Fig. 3 with call-outs; then the chosen-versus-finer comparison below, 10 s of it |
| 83–105 s | leg model | cut-away press at 3x with live vein and artery percentages; Fig. 4 |
| 105–117 s | conclusion and acknowledgment | one card |

Every number on screen is quoted from `main.pdf`. Live clips carry a playback-speed badge: "as computed" means the
frames appear as fast as one RTX 5090 produced them.
Built by `i4h-phantom/analysis/make_ismr_video.py` (inputs in `i4h-phantom/analysis/video_assets/`).

## The comparison clips

One probe, two ultrasound images, one clock. The 3-D view shows the chosen setting's probe. Beside it are the
ultrasound images from the chosen setting and from the intermediate setting. A slower setting shows the probe where it
*was*, which is what a robot acting on it would receive. The submission video's speed section is cut from the
smooth-colours clip.

| file | view |
|---|---|
| `submission_cutaway_element_colours.mp4` | cut-away, element colours |
| `submission_cutaway_smooth_colours.mp4` | cut-away, smooth tissue colours |

### What is on screen

- **Left:** 3-D view, chosen setting (8.33 ms time step, 35 iterations per step).
- **Middle:** ultrasound, chosen setting.
- **Right:** ultrasound, intermediate setting (5.56 ms time step, 140 iterations per step).
- Under each image: the probe position that image shows. Green = within one frame of the probe in the 3-D view; amber = further behind.

### How it is timed

- One clock of computer time for all three panels, played at 1×. Nothing is slowed down, sped up or interpolated.
- Each panel shows the newest frame its setting had finished at that moment.
- ⚠️ **Neither setting is real time.** The probe moves at the chosen setting's own pace.
- ⚠️ **The chosen image keeps up by construction:** it comes from the same run as the 3-D view.
- Every image is computed just after its own 3-D view, so even the chosen image is exactly one frame behind. Measured: 99.9–100 % of output frames, all four runs.
- The lag of the intermediate setting does not depend on this choice of clock. It is its measured cost per frame, shown below.

| view | setting | median frame interval | speed | last image |
|---|---|---|---|---|
| element colours | chosen | 74.0 ms | 0.22× real time | 50.5 s |
| element colours | intermediate | 212.9 ms | 0.078× real time | 121.0 s |
| smooth colours | chosen | 76.6 ms | 0.22× real time | 53.1 s |
| smooth colours | intermediate | 213.9 ms | 0.078× real time | 118.1 s |

⚠️ The cut-away rendering is slower than the plain view. The paper's 56.4 ms per frame is the **plain** view and is not
the number shown here.

### Where the frames come from

- A re-layout of `../intermediate/comparison/final_vs_intermediate_{cutsup,cuttex}.mp4`. Output frame *i* is source frame *i*.
- The intermediate run's own 3-D view is dropped; its ultrasound image is kept.
- ⚠️ The panels come from two runs, one per setting, both run 2026-11-05.
- Same subject, asserted before writing: mesh `d14afb0ae307`, B-mode scene `f4dcd823e93b`, same sweep, depths and camera.
- The original saved frames were deleted on 2026-11-05. Panels are cropped from the encoded comparison clip, so this is one further encoding generation.

### Checks (each refuses to write the video)

| check | element colours | smooth colours |
|---|---|---|
| source frame count = records' implied count | 3691 = 3691 | 3605 = 3605 |
| chosen image: new images that change the panel, true alignment | 96.8 % | 97.1 % |
| same, best shift of ±1 to ±3 frames | 76.8 % | 70.7 % |
| intermediate image, true alignment | 99.5 % | 99.5 % |
| same, best shift of ±1 to ±3 frames | 1.1 % | 1.1 % |
| new images that do not change the panel | all while the probe holds still | all while the probe holds still |
| ISMR rule (≤ 180 s, ≤ 10 MB, ≥ 480 px, ≥ 20 frames/s) | 123.0 s, 9.18 MB, 630 px, 30 | 120.2 s, 9.19 MB, 630 px, 30 |

- The alignment check shifts the timing records by up to ±3 frames. The true alignment must score highest, or the video is refused.
- The first version of this check used a fixed threshold. It refused a correctly aligned clip: images taken while the probe holds still barely differ. It was replaced by the shift test above.

### Limits

- One run per setting, for illustration. The paper's quantitative results come from the five-run repeats.
- The B-mode uses the simulator's default acoustic materials and has not been compared with clinical images.

### How the comparison clips were made

`i4h-phantom/analysis/make_submission_video.py`, from the `final_vs_intermediate_*` comparison clips and the two runs' timing records (`out/live_bmode/legV5_cmp_s2_i35*.json`, `legV5_cmp_s3_i140*.json`, written by `i4h-phantom/run_scripts/clips_comparison*_20261105.sh`).
