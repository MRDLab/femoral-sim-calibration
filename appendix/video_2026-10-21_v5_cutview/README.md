# 2026-10-21: live press on Femoral V5, 3-D view cut open at the ultrasound plane

The V5 repeat of [`../../V4_results/appendix/video_2026-10-12_cutview/`](../../V4_results/appendix/video_2026-10-12_cutview/README.md): same camera, cut and colouring.
V5 has no pubic ramus and is 81 mm long.

📌 **Illustration only, for the paper and grant figures.** Timing and closure come from runs without the cut-away.

| folder | 3-D cut face coloured by | run |
|---|---|---|
| `smooth/` | V5 CAD cross-section (smooth outlines); texture and its checks beside it | `legV5_s2_i35_cuttex` |
| `element_colours/` | each element's tissue (jagged outlines) | `legV5_s2_i35_cutsup` |

⚠️ The two folders are two runs. At 19 mm both read vein 47.9 %, artery 94.0 %; at 25.4 mm 28.2 against 28.1 %.

## Same subject, asserted

| check | value |
|---|---|
| finite-element mesh | V5 leg mesh `d14afb0ae307` |
| B-mode scene | `f4dcd823e93b`, 12 bodies, rest set `out/live_render_rest_v5` |
| cut texture | `smooth/cut_texture.{png,json}`, made for `d14afb0ae307`; the loop refuses any other mesh |
| texture against the elements at y = 0 | 99.1 % overall, 94.2 % of 1,582 bundle elements; mirrored control 1.5 % |
| frames saved | 663 of 663 (smooth), 667 of 667 (element colours) |

- ⚠️ The vein is LEFT of the artery in this view and RIGHT of it in the B-mode. Both are correct; the encoder asserts it.
- ⚠️ The coloured outlines are not the solver's tissue map; they differ by up to one element (30 mm in the coarse region).
- The cut-away runs used the main checkout's uncommitted cut-away code. Hashes: `~/i4h-phantom/out/paper_v5/pictures_detail.log`.
- 🪤 Two earlier attempts saved every fifth frame, and one failed on an argument. Archived: `out/live_bmode/archive_v5_save_every5_20261021/`.
