# data/

The numbers behind every figure and table, as CSV or JSON, written by the scripts named here from the raw run files
(which are not in this repository). Each figure script reads from here where it can.

| file or folder | written by | contents |
|---|---|---|
| `meshF_repeats/per_run.csv`, `per_setting_summary.csv` | `analyze_meshF_repeats.py` | every leg run (five per setting): wall clock per frame, lumen areas at each depth, settling |
| `meshF_repeats/cost_model.csv` | same | two- and three-term cost fits with bootstrap SDs |
| `meshF_repeats/vein_closure_realtime_setting.csv` | same | the chosen setting's lumen areas (Table S7) |
| `meshF_repeats/wallclock_figure_caption_numbers.json` | same | the numbers the speed-figure caption quotes |
| `meshF_repeats/excluded_runs.csv` | same | runs the queue log did not mark `rc=0 GPU IDLE`, and why |
| `headline_numbers.json` | `headline_numbers.py` | every speed and accuracy number the abstract and conclusion quote |
| `cast_validation.csv`, `cast_trials.csv` | `fig_cast_validation.py` | benchtop means ± SD, simulated values and errors per depth; every individual press |
| `cast_segmentation_tiles.csv` | `fig_cast_segmentation.py` | which benchtop image is which tile, with its picks |
| `sim_vs_real_numbers.json`, `pipeline_frame_numbers.json` | `fig_sim_vs_real.py`, `fig_pipeline_frame.py` | crops, scales and readings used by those composites |
| `slab_cost/per_pass.csv` | `fig_slab_studies.py` | wall clock per solver pass for each slab element size |
| `slab_calibration/` | `fig_slab_calibration.py` | slab against leg: mesh sizes, frame check (centroid agreement), tets per body, press comparison |
| `slab_inversion/`, `leg_inversion/` | `audit_slab_inversion.py`, `audit_leg_inversion.py` | inverted elements by depth and their distance to the vessels |
| `sensitivity/` | `analyze_sensitivity.py` | vein-wall and nerve stiffness cases (Table S6) |
| `leg_mesh_figure_numbers.json` | `fig_meshF_pictures.py` | numbers printed on the leg-mesh pictures |
| `plagiarism/overlaps.csv` | `plagiarism_check.py` | 8-word runs shared with the cited PDFs on disk (currently none) |
