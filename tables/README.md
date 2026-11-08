# tables/

LaTeX table fragments. All but `cast_validation.tex` are written by `scripts/make_tables.py` from the run files;
`cast_validation.tex` by `scripts/fig_cast_validation.py`. `scripts/verify_tables.py` re-derives every printed cell by an
independent path and must print `0 failures`. `values.json` records every cell with the file it came from.
Do not edit the `.tex` files by hand.

| file | paper slot | contents |
|---|---|---|
| `simulation_geometry.tex` | Table II | geometry and numerical settings of the slab and the leg mesh |
| `phantom_parameters.tex` | Table III | the benchtop phantom's dimensions and materials against the leg model |
| `mesh_results.tex` | S1 | slab element-size study: tets, vein at 25.4 mm, closure floor, headroom, inverted elements, wall clock per press |
| `timestep_slab.tex` | S2 | slab time-step study: settling, largest gap from the reference, wall clock per press |
| `cost_and_graph.tex` | S3 | leg cost model (per pass, per step, per frame) and the CUDA Graph gain |
| `speed_accuracy.tex` | S4 | every leg setting: ms per frame, speed-up, vein and artery gap from the reference |
| `cast_validation.tex` | S5 | benchtop against simulated lumen area at every depth, with the error |
| `appendix_wall_sensitivity.tex` | S6 | vein-wall and nerve stiffness cases |
| `appendix_vein.tex`, `appendix_artery.tex` | S7, S8 | vein and artery lumen area for every setting at every depth |

Table I (structures modeled, with moduli) is written by hand in `main_V5.tex`. `vein_closure.tex` (the chosen setting alone) is generated but not used: its rows are inside S7 and S8.
