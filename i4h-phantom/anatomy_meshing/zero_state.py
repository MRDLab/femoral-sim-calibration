"""Write a zero-displacement state file for deform_meshes.py: the rest nodes of a mesh, and one state equal to them.

    $PY_ANATOMY anatomy_meshing/zero_state.py out/mesh_study_v5/tetmesh_v5_s020_dx1p5.npz out/live_render_rest_v5/zero_state_slab_dx1p5.npz

deform_meshes.py builds the B-mode render set from a solved state. At zero displacement it only needs the solved
region's nodes, which set the imaging slab it keeps. The paper's render set (out/live_render_rest_v5) was built from the
1.5 mm slab's nodes this way; the live worker reports the result as scene sha f4dcd823e93b.
"""
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
rest = np.load(src)["rest"].astype(np.float64)
np.savez(dst, rest=rest, **{"s0.0": rest.copy()})
print(f">>> {dst}: {len(rest):,} nodes from {src}, one state (0.0 mm) equal to rest")
