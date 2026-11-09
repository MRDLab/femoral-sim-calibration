"""Compare a rebuilt tetrahedral mesh with a reference, array by array.

    $PY_ANATOMY anatomy_meshing/compare_mesh.py REBUILT.npz REFERENCE.npz

Why not the file digest: numpy writes the build time into every .npz, so a rebuild never has the same sha256 as the
file it reproduces. The run scripts check file digests, so they need the distributed meshes (README.md, section 6a).
This script is how to confirm a rebuild is the same mesh.

Exit 0 when every array the press reads is identical. `body_of` (a 2-D grid of body labels kept for the record, read by
no press script) and `meta` (build date and absolute paths) are reported but do not fail the comparison: one grid cell
of `body_of` can change between identical builds where two bodies tie.
"""
import sys

import numpy as np

INFORMATIONAL = {"body_of", "meta"}

a = np.load(sys.argv[1], allow_pickle=True)
b = np.load(sys.argv[2], allow_pickle=True)
bad = []
for k in sorted(set(a.files) | set(b.files)):
    if k not in a.files or k not in b.files:
        print(f"  {k:<20} MISSING in {'rebuilt' if k not in a.files else 'reference'}")
        bad.append(k)
        continue
    x, y = a[k], b[k]
    same = x.shape == y.shape and (str(x) == str(y) if x.dtype.kind == "O" else np.array_equal(x, y))
    if same:
        print(f"  {k:<20} identical")
        continue
    n = int((x != y).sum()) if x.shape == y.shape and x.dtype.kind != "O" else -1
    tag = "differs (informational)" if k in INFORMATIONAL else "DIFFERS"
    print(f"  {k:<20} {tag}" + (f": {n} of {x.size} entries" if n >= 0 else ""))
    if k not in INFORMATIONAL:
        bad.append(k)
print("SAME MESH" if not bad else f"!!! DIFFERENT MESH: {', '.join(bad)}")
sys.exit(1 if bad else 0)
