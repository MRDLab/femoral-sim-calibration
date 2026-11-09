"""Put the repository's code directories on `sys.path`.

Until 2026-08-25 every script sat in one flat directory, so `import femoral_fem`
worked from anywhere by accident: Python puts the running script's own directory
at the front of `sys.path`, and that directory held everything. The scripts are
now grouped by what they do, which breaks that, so a script that imports another
module in this repository starts with

    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import repo_paths  # noqa: F401

The first two lines put the repository root on the path so this module can be
found; importing it adds the code directories listed below.

The shared libraries -- femoral_fem, snh_force, materials, probe_geometry,
v3_section, bmode_view -- deliberately stay at the repository root. Moving any of
those six into a subdirectory breaks the importers below. Do not.

🚨 `isaaclab/` DOES NOT IMPORT THIS MODULE, AND THAT IS DELIBERATE -- DO NOT
"FIX" IT. Those scripts run in a different interpreter: envs/isaaclab (README.md),
carrying newton 1.2.1 / warp 1.13.0, against the 1.4.0 / 1.15.0 that the standalone
rig in continuum_mechanics/ runs on. Importing repo_paths would put all ten code
directories on the path, so `import femoral_fem` would succeed inside the Isaac Lab
interpreter and silently build a phantom against the wrong Newton. No material
parameter or gain has been shown to transfer between the two.

So each isaaclab/ script inserts the repository ROOT only, and takes just the one
shared library it needs:

    isaaclab/press_femoral_punch.py
    isaaclab/diag_ftissue_isaaclab.py
    isaaclab/diag_ftissue_curve.py
        sys.path.insert(0, <repository root, resolved from __file__>)
        from snh_force import TetElasticitySNH

snh_force is safe to cross that boundary because it is pure NumPy and touches no
solver -- which is the whole reason it works as a reference force channel. Nothing
else in this repository is safe to cross it. Until 2026-08-25 these scripts lived
outside the repository and hard-coded an absolute home-directory path; folding them in is
what let the absolute path go.

`archive/` and `run_scripts/` are NOT on the path on purpose: an archived script
must never shadow the one that replaced it, and `run_scripts/` holds no modules.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

CODE_DIRS = (
    "anatomy_meshing",
    "continuum_mechanics",
    "force_measurement",
    "robot_controller",
    "raysim_ultrasound",
    "isaac_sim_scene",
    "analysis",
    "figures",
    "platform_probes",
    "geometry_checks",
)


def _add(path):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


_add(ROOT)
for _name in CODE_DIRS:
    _add(os.path.join(ROOT, _name))
