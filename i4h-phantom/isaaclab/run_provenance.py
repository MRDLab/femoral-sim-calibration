"""Make a run state its own configuration and say which artifact it pressed.

🚨 CLOSES THE SECOND HALF OF OPEN ITEM 18.

Before this existed, `press_femoral_punch.py` wrote `substeps`, `iterations`, `ke` and the lumen
areas -- and NOTHING about which mesh was pressed. Two consequences, both real and both dated:

  * 2026-08-25: the 2026-08-21 press (`out/press_i40_periv_s006.json`) was read as V3 when it was
    V4 -- `femoral_v4_gap12_s006_periv`, 32,436 nodes / 170,148 tets, with the perivascular collar
    and the pubic ramus present. The output could not settle the question; the file NAME and a
    hard-coded string in `analyse_ftissue_convergence.py:130` were the only record. Worse, both
    scripts DEFAULT to `tetmesh_femoral_v4_s030.npz`, so the default disagrees with the run.
  * Item 35 (`35-isaaclab-press-does-not-reconcile.md`) exists because the 2026-08-18 press never
    recorded its own sheath/collar setting, so a genuine discrepancy could not be attributed.

This module lives in ONE place on purpose. Two scripts each carrying their own copy of the
provenance block would be the project's own recurring defect #1 -- two artifacts and nothing that
reconciles them -- reintroduced by the very code meant to prevent it.

⚠️ What this does NOT do: it records what a run was CONFIGURED with. It cannot detect a mesh whose
CONTENT is wrong, only one that differs from another run's. Provenance is a reconciliation tool,
not a correctness check.
"""

import datetime
import hashlib
import os

import numpy as np

__all__ = ["run_config", "mesh_provenance", "provenance_block"]

_SCALAR = (bool, int, float, str, type(None))


def run_config(args):
    """Every argument the run was launched with, serialised defensively.

    An unserialisable argument must never cost the run its result file, so anything that is not a
    JSON scalar is stringified rather than raised on.
    """
    return {k: (v if isinstance(v, _SCALAR) else str(v))
            for k, v in sorted(vars(args).items())}


def _file_record(path):
    rec = {"path": path or "", "exists": bool(path) and os.path.exists(path)}
    if not rec["exists"]:
        return rec
    st = os.stat(path)
    rec["bytes"] = st.st_size
    rec["mtime_utc"] = datetime.datetime.fromtimestamp(
        st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rec["sha256"] = h.hexdigest()
    return rec


def mesh_provenance(args, npz=None):
    """Which artifact was pressed: paths, sizes, mtimes, content digests, and the bodies inside.

    📌 The DIGEST is what makes this a reconciliation rather than a label. Two runs quoting the
    same npz path are comparable only if they quote the same digest -- a path is a name, a hash is
    the artifact.

    📌 The BODY LIST is here because the perivascular collar, the femoral canal and the nerve are
    present-or-absent facts that change the result, and until now they were recoverable only from a
    file name. `femoral_v4_gap12_s006_periv` carries Perivascular / FemoralCanal / Nerve; the s030
    mesh carries none of the three.
    """
    prov = {k: _file_record(getattr(args, k, "") or "") for k in ("npz", "usd")}
    if npz is not None:
        try:
            prov["bodies"] = sorted(str(b) for b in np.unique(npz["tet_body"]))
        except Exception as exc:                       # never lose the result file over metadata
            prov["bodies"] = f"unavailable: {exc}"
        prov["n_nodes_in_file"] = int(len(npz["rest"])) if "rest" in npz else None
        prov["n_tets_in_file"] = int(len(npz["tets"])) if "tets" in npz else None
    return prov


def provenance_block(args, npz=None):
    """The pair, ready to splat into the result dict: `**provenance_block(args_cli, d)`."""
    return {"run_config": run_config(args),
            "mesh_provenance": mesh_provenance(args, npz)}
