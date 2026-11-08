"""The anatomy, meshes and run folders every figure and table in the paper is drawn from. One place. 2026-10-21.

From 2026-10-21 the paper describes Femoral V5, the cast gel leg: V4 without the pubic ramus, cut to 81 mm
(i4h-phantom/anatomy_meshing/v5_from_v4.py). Every script imports its paths and digests from here, so a figure cannot
mix generations: a panel from V4 beside a panel from V5 is the defect the 2026-08-21 GIF shipped (item 37).

The V4 values are kept below, unused, so the V4 figures in git history can be traced to their inputs.
"""
import hashlib
import pathlib

PH = pathlib.Path(__file__).resolve().parent.parent / "out"

GENERATION = "V5"
ANATOMY = PH / "femoral_v5"                       # OBJs and sections, LPS mm
SECTION = ANATOMY / "section_s020.json"
LEG = PH / "mesh_leg_v5" / "tetmesh_leg_v5_s020_f1p5_c30_w30_bone.npz"
LEG_SHA = "d14afb0ae307"
SLAB = {"2p5": (PH / "mesh_study_v5" / "tetmesh_v5_s020_dx2p5.npz", "5935645c3cd0"),
        "2": (PH / "mesh_study_v5" / "tetmesh_v5_s020_dx2.npz", "afedb8754e8e"),
        "1p5": (PH / "mesh_study_v5" / "tetmesh_v5_s020_dx1p5.npz", "3ab9d1302397"),
        "1p25": (PH / "mesh_study_v5" / "tetmesh_v5_s020_dx1p25.npz", "2702d8c674f2"),
        "1p0": (PH / "mesh_study_v5" / "tetmesh_v5_s020_dx1p0.npz", "c012b923ef7c")}
SLAB_15, SLAB_15_SHA = SLAB["1p5"]
SENS = PH / "mesh_leg_v5_sensitivity"             # manifest.json holds each case's digest
RUNS = PH / "paper_v5"                            # queue.log; leg/rR, leg_sens/<case>/rR, leg_inversion/rR, slab/rR
RUN_PREFIX = "legV5_"                             # leg run tags: legV5_s2_i35, legV5_s168_i140_ref, ...
LIVE = PH / "live_bmode"                          # picture runs: legV5_s2_i35, legV5_s2_i35_cuttex, legV5_s2_i35_cutsup
LIVE_RUN = "legV5_s2_i35"
CUT_TEXTURE = PH / "cut_texture_meshV5"
LEG_LENGTH_MM = 81.0

# V4, for the record only (Mesh F and the 2026-09-30 to 10-12 runs). Do not import these into a V5 figure.
V4 = dict(ANATOMY=PH / "femoral_v4_gap12", LEG=PH / "mesh_leg" / "tetmesh_leg_s020_f1p5_c30_w30_bone.npz",
          LEG_SHA="c4e13e7d7574", SLAB_15=PH / "mesh_study" / "tetmesh_s020_dx1p5.npz", SLAB_15_SHA="6f3c9f86ab61",
          LEG_LENGTH_MM=120.0)


import re  # noqa: E402

# The V5 queue writes one log. Lines: "DONE r1 legV5_s2_i35 (2 x 35, graph on) rc=0 GPU IDLE after 30 s",
# "DONE r1 sens control (...) rc=0 ...", "DONE r1 inversion positions (...) rc=0 ...", "DONE r1 slab ts_s8_i10 (...) rc=0 ...".
_DONE = {"leg": r"DONE r(\d+) (legV5_\S+) \([^)]*\) rc=(\d+) GPU (\w+)",
         "sens": r"DONE r(\d+) sens (\S+) \([^)]*\) rc=(\d+) GPU (\w+)",
         "inversion": r"DONE r(\d+) (inversion) positions \([^)]*\) rc=(\d+) GPU (\w+)",
         "slab": r"DONE r(\d+) slab (\S+) \([^)]*\) rc=(\d+) GPU (\w+)"}


def queue_status(group):
    """{(repeat, tag): (rc, gpu_state)} for one group of the V5 queue, as its own log records it. The log decides."""
    p = RUNS / "queue.log"
    out = {}
    if p.exists():
        for m in re.finditer(_DONE[group], p.read_text()):
            out[(int(m[1]), m[2])] = (int(m[3]), m[4])
    return out


def queue_seconds(group):
    """{(repeat, tag): wall seconds} for runs the V5 queue logged rc=0 on an idle GPU (the launcher times each run)."""
    p = RUNS / "queue.log"
    out = {}
    if p.exists():
        for m in re.finditer(_DONE[group] + r" after (\d+) s", p.read_text()):
            if int(m[3]) == 0 and m[4] == "IDLE":
                out[(int(m[1]), m[2])] = float(m[5])
    return out


def usable(group, tag, rows_key="depth_rows", n_depths=7):
    """[(repeat, path)] of the runs of `tag` that count: logged rc=0 on an idle GPU, JSON present, all depths present.
    Slab runs must also say complete. Never anything but the V5 queue's own runs."""
    out = []
    for (rep, t), (rc, gpu) in sorted(queue_status(group).items()):
        if t != tag or rc != 0 or gpu != "IDLE":
            continue
        p = run_json(group, rep, t)
        if not p.exists():
            continue
        d = __import__("json").load(open(p))
        rows = d.get(rows_key) or d.get("rows") or []
        if len(rows) != n_depths or (group == "slab" and not d.get("complete")):
            continue
        out.append((rep, p))
    return out


def run_json(group, rep, tag):
    """Where the V5 queue wrote a run's JSON."""
    return {"leg": RUNS / "leg" / f"r{rep}" / f"{tag}.json",
            "sens": RUNS / "leg_sens" / tag / f"r{rep}" / f"{tag}.json",
            "inversion": RUNS / "leg_inversion" / f"r{rep}" / f"{RUN_PREFIX}s2_i35.json",
            "slab": RUNS / "slab" / f"r{rep}" / f"{tag}.json"}[group]


def sha12(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:12]


def verified(path, digest, what="mesh"):
    """Return `path` if its content digest is `digest`; refuse otherwise. A mesh is named by content, never by file name."""
    got = sha12(path)
    if got != digest:
        raise SystemExit(f"!!! {what} {path} has digest {got}, expected {digest} ({GENERATION}): refusing to draw it")
    return path
