#!/usr/bin/env python3
"""F_tissue, computed from STABLE NEO-HOOKEAN stress instead of a linear operator.

🚨 THE DEFECT THIS FIXES, AND IT INVALIDATES NUMBERS ALREADY IN THE RECORD.
`F_tissue = -sum over FREE nodes of (K u)_z` has been the project's second, "trustworthy"
force channel since 2026-08-11. K is a LINEAR small-strain stiffness operator
(diag_mesh_stiffness.py:79). It is verified to 0.0000 % against a closed-form patch test --
but a patch test is a SMALL-STRAIN test, so it certifies the operator exactly where the
operator is valid and says nothing about where we have been reading it.

Measured on 2026-08-13: the press ladders run 40-60 % surface strain and carry elements with
J < 0 from 9 mm of indentation onward. A linear operator there is not approximately right, it
is answering a different question. That makes every deep-rung newton suspect INCLUDING the
ones the record calls trustworthy, and it undercuts the "two independent channels" argument
item 24 rests on: two channels can agree perfectly and both be wrong.

🎯 WHAT THIS COMPUTES INSTEAD. The internal nodal force from the SAME energy newton's VBD
actually integrates -- stable Neo-Hookean (Smith et al. 2018), as implemented in
newton/_src/solvers/vbd/particle_vbd_kernels.py:189-222:

    mu_nh    = mu
    lmbd_nh  = lmbd + mu
    alpha    = 1 + mu_nh / lmbd_nh
    Psi      = mu_nh/2 * (||F||^2 - 3) + lmbd_nh/2 * (J - alpha)^2
    P(F)     = mu_nh * F + lmbd_nh * (J - alpha) * cof(F)          (first Piola-Kirchhoff)

and for a linear tet with rest shape matrix Dm and rest volume V0,

    [f1 f2 f3] = -V0 * P * Dm^{-T},      f0 = -(f1 + f2 + f3)

📌 SO THE REFERENCE CHANNEL NOW USES THE SAME CONSTITUTIVE MODEL AS THE SOLVER. That removes
the model mismatch; it does NOT make the channel "validated". It is still verification, not
validation -- see item 24's 2026-08-12 note. What it buys is that the operator is no longer
answering a different question from the one the solver is integrating.

⚠️ AND IT MUST STILL AGREE WITH THE LINEAR OPERATOR AT SMALL STRAIN. That is the check in
`verify_against_linear()` below, and it is the reason to trust the implementation at all: two
independent derivations of the same physics have to coincide where both are valid, and diverge
only where the linear one stops being.
"""
from __future__ import annotations

import numpy as np


class TetElasticitySNH:
    """Stable Neo-Hookean internal forces on a tet mesh.

    rest : (n, 3) rest positions
    tets : (m, 4) node indices
    mu, lmbd : (m,) Lame parameters per element -- the SAME arrays newton is handed via
               model.tet_materials, so the two cannot drift apart.
    """

    def __init__(self, rest: np.ndarray, tets: np.ndarray, mu: np.ndarray, lmbd: np.ndarray):
        self.rest = np.asarray(rest, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.lmbd = np.asarray(lmbd, dtype=np.float64).reshape(-1)
        self.n = len(self.rest)

        r = self.rest[self.tets]                                  # (m,4,3)
        Dm = np.stack([r[:, 1] - r[:, 0], r[:, 2] - r[:, 0], r[:, 3] - r[:, 0]], axis=2)
        self.V0 = np.linalg.det(Dm) / 6.0
        # 🪤 A NEGATIVE REST VOLUME IS AN INVERTED REST ELEMENT AND MUST NOT BE SILENTLY USED.
        # femoral_fem.build() already asserts this on the rest mesh; asserted again here so the
        # operator cannot be handed a mesh that the caller forgot to check.
        if (self.V0 <= 0).any():
            raise SystemExit(f"!!! {(self.V0 <= 0).sum()} tets have non-positive REST volume -- "
                             f"the operator refuses rather than integrating an inverted mesh")
        self.DmInv = np.linalg.inv(Dm)

    def forces(self, pos: np.ndarray) -> np.ndarray:
        """Internal nodal forces (n, 3) at deformed positions `pos`. Sign: force ON the node."""
        p = np.asarray(pos, dtype=np.float64)[self.tets]           # (m,4,3)
        Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
        F = Ds @ self.DmInv                                        # (m,3,3)
        J = np.linalg.det(F)
        # cof(F) = J * F^{-T}, computed as the adjugate transpose so it survives J -> 0 and
        # J < 0. Using J*inv(F).T would blow up on exactly the inverted elements this operator
        # now has to report on, which is the whole reason it exists.
        cof = np.stack([np.cross(F[:, :, 1], F[:, :, 2]),
                        np.cross(F[:, :, 2], F[:, :, 0]),
                        np.cross(F[:, :, 0], F[:, :, 1])], axis=2)
        mu_nh = self.mu
        lmbd_nh = self.lmbd + self.mu
        safe = np.sign(lmbd_nh) * np.maximum(np.abs(lmbd_nh), 1e-6)
        alpha = 1.0 + mu_nh / safe
        P = (mu_nh[:, None, None] * F
             + (lmbd_nh * (J - alpha))[:, None, None] * cof)
        H = -self.V0[:, None, None] * (P @ np.transpose(self.DmInv, (0, 2, 1)))   # (m,3,3)
        f = np.zeros((self.n, 3), dtype=np.float64)
        for a in range(3):
            np.add.at(f, self.tets[:, a + 1], H[:, :, a])
        np.add.at(f, self.tets[:, 0], -H.sum(axis=2))
        return f

    def probe_load_z(self, pos: np.ndarray, free: np.ndarray) -> float:
        """F_tissue: the z load the probe must be applying, summed over FREE nodes.

        At static equilibrium every free node satisfies f_int + f_ext = 0, and the only
        external force on a free node is the probe's push -- so -sum(f_int_z) over free nodes
        IS the probe load, for ANY constitutive model. That argument is what makes the channel
        material-independent, and it is unchanged by swapping the constitutive law; what
        changes is that the law is now the one the solver integrates.
        """
        # 🪤 SIGN. `forces()` returns the force ON each node, i.e. -dPsi/dx. The linear
        # operator this replaces returns K u, which is the NEGATIVE of that, and the old
        # channel therefore read `-sum((K u)_z)`. Verified on a single tet 2026-08-13:
        # forces() == -K.apply(u) to 2.1e-11 at small strain. So the equivalent here is
        # +sum, not -sum. Negating twice was the 154 % small-strain disagreement the
        # cross-check gate caught before any ladder ran on it.
        return float(self.forces(pos)[np.asarray(free, dtype=bool), 2].sum())

    def mesh_quality(self, pos: np.ndarray) -> dict:
        p = np.asarray(pos, dtype=np.float64)[self.tets]
        Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
        J = np.linalg.det(Ds @ self.DmInv)
        return dict(min_jacobian=float(J.min()), n_inverted=int((J <= 0.0).sum()),
                    frac_below_0p1=float((J < 0.1).mean()))


def verify_against_linear(K_linear, snh, rest, free, strain=1e-7, seed=0):
    """The check that makes this trustworthy: at SMALL strain the two must coincide.

    Applies a tiny random displacement and compares -sum((K u)_z) against the SNH channel.
    They are independent derivations of the same physics, so agreement at small strain
    verifies the implementation, and DISAGREEMENT AT LARGE STRAIN is the finding rather than
    a bug. Returns (linear_N, snh_N, rel_err).
    """
    # 🚨 THE PERTURBATION IS SCALED TO A STRAIN, NOT AN ABSOLUTE DISPLACEMENT, AND GETTING
    # THAT WRONG COST A FALSE ALARM ON 2026-08-13. A 1e-6 m field on a 1.5 mm element is a
    # strain of 6.7e-4, not 1e-6 -- so the "46 % disagreement" on the real mesh was the
    # expected O(strain) nonlinear correction, correctly reported, and read as a bug. On a
    # unit-cube test mesh the same number is a strain of 1e-6 and passes, which is exactly how
    # a scale-dependent check hides. Normalise by the mesh's own element size.
    rng = np.random.default_rng(seed)
    r = np.asarray(rest)
    h = float(np.median(np.linalg.norm(r[snh.tets[:, 1]] - r[snh.tets[:, 0]], axis=1)))
    u = rng.standard_normal(r.shape) * (strain * h)
    lin = float(-K_linear.apply(u)[np.asarray(free, dtype=bool), 2].sum())
    # SNH is measured about the rest state, so subtract its (numerically nonzero) rest residual
    f0 = snh.probe_load_z(rest, free)
    nl = snh.probe_load_z(np.asarray(rest) + u, free) - f0
    rel = abs(nl - lin) / max(abs(lin), 1e-30)
    # 🎯 rel MUST FALL LINEARLY WITH `strain`. That convergence is the verification -- a fixed
    # tolerance at one scale is not, because the nonlinear term is real and must not vanish.
    return lin, nl, rel, h
