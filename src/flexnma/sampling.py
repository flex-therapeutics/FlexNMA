"""Calibrated sampling with soft regional restraints."""

import json
from importlib.resources import files
import numpy as np
from . import core as PM
from .geometry import _internals, nerf, kab, frames_of, pdiv
from .structure import read_structure, select_residues


def ensemble(P, bL, bTH, tau, loc, scale, K, seed, held=None):
    rng = np.random.default_rng(seed)
    held = list(held or [])
    Hi, hfc = PM.hold_lock(P, held) if held else (None, None)
    pairs = list(zip(*np.where(np.triu(PM.strand_pairs(P[1::3]), 1))))
    Pi, fc = PM.pair_lock(P, pairs) if pairs else (None, None)
    out = []
    for _ in range(K):
        t, _ = PM.draw_torsion(P, rng, n_modes=30)
        if Pi is not None:
            t[fc] = Pi @ t[fc]
        if Hi is not None:
            t[hfc] = Hi @ t[hfc]
        t = t / max(float(np.sqrt((t * t).mean())), 1e-9) * np.radians(scale)
        if held:
            t = PM.hold_correct(P, bL, bTH, tau, t, held, rounds=8)
        if pairs:
            t = PM.pair_correct(P, bL, bTH, tau, t, pairs, rounds=8, clash_cut=0)
        Q = nerf(P, bL, bTH, tau + t)
        Q = PM.kabsch_on(Q, P, held) if held else kab(Q, P)
        rot, trans = frames_of(Q)
        out.append(np.einsum("rij,raj->rai", rot, loc) + trans[:, None, :])
    return out


def sample(input_path, samples=60, spread=None, restrain="", seed=0):
    if samples < 1:
        raise ValueError("samples must be positive")
    residues, atoms = read_structure(input_path)
    held = select_residues(restrain, residues)
    if any("O" not in a for a in atoms):
        raise ValueError("Backbone oxygen atoms are required")
    P = np.array([a[n] for a in atoms for n in ("N", "CA", "C")])
    bL, bTH, tau = _internals(P)
    rot, trans = frames_of(P)
    # N/CA/C are sampled; O follows the local residue frame.
    xyz = np.array([[a[n] for n in ("N", "CA", "C", "O")] for a in atoms])
    loc = np.einsum("naj,njk->nak", xyz - trans[:, None], rot)
    if spread is None:
        cal = json.loads(
            files("flexnma").joinpath("spread_calibration.json").read_text()
        )
        w, _ = PM.anm_lowmodes(P[1::3], n_modes=30)
        spread = float(np.interp(np.sum(1 / w[:20]), cal["x"], cal["y"]))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("spread must be positive and finite")
    lo, hi = 0.05, 30.0
    for _ in range(10):
        scale = (lo + hi) / 2
        measured = pdiv(ensemble(P, bL, bTH, tau, loc, scale, 8, 0, held))
        if measured < spread:
            lo = scale
        else:
            hi = scale
    scale = (lo + hi) / 2
    return (
        np.asarray(ensemble(P, bL, bTH, tau, loc, scale, samples, seed, held)),
        residues,
        spread,
    )


def write_pdb(path, models, residues):
    from pathlib import Path

    lines = []
    for k, model in enumerate(models, 1):
        lines.append(f"MODEL     {k:4d}")
        serial = 0
        for coords, (chain, resid, icode, name) in zip(model, residues):
            for atom, (x, y, z) in zip(("N", "CA", "C", "O"), coords):
                serial += 1
                lines.append(
                    f"ATOM  {serial:5d} {atom:^4s} {name:3s} {chain}{resid:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2s}"
                )
        lines.extend(["TER", "ENDMDL"])
    lines.append("END")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
