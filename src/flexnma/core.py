import numpy as np

_CACHE = {}


def strand_pairs(ca, cut=5.6, min_sep=4):
    r = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    sep = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])
    return (r < cut) & (sep > min_sep)


def anm_lowmodes(ca, n_modes=30, cut=13.0, sheet_boost=1.0):
    n = len(ca)
    d = ca[:, None, :] - ca[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = d / r[..., None]
    contact = (r <= cut) & (r > 1e-06)
    u = np.where(contact[..., None], u, 0.0)
    B = u[..., :, None] * u[..., None, :]
    if sheet_boost != 1.0:
        g = np.where(strand_pairs(ca), float(sheet_boost), 1.0)
        B = B * g[..., None, None]
    H = np.zeros((n, n, 3, 3))
    H -= B
    diag = B.sum(axis=1)
    idx = np.arange(n)
    H[idx, idx] = diag
    H = H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    k = min(n_modes, 3 * n - 6)
    try:
        import scipy.linalg as _sla

        w, V = _sla.eigh(H, subset_by_index=[0, 6 + k - 1])
    except Exception:
        w, V = np.linalg.eigh(H)
    return (np.maximum(w[6 : 6 + k], 1e-08), V[:, 6 : 6 + k])


def ca_jacobian(P, want=None):
    n_at = len(P)
    n = n_at // 3
    T = n_at - 3
    J = np.zeros((3 * n, T))
    ca_idx = np.arange(n) * 3 + 1
    sel = None
    for k in range(T):
        o = P[k + 1]
        u = P[k + 2] - P[k + 1]
        nu = np.linalg.norm(u)
        if nu < 1e-09:
            continue
        u = u / nu
        first = (k + 3 - 1) // 3
        if first >= n:
            continue
        idx = np.arange(first, n) if sel is None else sel[sel >= first]
        if not len(idx):
            continue
        rel = P[ca_idx[idx]] - o
        rows_ = (idx[:, None] * 3 + np.arange(3)).ravel()
        J[rows_, k] = np.cross(u, rel).ravel()
    return J


def build_projector(
    P, n_modes=30, ridge=0.001, cut=13.0, sheet_boost=1.0, derigid=False
):
    h = hash((np.round(P, 3).tobytes(), int(n_modes), round(float(sheet_boost), 4)))
    key = (h, len(P), n_modes, round(ridge, 6), cut, bool(derigid))
    ent = _CACHE.get(key)
    if ent is not None:
        return ent
    if len(_CACHE) > 256:
        _CACHE.clear()
    ca = P[1::3]
    lam, V = anm_lowmodes(ca, n_modes=n_modes, cut=cut, sheet_boost=sheet_boost)
    if derigid:
        x = ca - ca.mean(0)
        nres = len(ca)
        Rb = np.zeros((3 * nres, 6))
        for a_ in range(3):
            Rb[a_::3, a_] = 1.0
        for a_, (i_, j_) in enumerate(((1, 2), (2, 0), (0, 1))):
            Rb[i_::3, 3 + a_] = x[:, j_]
            Rb[j_::3, 3 + a_] = -x[:, i_]
        Q, _ = np.linalg.qr(Rb)
        V = V - Q @ (Q.T @ V)
        nrm = np.linalg.norm(V, axis=0, keepdims=True)
        V = V / np.maximum(nrm, 1e-12)
    J = ca_jacobian(P)
    free_cols = np.array([k for k in range(J.shape[1]) if k % 3 != 1])
    Jf = J[:, free_cols]
    JtJ = Jf.T @ Jf
    A = np.linalg.solve(
        JtJ + ridge * np.trace(JtJ) / JtJ.shape[0] * np.eye(JtJ.shape[0]), Jf.T
    )
    ent = (V, lam, A, Jf, free_cols, J.shape[1])
    _CACHE[key] = ent
    return ent


def pair_lock(P, pair_idx, ridge=1e-06):
    J = ca_jacobian(P)
    free_cols = np.array([k for k in range(J.shape[1]) if k % 3 != 1])
    Jf = J[:, free_cols]
    ca = P[1::3]
    rows = []
    for i, j in pair_idx:
        if i >= len(ca) or j >= len(ca):
            continue
        u = ca[i] - ca[j]
        n = np.linalg.norm(u)
        if n < 1e-06:
            continue
        u = u / n
        rows.append(u @ (Jf[3 * i : 3 * i + 3, :] - Jf[3 * j : 3 * j + 3, :]))
    if not rows:
        return (None, free_cols)
    C = np.asarray(rows)
    CC = C @ C.T
    Pi = np.eye(C.shape[1]) - C.T @ np.linalg.solve(
        CC + ridge * np.trace(CC) / max(CC.shape[0], 1) * np.eye(CC.shape[0]), C
    )
    return (Pi, free_cols)


def pair_correct(
    P,
    bL,
    bTH,
    tau,
    t,
    pair_idx,
    rounds=3,
    ridge=0.03,
    damp=0.5,
    max_step_deg=2.0,
    tol=0.01,
    jac_every=2,
    clash_cut=0.0,
    clash_pad=0.2,
    clash_sep=4,
    clash_w=1.0,
):
    t = t.copy()
    ca0 = P[1::3]
    d0 = np.array([np.linalg.norm(ca0[i] - ca0[j]) for i, j in pair_idx])
    free_cols = np.array([k for k in range(3 * (len(ca0) - 1)) if k % 3 != 1])
    _J = None
    for _rnd in range(rounds):
        Q = nerf_trace(P, bL, bTH, tau + t)
        ca = Q[1::3]
        res, rows = ([], [])
        if _J is None or _rnd % jac_every == 0:
            _J = ca_jacobian(Q)
        J = _J
        Jf = J[:, free_cols]
        for k, (i, j) in enumerate(pair_idx):
            if i >= len(ca) or j >= len(ca):
                continue
            u = ca[i] - ca[j]
            n = np.linalg.norm(u)
            if n < 1e-06:
                continue
            u = u / n
            res.append(n - d0[k])
            rows.append(u @ (Jf[3 * i : 3 * i + 3, :] - Jf[3 * j : 3 * j + 3, :]))
        _nclash = 0
        if clash_cut > 0:
            import os as _os

            clash_pad = float(_os.environ.get("CLASH_PAD", clash_pad))
            clash_w = float(_os.environ.get("CLASH_W", clash_w))
            _held = {(min(i, j), max(i, j)) for i, j in pair_idx}
            _tgt = clash_cut + clash_pad
            _D = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
            _ii, _jj = np.triu_indices(len(ca), k=clash_sep)
            for i, j in zip(_ii, _jj):
                n = _D[i, j]
                if n >= clash_cut or n < 1e-06 or (int(i), int(j)) in _held:
                    continue
                u = (ca[i] - ca[j]) / n
                res.append(clash_w * (n - _tgt))
                rows.append(
                    clash_w
                    * (u @ (Jf[3 * i : 3 * i + 3, :] - Jf[3 * j : 3 * j + 3, :]))
                )
                _nclash += 1
        if not rows:
            break
        C = np.asarray(rows)
        r = np.asarray(res)
        _rpair = r[: len(r) - _nclash] if _nclash else r
        if _nclash == 0 and len(_rpair) and (np.abs(_rpair).mean() < tol):
            break
        CC = C @ C.T
        lam = ridge * max(np.trace(CC) / max(CC.shape[0], 1), 1e-12)
        dt_free = -C.T @ np.linalg.solve(CC + lam * np.eye(CC.shape[0]), r)
        cap = np.radians(max_step_deg)
        nrm = np.abs(dt_free).max()
        if nrm > cap:
            dt_free = dt_free * (cap / nrm)
        t[free_cols] += damp * dt_free
    return t


def draw_torsion(
    P,
    rng,
    n_modes=30,
    ridge=0.001,
    cut=13.0,
    spectrum=1.0,
    sheet_boost=1.0,
    tilt=None,
    derigid=False,
):
    V, lam, A, Jf, free_cols, T = build_projector(
        P, n_modes, ridge, cut, sheet_boost, derigid
    )
    z = rng.standard_normal(V.shape[1]) / np.maximum(lam, 1e-08) ** (0.5 * spectrum)
    if tilt is not None:
        z = z * np.asarray(tilt, float)[: V.shape[1]]
    d = V @ z
    tf = A @ d
    t = np.zeros(T)
    t[free_cols] = tf
    got = Jf @ tf
    denom = float(np.linalg.norm(d))
    frac = float(np.linalg.norm(got) / denom) if denom > 1e-09 else 0.0
    return (t, frac)


def nerf_trace(P0, bL, bTH, tau):
    Q = P0.copy()
    nt = len(tau)
    for k in range(3, len(P0)):
        if k - 3 >= nt:
            break
        a, b, c = (Q[k - 3], Q[k - 2], Q[k - 1])
        bc = c - b
        bc = bc / max(np.linalg.norm(bc), 1e-09)
        nrm = np.cross(b - a, bc)
        nrm = nrm / max(np.linalg.norm(nrm), 1e-09)
        m = np.cross(nrm, bc)
        th, ch, bd = (bTH[k - 2], tau[k - 3], bL[k - 1])
        Q[k] = (
            c
            + -bd * np.cos(th) * bc
            + bd * np.sin(th) * np.cos(ch) * m
            + bd * np.sin(th) * np.sin(ch) * nrm
        )
    return Q


def kabsch_onto(Q, P):
    pc, qc = (Q.mean(0), P.mean(0))
    U, _, Vh = np.linalg.svd((Q - pc).T @ (P - qc))
    d = np.sign(np.linalg.det(U @ Vh))
    return (Q - pc) @ (U @ np.diag([1.0, 1.0, d]) @ Vh) + qc


def hold_lock(P, held, ridge=1e-06):
    J = ca_jacobian(P)
    free_cols = np.array([k for k in range(J.shape[1]) if k % 3 != 1])
    Jf = J[:, free_cols]
    rows = [Jf[3 * i + c, :] for i in held for c in range(3)]
    if not rows:
        return (None, free_cols)
    C = np.asarray(rows)
    CC = C @ C.T
    Pi = np.eye(C.shape[1]) - C.T @ np.linalg.solve(
        CC + ridge * np.trace(CC) / max(CC.shape[0], 1) * np.eye(CC.shape[0]), C
    )
    return (Pi, free_cols)


def hold_correct(
    P, bL, bTH, tau, t, held, rounds=8, ridge=0.03, damp=0.5, max_step_deg=2.0
):
    t = t.copy()
    ca0 = P[1::3]
    free_cols = np.array([k for k in range(3 * (len(ca0) - 1)) if k % 3 != 1])
    for _ in range(rounds):
        Q = nerf_trace(P, bL, bTH, tau + t)
        Q = kabsch_on(Q, P, held)
        ca = Q[1::3]
        J = ca_jacobian(Q, want=held)
        Jf = J[:, free_cols]
        rows, res = ([], [])
        for i in held:
            if i >= len(ca):
                continue
            for c in range(3):
                rows.append(Jf[3 * i + c, :])
                res.append(ca[i][c] - ca0[i][c])
        if not rows:
            break
        C = np.asarray(rows)
        r = np.asarray(res)
        CC = C @ C.T
        lam = ridge * max(np.trace(CC) / max(CC.shape[0], 1), 1e-12)
        dt = -C.T @ np.linalg.solve(CC + lam * np.eye(CC.shape[0]), r)
        cap = np.radians(max_step_deg)
        nrm = np.abs(dt).max()
        if nrm > cap:
            dt = dt * (cap / nrm)
        t[free_cols] += damp * dt
    return t


def kabsch_on(Q, P, idx):
    sel = np.array([3 * i + c for i in idx for c in range(3)])
    A = Q[sel]
    B = P[sel]
    ac, bc = (A.mean(0), B.mean(0))
    U, _, Vh = np.linalg.svd((A - ac).T @ (B - bc))
    d = np.sign(np.linalg.det(U @ Vh))
    R = U @ np.diag([1.0, 1.0, d]) @ Vh
    return (Q - ac) @ R + bc
