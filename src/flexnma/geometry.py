import numpy as np


def nerf(P0, bL, bTH, t):
    Q = P0.copy()
    for k in range(3, len(P0)):
        if k - 3 >= len(t):
            break
        a, b_, c = (Q[k - 3], Q[k - 2], Q[k - 1])
        bc = c - b_
        bc /= max(np.linalg.norm(bc), 1e-09)
        n = np.cross(b_ - a, bc)
        n /= max(np.linalg.norm(n), 1e-09)
        m = np.cross(n, bc)
        th, ch, bd = (bTH[k - 2], t[k - 3], bL[k - 1])
        Q[k] = (
            c
            + -bd * np.cos(th) * bc
            + bd * np.sin(th) * np.cos(ch) * m
            + bd * np.sin(th) * np.sin(ch) * n
        )
    return Q


def kab(Q, P):
    pc, qc = (Q.mean(0), P.mean(0))
    U, _, Vh = np.linalg.svd((Q - pc).T @ (P - qc))
    d = np.sign(np.linalg.det(U @ Vh))
    return (Q - pc) @ (U @ np.diag([1.0, 1.0, d]) @ Vh) + qc


def frames_of(Pt):
    N, CA, C = (Pt[0::3], Pt[1::3], Pt[2::3])
    e1 = C - CA
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-09)
    v = N - CA
    v -= (v * e1).sum(1, keepdims=True) * e1
    e2 = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-09)
    return (np.stack([e1, e2, np.cross(e1, e2)], axis=-1), CA)


def pdiv(models):
    v = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            A = models[i][:, 1]
            B = models[j][:, 1]
            pc, qc = (A.mean(0), B.mean(0))
            U, _, Vh = np.linalg.svd((A - pc).T @ (B - qc))
            d = np.sign(np.linalg.det(U @ Vh))
            R = U @ np.diag([1.0, 1.0, d]) @ Vh
            v.append(np.sqrt((((A - pc) @ R + qc - B) ** 2).sum(1).mean()))
    return float(np.mean(v))


def _internals(P):
    n = len(P)
    bL = np.linalg.norm(P[1:] - P[:-1], axis=1)
    bTH = np.zeros(max(n - 2, 0))
    for j in range(len(bTH)):
        u = P[j] - P[j + 1]
        v = P[j + 2] - P[j + 1]
        cu = np.linalg.norm(u)
        cv = np.linalg.norm(v)
        if cu > 1e-09 and cv > 1e-09:
            bTH[j] = np.arccos(np.clip(u @ v / (cu * cv), -1.0, 1.0))
    tau = np.zeros(max(n - 3, 0))
    for j in range(len(tau)):
        a, b_, c, d = (P[j], P[j + 1], P[j + 2], P[j + 3])
        bc = c - b_
        bc = bc / max(np.linalg.norm(bc), 1e-09)
        nrm = np.cross(b_ - a, bc)
        nrm = nrm / max(np.linalg.norm(nrm), 1e-09)
        m = np.cross(nrm, bc)
        w = d - c
        tau[j] = np.arctan2(w @ nrm, w @ m)
    return (bL, bTH, tau)
