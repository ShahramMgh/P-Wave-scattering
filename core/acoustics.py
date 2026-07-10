"""
acoustics.py — wave-based 3D room acoustics engine (scalar Helmholtz, MFS).

Professional low-frequency room-acoustics analysis, the wave-based regime where
geometric/ray methods (ODEON, CATT) are invalid and commercial wave solvers
(Treble, COMSOL) operate:

  * frequency domain Helmholtz solve per FFT bin -> inverse FFT to time domain
    (unconditionally stable; no time-marching instability)
  * Method of Fundamental Solutions: point sources OUTSIDE the domain, strengths
    fitted to the wall boundary condition on collocation points (regularised
    square solve; offset 4x spacing = the well-conditioned regime from the MFS
    literature: Fairweather & Karageorghis; Alves; Antunes)
  * locally-reacting impedance walls:  dp/dn + i k beta(f) p = 0, with
    frequency-dependent admittance beta(f) derived per surface from published
    octave-band absorption coefficients alpha(f):
        R = sqrt(1 - alpha),  beta = (1 - R) / (1 + R)
  * interior monopole source with selectable band-limited signature
    (Ricker / tone burst / click / chirp)
  * receiver microphone -> impulse response, transfer function |H(f)|
  * ISO 3382-style metrics: RT60 (Sabine, Eyring, Schroeder-integration T20),
    EDT, C50, C80, D50 + analytic rigid-room mode table & Schroeder frequency

Conventions: time factor e^{+i w t}, outgoing G = e^{-ikr}/(4 pi r).

Usage:
    python acoustics.py --selftest        # physics checks (~1 min)
    python acoustics.py                   # demo run -> room3d.html
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass

import numpy as np


# ============================================================================
# 1. Medium & materials
# ============================================================================
@dataclass
class Air:
    c: float = 343.0      # sound speed  [m/s]
    rho: float = 1.2      # density      [kg/m^3]


# Octave-band absorption coefficients alpha at BANDS Hz (standard building-
# acoustics tables, e.g. Vorlander "Auralization", Long "Architectural
# Acoustics"; 63 Hz extrapolated where sources only start at 125 Hz).
BANDS = (63.0, 125.0, 250.0, 500.0)
MATERIALS = {
    'concrete':       (0.01, 0.01, 0.01, 0.02),
    'brick':          (0.03, 0.03, 0.03, 0.04),
    'ceramic tile':   (0.01, 0.01, 0.01, 0.02),
    'glass window':   (0.25, 0.35, 0.25, 0.18),
    'gypsum drywall': (0.25, 0.29, 0.10, 0.05),
    'wood floor':     (0.20, 0.15, 0.11, 0.10),
    'carpet':         (0.03, 0.02, 0.06, 0.14),
    'heavy curtain':  (0.10, 0.14, 0.35, 0.55),
    'acoustic panel': (0.15, 0.28, 0.70, 0.90),
    'ceiling tile':   (0.30, 0.30, 0.40, 0.50),
    'open window':    (0.99, 0.99, 0.99, 0.99),
}


def alpha_of(mat, f):
    """Absorption coefficient at frequency f (log-f interpolation, clamped)."""
    lf = np.log(np.clip(f, BANDS[0], BANDS[-1]))
    return float(min(np.interp(lf, np.log(BANDS), MATERIALS[mat]), 0.99))


def beta_of(mat, f):
    """Normalised admittance from alpha (normal-incidence energy relation)."""
    R = np.sqrt(1.0 - alpha_of(mat, f))
    return (1.0 - R) / (1.0 + R)


# ============================================================================
# 2. Scalar kernels (3D Helmholtz, outgoing for e^{+iwt})
# ============================================================================
def G_monopole(x, y, omega, med: Air):
    """Free-space Green's function  G = e^{-ikr}/(4 pi r).
    x: (Nx,3) field pts, y: (Ny,3) source pts  ->  (Nx,Ny) complex."""
    r = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    k = omega / med.c
    return np.exp(-1j * k * r) / (4.0 * np.pi * r)


def dGdn(x, nrm, y, omega, med: Air):
    """Normal derivative at x:  dG/dn = G'(r) (e . n),  e = (x-y)/r."""
    d = x[:, None, :] - y[None, :, :]
    r = np.linalg.norm(d, axis=2)
    k = omega / med.c
    Gp = -np.exp(-1j * k * r) * (1.0 + 1j * k * r) / (4.0 * np.pi * r * r)
    edotn = np.einsum('csk,ck->cs', d / r[..., None], nrm)
    return Gp * edotn


# ============================================================================
# 3. Source signatures (all band-limited, no DC)
# ============================================================================
def source_signal(kind, t, f0):
    """Return (s(t), t0). Ricker: broadband, sharp front. Tone burst: narrow-
    band ~4 cycles. Click: differentiated Gaussian. Chirp: 0.5-1.6 f0 sweep."""
    if kind == 'toneburst':
        t0 = 3.0 / f0
        s = np.sin(2 * np.pi * f0 * (t - t0)) * np.exp(-(np.pi * f0 * (t - t0) / 4.0) ** 2)
    elif kind == 'click':
        t0 = 1.6 / f0
        s = -(t - t0) * np.exp(-(np.pi * f0 * (t - t0)) ** 2)
        s /= np.abs(s).max()
    elif kind == 'chirp':
        t0, Tc = 0.5 / f0, 6.0 / f0
        u = np.clip((t - t0) / Tc, 0, 1)
        phase = 2 * np.pi * Tc * (0.5 * f0 * u + 0.55 * f0 * u * u)
        s = np.sin(phase) * np.sin(np.pi * u) ** 2 * ((t >= t0) & (t <= t0 + Tc))
    else:  # ricker
        t0 = 1.5 / f0
        a = (np.pi * f0 * (t - t0)) ** 2
        s = (1.0 - 2.0 * a) * np.exp(-a)
    return s, t0


# ============================================================================
# 4. Geometry: box room / OBJ mesh
# ============================================================================
def box_room(L, n_wall):
    """Collocation points on the 6 faces of [0,Lx]x[0,Ly]x[0,Lz].
    Returns (pts, outward_normals, spacing, face_id) with face ids
    0:x=0 1:x=Lx 2:y=0 3:y=Ly 4:floor(z=0) 5:ceiling(z=Lz)."""
    Lx, Ly, Lz = L
    faces = [(0, 0.0, [-1, 0, 0]), (0, Lx, [1, 0, 0]),
             (1, 0.0, [0, -1, 0]), (1, Ly, [0, 1, 0]),
             (2, 0.0, [0, 0, -1]), (2, Lz, [0, 0, 1])]
    areas = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])
    spacing = np.sqrt(areas.sum() / n_wall)
    pts, nrms, fids = [], [], []
    dims = np.array(L)
    for fi, (ax, val, nv) in enumerate(faces):
        u_ax, v_ax = [a for a in range(3) if a != ax]
        nu = max(2, int(round(dims[u_ax] / spacing)))
        nv_ = max(2, int(round(dims[v_ax] / spacing)))
        us = (np.arange(nu) + 0.5) * dims[u_ax] / nu     # cell-centred
        vs = (np.arange(nv_) + 0.5) * dims[v_ax] / nv_
        U, V = np.meshgrid(us, vs)
        P = np.zeros((U.size, 3))
        P[:, ax] = val
        P[:, u_ax] = U.ravel()
        P[:, v_ax] = V.ravel()
        pts.append(P)
        nrms.append(np.tile(np.asarray(nv, float), (U.size, 1)))
        fids.append(np.full(U.size, fi))
    return np.vstack(pts), np.vstack(nrms), spacing, np.concatenate(fids)


def load_obj(path, flip_normals=False):
    """Minimal OBJ loader: triangle centroids as collocation points, face
    normals (assumed outward). Returns (pts, normals, spacing)."""
    verts, tris = [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == 'f':
                idx = [int(tok.split('/')[0]) - 1 for tok in p[1:]]
                for i in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[i], idx[i + 1]])
    V, F = np.asarray(verts), np.asarray(tris)
    P = V[F].mean(axis=1)
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    N = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(N, axis=1)
    N = N / (2 * area[:, None])
    if flip_normals:
        N = -N
    return P, N, np.sqrt(area.sum() / len(P))


# ============================================================================
# 5. Per-frequency solver
# ============================================================================
def _srcs(x0, amps=None):
    """Normalise source spec: (3,) or (n,3) positions + optional amplitudes."""
    X0 = np.atleast_2d(np.asarray(x0, float))
    a = np.ones(len(X0)) if amps is None else np.asarray(amps, float)
    return X0, a


def solve_freq(omega, med, wall, nrm, src, x0, beta_c, amps=None, tik=1e-10):
    """Fit MFS strengths so  dp/dn + i k beta_c p = 0  on the wall
    (beta_c: per-collocation-point admittance; x0: one or many monopoles).
    Tikhonov-regularised square solve (fast LU, no SVD)."""
    X0, a = _srcs(x0, amps)
    k = omega / med.c
    ikb = (1j * k * beta_c)[:, None]
    A = dGdn(wall, nrm, src, omega, med) + ikb * G_monopole(wall, src, omega, med)
    rhs = -(dGdn(wall, nrm, X0, omega, med)
            + ikb * G_monopole(wall, X0, omega, med)) @ a
    AH = A.conj().T
    lam = tik * np.trace(AH @ A).real / A.shape[1]
    return np.linalg.solve(AH @ A + lam * np.eye(A.shape[1]), AH @ rhs)


def bc_residual(omega, med, chk, chkn, src, x0, beta_c, f, amps=None):
    """Relative BC residual on independent wall points (the honesty check)."""
    X0, a = _srcs(x0, amps)
    k = omega / med.c
    ikb = (1j * k * beta_c)[:, None]
    op = dGdn(chk, chkn, src, omega, med) + ikb * G_monopole(chk, src, omega, med)
    inc = (dGdn(chk, chkn, X0, omega, med)
           + ikb * G_monopole(chk, X0, omega, med)) @ a
    res = op @ f + inc
    return np.max(np.abs(res)) / np.max(np.abs(inc))


def eval_field(x, src, f, x0, omega, med, Sw, amps=None, chunk=4000):
    """Total pressure  p = Sw * [ sum_m a_m G(x, x0_m) + sum_s f_s G(x, y_s) ]."""
    X0, a = _srcs(x0, amps)
    out = np.zeros(len(x), dtype=complex)
    for i in range(0, len(x), chunk):
        xs = x[i:i + chunk]
        out[i:i + chunk] = (G_monopole(xs, X0, omega, med) @ a
                            + G_monopole(xs, src, omega, med) @ f)
    return Sw * out


def envelope(Pf, Nt):
    """|analytic signal| from a one-sided spectrum Pf (nbins, npts):
    the instantaneous amplitude envelope (for SPL rendering)."""
    nb = Pf.shape[0]
    full = np.zeros((Nt,) + Pf.shape[1:], dtype=complex)
    full[:nb] = 2.0 * Pf
    full[0] *= 0.5
    if Nt % 2 == 0:
        full[nb - 1] *= 0.5
    return np.abs(np.fft.ifft(full, axis=0))


# ============================================================================
# 6. Room-acoustics metrics (ISO 3382 style)
# ============================================================================
def face_areas(L):
    Lx, Ly, Lz = L
    return np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])


def statistical_rt60(L, face_mats, f):
    """Sabine & Eyring reverberation times at frequency f."""
    S = face_areas(L)
    V = L[0] * L[1] * L[2]
    a = np.array([alpha_of(m, f) for m in face_mats])
    A = float(S @ a)
    abar = A / S.sum()
    sab = 0.161 * V / max(A, 1e-9)
    eyr = 0.161 * V / max(-S.sum() * np.log(max(1 - abar, 1e-9)), 1e-9)
    return sab, eyr, abar


def room_modes(L, c=343.0, fmax=200.0, nmax=8):
    """Analytic rigid-box eigenfrequencies up to fmax with type labels."""
    out = []
    for i in range(nmax):
        for j in range(nmax):
            for k in range(nmax):
                if i == j == k == 0:
                    continue
                f = 0.5 * c * np.sqrt((i / L[0]) ** 2 + (j / L[1]) ** 2 + (k / L[2]) ** 2)
                if f <= fmax:
                    nz = (i > 0) + (j > 0) + (k > 0)
                    out.append(dict(f=float(f), n=(i, j, k),
                                    type=('axial', 'tangential', 'oblique')[nz - 1]))
    return sorted(out, key=lambda d: d['f'])


def metrics_from_ir(h, dt):
    """Schroeder-integration metrics from a (band-limited) impulse response.
    Returns dict; rt60_t20 is None when the time window can't reach -25 dB."""
    e = h.astype(float) ** 2
    if e.sum() <= 0:
        return {}
    E = np.cumsum(e[::-1])[::-1]
    S = 10 * np.log10(np.maximum(E / E[0], 1e-12))
    t = np.arange(len(h)) * dt

    def fit_rt(d0, d1):
        m = (S <= d0) & (S >= d1)
        if m.sum() < 4 or S.min() > d1 + 1.0:
            return None
        p = np.polyfit(t[m], S[m], 1)
        return float(-60.0 / p[0]) if p[0] < 0 else None

    i50 = max(1, int(round(0.050 / dt)))
    i80 = max(1, int(round(0.080 / dt)))
    early50, late50 = e[:i50].sum(), e[i50:].sum()
    early80, late80 = e[:i80].sum(), e[i80:].sum()
    return dict(
        rt60_t20=fit_rt(-5.0, -25.0),
        edt=fit_rt(-0.5, -10.0),
        c50=float(10 * np.log10(max(early50, 1e-12) / max(late50, 1e-12))),
        c80=float(10 * np.log10(max(early80, 1e-12) / max(late80, 1e-12))),
        d50=float(early50 / e.sum()),
        schroeder_db=np.round(S, 2).tolist(),
    )


# ============================================================================
# 7. Driver: full time-domain room simulation
# ============================================================================
DEFAULT_MATS = ('brick', 'wood floor', 'gypsum drywall')   # walls, floor, ceiling


def run(L=(6.0, 5.0, 3.0), x0=(2.0, 2.0, 1.5), f0=80.0, materials=None,
        beta=None, signal='ricker', receivers=((4.5, 3.5, 1.2),),
        src_amps=None, n_wall=900, off_fac=4.0, Nt=224, dt=0.003, zslice=None,
        ng=(48, 40), med=None, geometry=None, walls=False, spl=False,
        progress=None, verbose=True):
    """Simulate one or more monopole sources in a room.

    x0         source position (3,) or several sources (n,3)
    src_amps   per-source amplitudes (e.g. (1,-1) = anti-phase stereo pair)
    materials  (walls, floor, ceiling) names from MATERIALS -> beta(f) per face
    beta       scalar admittance override (uniform, freq-independent); wins
    signal     'ricker' | 'toneburst' | 'click' | 'chirp'
    receivers  microphone positions -> impulse responses + |H(f)|
    walls      also evaluate p on floor + 2 back walls (box rooms only)
    spl        also return envelope (|analytic|) fields for dB rendering
    """
    med = med or Air()
    x0, src_amps = _srcs(x0, src_amps)
    mw, mf, mc = materials or DEFAULT_MATS
    face_mats = [mw, mw, mw, mw, mf, mc]
    if geometry is None:
        wall, nrm, spacing, fid = box_room(L, n_wall)
        chk, chkn, _, chk_fid = box_room(L, max(60, n_wall // 4))
    else:
        wall, nrm, spacing = geometry
        fid = np.zeros(len(wall), int)
        face_mats = [mw] * 6
        chk, chkn, chk_fid = wall[::4], nrm[::4], fid[::4]
    src = wall + off_fac * spacing * nrm            # sources outside the room

    def beta_vec(f, ids):
        if beta is not None:
            return np.full(len(ids), float(beta))
        bf = np.array([beta_of(m, f) for m in face_mats])
        return bf[ids]

    # frequency band the wall grid can resolve (~6 pts per wavelength)
    f_cap = med.c / (6.0 * spacing)
    t = np.arange(Nt) * dt
    sig, t0 = source_signal(signal, t, f0)
    Rf = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(Nt, dt)
    Rmax = np.abs(Rf).max()

    # evaluation points: slice + optional wall faces + receivers
    zs = x0[0, 2] if zslice is None else float(zslice)
    gx = np.linspace(0.02 * L[0], 0.98 * L[0], ng[0])
    gy = np.linspace(0.02 * L[1], 0.98 * L[1], ng[1])
    GX, GY = np.meshgrid(gx, gy)
    Gpts = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, zs)], 1)

    wall_defs, wall_meshes = [], []
    if walls and geometry is None:
        dens = ng[0] / (0.96 * L[0])                 # match slice density
        for name, ax_u, ax_v, ax_f, val in (
                ('floor z=0', 0, 1, 2, 0.0),
                ('wall x=0', 1, 2, 0, 0.0),
                ('wall y=0', 0, 2, 1, 0.0)):
            nu = max(8, int(round(dens * L[ax_u])))
            nv = max(8, int(round(dens * L[ax_v])))
            us = np.linspace(0.02 * L[ax_u], 0.98 * L[ax_u], nu)
            vs = np.linspace(0.02 * L[ax_v], 0.98 * L[ax_v], nv)
            U, V = np.meshgrid(us, vs)
            P = np.zeros((U.size, 3))
            P[:, ax_u], P[:, ax_v], P[:, ax_f] = U.ravel(), V.ravel(), val
            M = [None, None, None]
            M[ax_u], M[ax_v], M[ax_f] = U, V, np.full_like(U, val)
            wall_defs.append((name, P))
            wall_meshes.append((name, M[0], M[1], M[2]))
    rx = np.atleast_2d(np.asarray(receivers, float))
    splits = np.cumsum([len(Gpts)] + [len(P) for _, P in wall_defs] + [len(rx)])[:-1]
    allpts = np.vstack([Gpts] + [P for _, P in wall_defs] + [rx])

    active = [kk for kk, fk in enumerate(freqs)
              if fk > 0 and np.abs(Rf[kk]) >= 3e-3 * Rmax and fk <= f_cap]
    Pf = np.zeros((len(freqs), len(allpts)), dtype=complex)
    nsolved, resmax, tw = 0, 0.0, time.time()
    for kk in active:
        fk = freqs[kk]
        om = 2 * np.pi * fk
        f = solve_freq(om, med, wall, nrm, src, x0, beta_vec(fk, fid), src_amps)
        if np.abs(Rf[kk]) > 0.5 * Rmax:              # check near dominant freq
            resmax = max(resmax, bc_residual(om, med, chk, chkn, src, x0,
                                             beta_vec(fk, chk_fid), f, src_amps))
        Pf[kk] = eval_field(allpts, src, f, x0, om, med, Rf[kk], src_amps)
        nsolved += 1
        if progress:
            progress(nsolved, len(active))

    pt_all = np.fft.irfft(Pf, n=Nt, axis=0)
    parts = np.split(pt_all, splits, axis=1)
    pt = parts[0].reshape(Nt, ng[1], ng[0])
    wall_fields = [dict(name=nm, X=X, Y=Y, Z=Z, pt=p.reshape(Nt, *X.shape))
                   for (nm, X, Y, Z), p in zip(wall_meshes, parts[1:-1])]
    rx_p = parts[-1]                                 # (Nt, n_receivers)

    env = env_walls = None
    if spl:
        env_all = envelope(Pf, Nt)
        eparts = np.split(env_all, splits, axis=1)
        env = eparts[0].reshape(Nt, ng[1], ng[0])
        env_walls = [e.reshape(Nt, *w['X'].shape)
                     for e, w in zip(eparts[1:-1], wall_fields)]

    # transfer function at the first receiver (normalised by source spectrum)
    frf_f = [float(freqs[kk]) for kk in active]
    frf_H = [float(np.abs(Pf[kk, -len(rx)]) / np.abs(Rf[kk])) for kk in active]

    metrics = metrics_from_ir(rx_p[:, 0], dt)
    if geometry is None:
        sab, eyr, abar = statistical_rt60(L, face_mats, max(f0, 63.0))
        metrics.update(rt60_sabine=sab, rt60_eyring=eyr, mean_alpha=abar,
                       volume=L[0] * L[1] * L[2], area=float(face_areas(L).sum()))
        rt_ref = metrics.get('rt60_t20') or sab
        metrics['f_schroeder'] = float(2000 * np.sqrt(rt_ref / metrics['volume']))
        metrics['bands'] = [dict(f=b, **{k: float(v) for k, v in
                            zip(('sabine', 'eyring'),
                                statistical_rt60(L, face_mats, b)[:2])})
                            for b in BANDS if b <= max(f_cap, 63.0)]
        modes = room_modes(L, med.c, fmax=min(f_cap, 200.0))
    else:
        modes = []

    if verbose:
        print(f"[room] {len(wall)} wall pts, {nsolved} freqs (cap {f_cap:.0f} Hz) "
              f"in {time.time() - tw:.1f}s, BC residual ~{resmax:.1e}")
    return dict(t=t, X=GX, Y=GY, z=zs, pt=pt, walls=wall_fields, env=env,
                env_walls=env_walls, rx=rx, rx_p=rx_p, frf_f=frf_f, frf_H=frf_H,
                sig=sig, metrics=metrics, modes=modes, L=L, x0=x0, f0=f0,
                signal=signal, face_mats=face_mats, resmax=resmax,
                f_cap=f_cap, nsolved=nsolved, dt=dt)


# ============================================================================
# 8. Standalone animation (no web app needed)
# ============================================================================
def animate(res, stride=2, out="room3d.html"):
    """3D animated pressure slice + walls inside the room wireframe."""
    import plotly.graph_objects as go
    X, Y, pt, t, L, x0 = res['X'], res['Y'], res['pt'], res['t'], res['L'], res['x0']
    wallf = res.get('walls') or []
    vmax = max([np.percentile(np.abs(pt), 99.5)]
               + [np.percentile(np.abs(w['pt']), 99.5) for w in wallf])
    relief = 0.25 * L[2]
    idx = list(range(0, pt.shape[0], stride))
    surf = lambda i: go.Surface(
        x=X, y=Y, z=res['z'] + np.clip(pt[i], -vmax, vmax) / vmax * relief,
        surfacecolor=np.clip(pt[i], -vmax, vmax), cmin=-vmax, cmax=vmax,
        colorscale='RdBu', reversescale=True, colorbar=dict(title='p', len=0.6))
    wsurf = lambda w, i: go.Surface(
        x=w['X'], y=w['Y'], z=w['Z'],
        surfacecolor=np.clip(w['pt'][i], -vmax, vmax), cmin=-vmax, cmax=vmax,
        colorscale='RdBu', reversescale=True, showscale=False, name=w['name'])
    animated = lambda i: [surf(i)] + [wsurf(w, i) for w in wallf]
    cx, cy, cz = np.meshgrid([0, L[0]], [0, L[1]], [0, L[2]], indexing='ij')
    corners = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], 1)
    ex, ey, ez = [], [], []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                ex += [corners[i][0], corners[j][0], None]
                ey += [corners[i][1], corners[j][1], None]
                ez += [corners[i][2], corners[j][2], None]
    wire = go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
                        line=dict(color='gray', width=3), showlegend=False)
    X0 = np.atleast_2d(x0)
    spk = go.Scatter3d(x=X0[:, 0], y=X0[:, 1], z=X0[:, 2], mode='markers',
                       marker=dict(size=6, color='yellow'), name='source')
    frames = [go.Frame(data=animated(i), name=f"{t[i]:.3f}") for i in idx]
    fig = go.Figure(data=animated(idx[0]) + [wire, spk], frames=frames)
    steps = [dict(method='animate', label=f"{t[i]*1000:.0f}",
                  args=[[f"{t[i]:.3f}"], dict(mode='immediate',
                        frame=dict(duration=0, redraw=True))]) for i in idx]
    fig.update_layout(
        title=f"Room {L[0]}x{L[1]}x{L[2]} m — {res['signal']} f0={res['f0']} Hz",
        template='plotly_dark',
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))),
        updatemenus=[dict(type='buttons', x=0.05, y=0.05, buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(y=0, x=0.15, len=0.8,
                      currentvalue=dict(prefix='t = ', suffix=' ms'), steps=steps)])
    fig.write_html(out, include_plotlyjs=True, auto_play=False)
    print(f"[animate] wrote {out}")


# ============================================================================
# 9. Self-tests
# ============================================================================
def self_test():
    med = Air()
    ok = True

    # 1) G satisfies the Helmholtz equation (finite differences)
    om = 2 * np.pi * 100.0
    k = om / med.c
    y = np.zeros((1, 3)); xp = np.array([1.3, 0.4, 0.7]); h = 1e-4
    lap = 0.0
    for ax in range(3):
        e = np.zeros(3); e[ax] = h
        lap += (G_monopole((xp + e)[None], y, om, med)[0, 0]
                + G_monopole((xp - e)[None], y, om, med)[0, 0]
                - 2 * G_monopole(xp[None], y, om, med)[0, 0]) / h**2
    err = abs(lap + k * k * G_monopole(xp[None], y, om, med)[0, 0]) / abs(k * k)
    print(f"  [1] Helmholtz eq residual (FD):        {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")
    ok &= err < 1e-4

    # 2) rigid-wall BC residual on independent points
    res = run(L=(4.0, 3.0, 2.5), x0=(1.5, 1.2, 1.2), f0=60.0, beta=0.0,
              n_wall=420, Nt=96, dt=0.003, ng=(20, 16), verbose=False)
    print(f"  [2] rigid-wall BC residual:            {res['resmax']:.2e}  "
          f"{'OK' if res['resmax'] < 0.03 else 'FAIL'}")
    ok &= res['resmax'] < 0.03

    # 3) absorption sanity: tail energy dies with absorbing walls
    resA = run(L=(4.0, 3.0, 2.5), x0=(1.5, 1.2, 1.2), f0=60.0, beta=0.5,
               n_wall=420, Nt=96, dt=0.003, ng=(20, 16), verbose=False)
    tail = slice(60, 96)
    r = np.sum(resA['pt'][tail] ** 2) / np.sum(res['pt'][tail] ** 2)
    print(f"  [3] late energy soft/rigid walls:      {r:.3f}  {'OK' if r < 0.5 else 'FAIL'}")
    ok &= r < 0.5

    # 4) modal accuracy: rigid box (4,3,2.5) first axial-x mode at c/(2Lx)
    fm = med.c / (2 * 4.0)                            # 42.875 Hz
    wall, nrm, sp, fidv = box_room((4.0, 3.0, 2.5), 350)
    src = wall + 4.0 * sp * nrm
    x0 = np.array([1.1, 0.9, 0.8]); rxp = np.array([[3.7, 2.7, 2.2]])
    amp = []
    for f in (fm * 0.85, fm, fm * 1.15):
        om = 2 * np.pi * f
        fs = solve_freq(om, med, wall, nrm, src, x0, np.full(len(wall), 0.01))
        amp.append(abs(eval_field(rxp, src, fs, x0, om, med, 1.0)[0]))
    print(f"  [4] axial-mode resonance @ {fm:.1f} Hz:   "
          f"{amp[0]:.2f} < {amp[1]:.2f} > {amp[2]:.2f}  "
          f"{'OK' if amp[1] > 2 * max(amp[0], amp[2]) else 'FAIL'}")
    ok &= amp[1] > 2 * max(amp[0], amp[2])

    print("  self-test:", "ALL PASSED" if ok else "FAILURES ABOVE")
    return ok


def main():
    ap = argparse.ArgumentParser(description="3D room acoustics (wave-based MFS)")
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--f0', type=float, default=80.0)
    ap.add_argument('--signal', default='ricker')
    ap.add_argument('--out', default='room3d.html')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if self_test() else 1)
    res = run(f0=a.f0, signal=a.signal, walls=True)
    animate(res, out=a.out)


if __name__ == '__main__':
    main()
