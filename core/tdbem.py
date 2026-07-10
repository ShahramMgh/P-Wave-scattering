"""
tdbem.py — 3D elastic wave scattering in a homogeneous medium (time domain).

Phase 1 core: scattering of a transient plane elastic wave (Ricker pulse) by a
rigid scatterer in an unbounded homogeneous isotropic elastic solid, solved with
the Method of Fundamental Solutions (MFS) — the desingularized member of the
boundary-element family — driven by the *dynamic Kelvin (Stokes) fundamental
solution*. The transient response is obtained frequency-by-frequency and mapped
back to the time domain by inverse FFT (stable — no marching-on-in-time
instability). Results are rendered as a clean, self-contained 3D animation.

Why MFS for phase 1:
  * uses only the displacement fundamental solution U_ij(omega), which is
    verified here against the exact static Kelvin limit (see self_test);
  * needs NO singular surface integration;
  * spectrally accurate for smooth scatterers;
  * stable time-domain output via FFT.

Verified in this file (run `python tdbem.py --selftest`):
  1. U_ij(omega) -> exact static Kelvin tensor as omega->0        (~1e-4)
  2. rigid boundary condition residual at independent surface pts (~3e-4)
  3. incident plane-wave pulse reconstruction from spectrum       (~1e-13)
Convergence: the BC residual falls spectrally with n_surf when the interior
auxiliary sources sit at aux_ratio ~ 0.6 * a (0.6 well-conditioned; ~0.9 is not).

Roadmap (next phases): traction-free cavity (needs verified T_ij kernel),
half-space with free surface, CQ-BEM for large problems, SEM/BEM coupling.

Requires: numpy, scipy (optional), plotly.  Author scaffold: cleaned rewrite.
"""
from __future__ import annotations
import argparse
import numpy as np


# ============================================================================
# 1. Material
# ============================================================================
class Material:
    """Isotropic linear-elastic material. Speeds define everything."""
    def __init__(self, rho: float, c_p: float, c_s: float):
        self.rho, self.c_p, self.c_s = rho, c_p, c_s
        self.mu = rho * c_s * c_s
        self.lam = rho * c_p * c_p - 2.0 * self.mu
        self.nu = self.lam / (2.0 * (self.lam + self.mu))

    def __repr__(self):
        return f"Material(rho={self.rho}, c_p={self.c_p}, c_s={self.c_s}, nu={self.nu:.3f})"


# ============================================================================
# 2. Dynamic Kelvin (Stokes) fundamental solution  U_ij(x,y,omega)
#    Written as  U = A(r) delta_ij + B(r) e_i e_j   with e = (x-y)/r.
#    Convention e^{+i omega t}; outgoing waves ~ e^{-i k r}.
#    VERIFIED: A,B reduce to the exact static Kelvin tensor as omega -> 0.
# ============================================================================
def _phi_derivs(k, r):
    """phi = e^{-ikr}/r  and its first two r-derivatives (elementwise)."""
    E = np.exp(-1j * k * r)
    phi = E / r
    phi1 = -E * (1j * k * r + 1.0) / r**2
    phi2 = E * (-(k * k) / r + 2j * k / r**2 + 2.0 / r**3)
    return phi, phi1, phi2


def kelvin_AB(r, omega, m: Material):
    """Scalar coefficients A(r), B(r) of the dynamic Kelvin tensor (elementwise in r)."""
    kp, ks = omega / m.c_p, omega / m.c_s
    ps, ps1, ps2 = _phi_derivs(ks, r)
    pp, pp1, pp2 = _phi_derivs(kp, r)
    pre = 1.0 / (4.0 * np.pi * m.rho * omega**2)
    A = pre * ((ps1 - pp1) / r + ks * ks * ps)
    B = pre * ((ps2 - pp2) - (ps1 - pp1) / r)
    return A, B


def kelvin_U_blocks(x, y, omega, m: Material):
    """
    U as array (Nx, Ny, 3, 3): displacement component i at field point x[a]
    due to a unit harmonic point force component j at source y[b].
    x: (Nx,3), y: (Ny,3).
    """
    d = x[:, None, :] - y[None, :, :]          # (Nx,Ny,3)
    r = np.linalg.norm(d, axis=2)              # (Nx,Ny)
    e = d / r[..., None]
    A, B = kelvin_AB(r, omega, m)
    I = np.eye(3)
    return A[..., None, None] * I + B[..., None, None] * (e[..., :, None] * e[..., None, :])


def kelvin_U_static(r_vec, m: Material):
    """Exact static Kelvin tensor (verification reference)."""
    r = np.linalg.norm(r_vec); e = r_vec / r
    return (1.0 / (16 * np.pi * m.mu * (1 - m.nu) * r)) * (
        (3 - 4 * m.nu) * np.eye(3) + np.outer(e, e))


# ============================================================================
# 3. Geometry helpers
# ============================================================================
def fibonacci_sphere(n, radius=1.0, center=(0.0, 0.0, 0.0)):
    """Quasi-uniform points on a sphere (great for MFS collocation/sources)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    p = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1)
    return p * radius + np.asarray(center)


# ============================================================================
# 4. Incident field (transient plane wave assembled per frequency)
# ============================================================================
def incident_plane_P(x, omega, m: Material, direction, amp=1.0):
    """Plane P-wave, polarization along propagation direction n: u = amp*n*e^{-i kp n.x}."""
    n = np.asarray(direction, float); n = n / np.linalg.norm(n)
    kp = omega / m.c_p
    phase = np.exp(-1j * kp * (x @ n))
    return amp * phase[:, None] * n[None, :]


def ricker(t, f0, t0):
    """Ricker wavelet, central frequency f0, centered at t0 (correctly normalized)."""
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


# ============================================================================
# 5. MFS scattering solver (rigid scatterer: u_total = 0 on surface)
# ============================================================================
def rotated_fibonacci(n, radius=1.0, angle=0.3):
    """Fibonacci points rotated about z — for honest independent BC checks
    (a plain fibonacci_sphere(n2) partially aligns with the collocation spiral
    and under-reports the residual)."""
    p = fibonacci_sphere(n, radius)
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return p @ Rz.T


def solve_rigid_scatter(omega, m: Material, surf_pts, aux_pts, direction, amp=1.0):
    """
    Solve one frequency. Unknowns: point-force strengths f at interior aux
    sources so that u_incident + u_scattered = 0 at surface collocation points.
    Returns source strengths f (Ns,3).
    """
    U = kelvin_U_blocks(surf_pts, aux_pts, omega, m)   # (Nc,Ns,3,3)
    Nc, Ns = U.shape[0], U.shape[1]
    Amat = U.transpose(0, 2, 1, 3).reshape(3 * Nc, 3 * Ns)
    uinc = incident_plane_P(surf_pts, omega, m, direction, amp)
    rhs = -uinc.reshape(3 * Nc)
    f, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)
    return f.reshape(Ns, 3)


def eval_field(x, aux_pts, f, omega, m: Material, direction, amp=1.0):
    """Total displacement (incident + scattered) at points x for one frequency."""
    U = kelvin_U_blocks(x, aux_pts, omega, m)
    u_sc = np.einsum('xsij,sj->xi', U, f)
    u_in = incident_plane_P(x, omega, m, direction, amp)
    return u_in + u_sc


# ============================================================================
# 6. Time-domain driver:  Ricker pulse -> FFT -> per-omega solve -> IFFT
# ============================================================================
def run_scattering(m: Material,
                   a=1.0,
                   direction=(0.0, 0.0, 1.0),
                   f0=0.6, t0=3.0,
                   Nt=200, dt=0.06,
                   n_surf=250, aux_ratio=0.6,
                   grid_n=70, half=4.0,
                   verbose=True):
    """
    Compute the transient scattered wavefield on a 2D slice (x-z plane, y=0).
    Returns a dict with the time axis and wavefield ut of shape (Nt, grid_n*grid_n, 3).
    """
    direction = np.asarray(direction, float)
    t = np.arange(Nt) * dt
    src = ricker(t, f0, t0)
    R = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(Nt, dt)
    omegas = 2 * np.pi * freqs

    surf_pts = fibonacci_sphere(n_surf, a)
    aux_pts = fibonacci_sphere(n_surf, a * aux_ratio)

    xs = np.linspace(-half, half, grid_n)
    zs = np.linspace(-half, half, grid_n)
    X, Z = np.meshgrid(xs, zs)
    G = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    inside = np.linalg.norm(G, axis=1) < a * 1.02
    Gout = G[~inside]

    Uf = np.zeros((len(freqs), len(G), 3), dtype=complex)     # total field
    Uf_sc = np.zeros((len(freqs), len(G), 3), dtype=complex)  # scattered (reflection) only
    Rmax = np.max(np.abs(R))
    nsolved = 0
    for k, om in enumerate(omegas):
        if freqs[k] == 0 or abs(R[k]) < 1e-12 * Rmax:
            continue
        f = solve_rigid_scatter(om, m, surf_pts, aux_pts, direction, amp=R[k])
        u = eval_field(Gout, aux_pts, f, om, m, direction, amp=R[k])       # total
        u_in = incident_plane_P(Gout, om, m, direction, amp=R[k])          # incident
        full = np.zeros((len(G), 3), dtype=complex); full[~inside] = u
        full_sc = np.zeros((len(G), 3), dtype=complex); full_sc[~inside] = u - u_in
        Uf[k] = full; Uf_sc[k] = full_sc
        nsolved += 1
    if verbose:
        print(f"[run] solved {nsolved} frequencies, {grid_n}x{grid_n} slice, {n_surf} sources, "
              f"incidence {np.asarray(direction, float)}")

    ut = np.fft.irfft(Uf, n=Nt, axis=0)          # (Nt, Ngrid, 3) real — total
    ut_sc = np.fft.irfft(Uf_sc, n=Nt, axis=0)    # scattered / reflected only
    ut[:, inside, :] = np.nan
    ut_sc[:, inside, :] = np.nan
    return dict(t=t, src=src, X=X, Z=Z, inside=inside, ut=ut, ut_sc=ut_sc,
                grid_n=grid_n, a=a, half=half, m=m, direction=direction)


# ============================================================================
# 7. Clean 3D animation (self-contained HTML, no server)
# ============================================================================
def animate(res, comp=2, field='total', stride=2, relief=1.2, out="scattering_3d.html"):
    """
    Render the x-z slice as a rippling 3D sheet (height & color = displacement
    component `comp`: 0=u_x, 1=u_y, 2=u_z) with the scatterer as a grey sphere.
    field='total' shows incident+reflected; field='scattered' shows the
    reflection pattern alone. Writes self-contained HTML (Play + time slider).
    """
    import plotly.graph_objects as go
    key = 'ut_sc' if field == 'scattered' else 'ut'
    ut, X, Z, a, t = res[key], res['X'], res['Z'], res['a'], res['t']
    gn = res['grid_n']
    F = ut[..., comp].reshape(-1, gn, gn)
    vmax = np.nanmax(np.abs(F)) * 0.9
    ky = relief * a / vmax
    idx = list(range(0, F.shape[0], stride))

    u = np.linspace(0, 2 * np.pi, 30); v = np.linspace(0, np.pi, 30)
    sx = a * np.outer(np.cos(u), np.sin(v))
    sy = a * np.outer(np.sin(u), np.sin(v))
    sz = a * np.outer(np.ones_like(u), np.cos(v))
    sphere = go.Surface(x=sx, y=sy, z=sz, surfacecolor=np.zeros_like(sx),
                        colorscale=[[0, '#888'], [1, '#888']], showscale=False,
                        lighting=dict(ambient=0.8), name='scatterer')

    def sheet(fr):
        return go.Surface(x=X, y=np.nan_to_num(fr) * ky, z=Z, surfacecolor=fr,
                          cmin=-vmax, cmax=vmax, colorscale='RdBu', reversescale=True,
                          colorbar=dict(title='u', len=0.6),
                          lighting=dict(ambient=0.6, diffuse=0.5), name='wavefield')

    frames = [go.Frame(data=[sheet(F[i]), sphere], name=f"{t[i]:.2f}") for i in idx]
    fig = go.Figure(data=[sheet(F[idx[0]]), sphere], frames=frames)
    steps = [dict(method='animate', label=f"{t[i]:.1f}",
                  args=[[f"{t[i]:.2f}"], dict(mode='immediate',
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))]) for i in idx]
    d = np.asarray(res['direction'], float); d = d / np.linalg.norm(d)
    fig.update_layout(
        title=f"3D elastic wave scattering by a rigid sphere — {field} field, "
              f"incidence ({d[0]:.0f},{d[1]:.0f},{d[2]:.0f}) (time domain)",
        template="plotly_dark",
        scene=dict(xaxis_title='x', yaxis_title='displacement relief', zaxis_title='z',
                   aspectmode='manual', aspectratio=dict(x=1, y=0.5, z=1),
                   camera=dict(eye=dict(x=1.7, y=1.3, z=0.9))),
        updatemenus=[dict(type='buttons', showactive=False, x=0.05, y=0.05, buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=60, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='❚❚ Pause', method='animate',
                 args=[[None], dict(mode='immediate',
                                    frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, y=0, x=0.15, len=0.8,
                      currentvalue=dict(prefix='t = '), steps=steps)])
    fig.write_html(out, include_plotlyjs=True, auto_play=False)
    print(f"[animate] wrote {out}")
    return out


# ============================================================================
# 8. Self-tests (prove "precise")
# ============================================================================
def self_test():
    m = Material(2000.0, 3000.0, 1600.0)
    # (1) static Kelvin limit
    rv = np.array([[1.3, -0.7, 0.9]]); y = np.array([[0.0, 0.0, 0.0]])
    Ustat = kelvin_U_static(rv[0], m)
    U = kelvin_U_blocks(rv, y, 1e-3, m)[0, 0].real
    e1 = np.max(np.abs(U - Ustat)) / np.max(np.abs(Ustat))
    print(f"[test1] dynamic U -> static Kelvin (omega->0): relerr = {e1:.2e}")
    assert e1 < 1e-4

    # (2) rigid BC residual at DENSE INDEPENDENT points (honest convergence check)
    mm = Material(1.0, 1.87, 1.0)
    surf = fibonacci_sphere(300, 1.0); aux = fibonacci_sphere(300, 0.6)
    f = solve_rigid_scatter(6.0, mm, surf, aux, (0, 0, 1.0))
    chk = rotated_fibonacci(500, 1.0, angle=0.37)
    u = eval_field(chk, aux, f, 6.0, mm, (0, 0, 1.0))
    uin = incident_plane_P(chk, 6.0, mm, (0, 0, 1.0))
    e2 = np.max(np.linalg.norm(u, axis=1)) / np.max(np.linalg.norm(uin, axis=1))
    print(f"[test2] rigid BC residual at dense independent surface pts: {e2:.2e}")
    assert e2 < 1e-2

    # (3) incident pulse reconstruction
    Nt, dt, f0, t0, c_p = 200, 0.06, 0.6, 3.0, 1.87
    tt = np.arange(Nt) * dt
    R = np.fft.rfft(ricker(tt, f0, t0)); fr = np.fft.rfftfreq(Nt, dt)
    kp = 2 * np.pi * fr / c_p
    recon = np.fft.irfft(R * np.exp(-1j * kp * 2.5), n=Nt)
    analytic = ricker(tt, f0, t0 + 2.5 / c_p)
    e3 = np.max(np.abs(recon - analytic)) / np.max(np.abs(analytic))
    print(f"[test3] incident plane-wave pulse reconstruction relerr: {e3:.2e}")
    assert e3 < 1e-9
    print("all self-tests passed ✓")


# ============================================================================
# 9. CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="3D elastic wave scattering (time domain).")
    ap.add_argument("--selftest", action="store_true", help="run verification tests and exit")
    ap.add_argument("--out", default="scattering_3d.html")
    ap.add_argument("--nt", type=int, default=200)
    ap.add_argument("--nsurf", type=int, default=250)
    ap.add_argument("--grid", type=int, default=70)
    ap.add_argument("--f0", type=float, default=0.6)
    ap.add_argument("--dir", default="x", choices=["x", "z"],
                    help="incidence direction: x=horizontal (reflection view), z=vertical")
    ap.add_argument("--field", default="total", choices=["total", "scattered"],
                    help="scattered = reflection pattern only")
    args = ap.parse_args()

    if args.selftest:
        self_test(); return

    # nondimensional medium: rho=1, c_s=1, c_p=1.87 (nu~0.3), unit sphere
    m = Material(1.0, 1.87, 1.0)
    direction = {"x": (1.0, 0.0, 0.0), "z": (0.0, 0.0, 1.0)}[args.dir]
    res = run_scattering(m, direction=direction, f0=args.f0,
                         Nt=args.nt, n_surf=args.nsurf, grid_n=args.grid)
    animate(res, field=args.field, out=args.out)


if __name__ == "__main__":
    main()
