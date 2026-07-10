# core/ — RoomWave Studio engine & app

Full documentation lives one level up:

- **[../README.md](../README.md)** — overview, quick start, features
- **[../docs/PHYSICS.md](../docs/PHYSICS.md)** — formulation, calibration,
  metrics, verification, references
- **[../docs/USER_GUIDE.md](../docs/USER_GUIDE.md)** — UI walkthrough,
  Python API, troubleshooting

| file | role |
|---|---|
| `acoustics.py` | physics engine: scalar Helmholtz MFS, frequency-dependent impedance materials, transient + steady solvers, ISO 3382 metrics, `--selftest` |
| `app.py` | RoomWave Studio local web app (stdlib `http.server` + Plotly.js), port 8747 |
| `tdbem.py` | phase-1 elastodynamic solver: Ricker P-wave scattering off a rigid sphere, dynamic Kelvin/Stokes `U_ij(ω)`, MFS + inverse FFT, `--selftest` |
| `tdbem_core.ipynb` | historical notebook the project started from (physically invalid; provenance only) |

```bash
python app.py                 # web app  -> http://localhost:8747
python acoustics.py --selftest
python tdbem.py --selftest
```

Environment note (this machine): use `/home/shm/anaconda3/bin/python`
(numpy + plotly). Flask and matplotlib are broken in that env — the app
deliberately depends on neither.

## Elastic solver status (`tdbem.py`)

Verified phase 1: dynamic Kelvin tensor → static limit (~1e-4), rigid BC
residual ~3e-4, pulse reconstruction ~1e-13. Phase-2 work (traction kernel
`T_ij` validated against numerical differentiation of `U`, oblique P/SV
free-field with post-critical evanescent branches, free-surface Gaussian-hill
site response — topographic amplification ×1.10, BC residual 4.5e-3) was
completed and verified in session scratch space but is not yet merged here.
