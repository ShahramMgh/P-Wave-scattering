# RoomWave Studio — wave-based 3D room acoustics analyzer

Professional low-frequency room-acoustics analysis in the browser, powered by a
verified scalar-Helmholtz MFS engine. (Wave-based = the regime where ray/image
methods are invalid; the same approach commercial wave solvers use below the
Schroeder frequency.)

| file | role |
|---|---|
| `app.py` | **RoomWave Studio** web app (stdlib http.server + Plotly.js) |
| `acoustics.py` | physics engine: Helmholtz MFS, materials, ISO 3382 metrics |
| `tdbem.py` | earlier elastic/seismic solver (phase 1, verified, kept) |

## Run

```bash
PY=/home/shm/anaconda3/bin/python
$PY app.py            # open http://localhost:8747
```

## Features

- **3D wave field**: animated pressure on a slice + the wall surfaces
  (cutaway), pressure or SPL(dB) scale, playback speed, time slider
- **Per-surface materials** with published octave-band absorption data
  (concrete, brick, tile, glass, drywall, wood, carpet, curtain, acoustic
  panel, ceiling tile, open window) → frequency-dependent impedance BC
  `dp/dn + ik β(f) p = 0`, `β = (1−√(1−α))/(1+√(1−α))`
- **Receiver mic** → impulse response + Schroeder decay curve + transfer
  function |H(f)| overlaid with the analytic room-mode frequencies
- **ISO 3382 metrics**: RT60 three ways (Sabine, Eyring, T20 via Schroeder
  integration of the simulated IR), EDT, C50, C80, D50, Schroeder frequency,
  mode table
- **Auralization** (WebAudio): listen to the impulse response, or a kick drum
  convolved through the simulated room
- **Presets**: living room, recording studio, bathroom, small hall
- Source signals: Ricker, tone burst, click, chirp; arbitrary closed `.obj`
  room meshes supported (outward normals)
- Quality tiers: preview ~10 s (band ≤ ~110 Hz) / standard ~2 min (≤ ~155 Hz)
  / high ~4 min (≤ ~190 Hz) on a laptop; live progress bar; honest BC residual
  reported for every run

## Verification (`$PY acoustics.py --selftest`)

1. Green's function satisfies the Helmholtz equation (FD, ~3e-9)
2. rigid-wall BC residual on independent wall points (~1e-2)
3. absorbing walls kill the late reverberant energy vs rigid
4. **modal accuracy**: resonance at the analytic axial mode c/2Lx (42.9 Hz),
   6× above neighbouring frequencies

Cross-check on a live run (6×5×3 brick/wood/drywall): RT60 Sabine 0.94 s,
Eyring 0.88 s, simulated T20 0.79 s — independent statistical vs wave-based
estimates agree; FRF peaks land on the analytic mode lines (28.6, 34.3,
44.6 Hz …).

## Physics notes

- Frequency-domain MFS + inverse FFT: unconditionally stable, no singular
  integration; Tikhonov-regularised square solve; sources offset 4× spacing
  (the well-conditioned MFS regime).
- Band cap = c/(6·spacing): laptop budgets resolve up to ~200 Hz. Higher
  bands need fast solvers (FMM/ℋ-matrix) — roadmap; above the Schroeder
  frequency a geometric method is the right tool anyway (hybrid, as in
  commercial practice).
- Flask is broken in this env → stdlib `http.server`; plotly.js served from
  the installed plotly package. Long solves are async: POST /simulate,
  poll /progress, GET /result.

## Legacy: elastic solver (`tdbem.py`)

Phase-1 seismic scattering (rigid sphere, dynamic Kelvin U_ij(ω), MFS+FFT),
all self-tests pass. Phase-2 (traction kernel, oblique free-field,
free-surface hill, amplification ×1.10) was verified in session scratch but
not merged. `tdbem_core.ipynb` is the old invalid notebook, provenance only.
