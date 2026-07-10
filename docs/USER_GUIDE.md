# RoomWave Studio — user guide

## Running

```bash
cd core
python app.py          # needs numpy + plotly installed
```

Open **http://localhost:8747**. The server is Python's standard library
(`http.server`) — no Flask/Django required. Plotly.js is served from your
installed `plotly` package, so the app works fully offline.

Long simulations run asynchronously: the UI shows a live progress bar
(one tick per solved frequency), and you can watch the solver log in the
terminal.

## The sidebar, top to bottom

**Preset** — five ready-made scenarios: Living room, Recording studio
(absorptive), Bathroom (tiled, very live), Small hall, Stereo speaker pair
(two sources). A preset fills every field; you can tweak afterwards.

**Room** — box dimensions in metres. Or point *OBJ mesh file* at a closed,
outward-normal triangle mesh to simulate an arbitrary shape (leave empty for
the box).

**Surface materials** — independent wall / floor / ceiling materials with
real octave-band absorption data. Material choice is the single biggest
lever on the room's sound: compare `ceramic tile` everywhere (bathroom
reverb) against `acoustic panel` (studio dryness) and watch RT60 move.

**Source** — position; signal type; centre frequency `f0`; speaker level at
1 m (dB SPL, 50–110).

- **Speaker tone (steady)** *(default)* — a continuously playing sine.
  Solves in ~1 s. The 3D view loops seamlessly. **Tip:** set `f0` to a mode
  frequency from the Acoustic Metrics tab (e.g. the first axial mode) and
  watch the room lock into a standing wave; with *Field scale = SPL* you get
  the time-invariant node/antinode map used for speaker and listener
  placement.
- **Ricker pulse / click / chirp** — broadband transients: expanding
  wavefronts, reflections, reverberant decay. These populate the impulse
  response, spectrogram, transfer function and decay metrics.
- **Tone burst** — a ~4-cycle narrowband packet; good for watching a single
  frequency propagate as a wave train.

**Second source** — enables a stereo pair with in-phase or **anti-phase**
polarity. Anti-phase at low frequency shows the cancellation plane between
the speakers — the physics behind "why does the bass disappear when I sit
here".

**Receiver mic** — the analysis point for impulse response, transfer
function and SPL readouts. Shown as the cyan `R` marker in 3D.

**Visualisation** — walls+slice cutaway or slice only; Pressure (Pa,
diverging colormap) or SPL (absolute dB re 20 µPa); colormap; slice height
(defaults to source height); relief exaggeration slider (live, no
recompute).

**Quality** — wall-point budget and time window:

| preset | wall pts | band | typical time (pulse) | steady tone |
|---|---|---|---|---|
| Preview | 450 | ≤ ~110 Hz | ~10 s | ~1 s |
| Standard | 900 | ≤ ~155 Hz | ~2 min | ~2 s |
| High | 1400 | ≤ ~190 Hz | ~4 min | ~5 s |

The status panel always shows the number of solved frequencies and the
**boundary-condition residual** of the run — the solver's honesty number
(0.3–3 % is normal).

## The tabs

**3D Wave Field** — animated pressure. Play/pause, speed, time slider.
Yellow diamonds = sources, cyan dot = mic. Camera: drag to orbit, scroll to
zoom; the camera icon in the plot toolbar saves a PNG.

**Impulse Response** *(pulse signals)* — pressure at the mic (Pa), the
source signal, and the Schroeder decay curve (dB, right axis). Below it a
spectrogram shows how each frequency decays — the low-frequency ridges that
ring longest are the room modes. Buttons: 🔊 play the IR; 🥁 kick drum and 👏
clap convolved through the room (wet/dry slider); ⬇ WAV exports the IR as
audio.

**Frequency Response** *(pulse signals)* — |H(f)| from source to mic, dB.
Dashed vertical lines are the analytic rigid-room modes (red axial, amber
tangential, grey oblique) — simulated peaks landing on them is a live
validation. Toggles: mode lines, ⅓-octave smoothing, overlay of the
**previous run** for A/B comparison.

**Acoustic Metrics** — ISO 3382 cards (RT60 ×3, EDT, C50, C80, D50, SPL at
mic, Schroeder frequency, volume, mean absorption), each showing the **delta
against your previous run** in green/red; per-band Sabine/Eyring RT60 chart;
room-mode table; CSV and JSON export.

## A/B workflow

Every simulation keeps the previous one in memory. Change one thing —
carpet → ceramic tile, mic position, anti-phase — re-simulate, and read the
effect off the FRF overlay and the metric deltas. This is the core loop of
acoustic treatment design.

## Python API (no browser)

```python
import sys; sys.path.insert(0, 'core')
from acoustics import run, run_steady, animate, self_test

# transient: full time-domain simulation
res = run(L=(6,5,3), x0=(2,2,1.5), f0=80, signal='ricker',
          materials=('brick','wood floor','gypsum drywall'),
          receivers=((4.5,3.5,1.2),), spl_1m=85, n_wall=900, walls=True)
print(res['metrics'])            # RT60, EDT, C50, ... in a dict
animate(res, out='room3d.html')  # standalone interactive HTML

# steady state: one frequency, complex phasor field
st = run_steady(L=(6,5,3), x0=(2,2,1.5), f0=57.2, n_wall=900)
P = st['P']                      # complex pressure (Pa) on the slice grid

# two anti-phase sources
res = run(x0=[(1.8,1,1.2),(4.2,1,1.2)], src_amps=(1,-1), ...)

self_test()                      # the 4 verification checks
```

Key `run()` returns: `pt` (Nt×ny×nx pressure movie, Pa), `rx_p` (mic
signals), `frf_f`/`frf_H` (transfer function), `metrics`, `modes`, `resmax`
(BC residual), `walls` (wall-face movies when `walls=True`).

## Troubleshooting

- **"tone … exceeds the resolvable band"** — the steady tone is above the
  quality preset's frequency cap. Raise quality or lower `f0`.
- **"OBJ file not found"** — the mesh path field has stale text; clear it
  (the box room needs it empty). The app clears it on load.
- **429 "a simulation is already running"** — one solve at a time; wait for
  the progress bar.
- **Slow?** Use Preview while exploring, Standard/High for final numbers.
  Steady-tone mode is near-instant at any quality.
- **Port busy** — a previous server is alive: `fuser -k 8747/tcp`, then
  restart. (Do not `pkill -f app.py`; other software may match.)
- **RT60 T20 shows "window too short"** — the room decays more slowly than
  the simulated time window; the Sabine/Eyring predictions are still valid,
  or re-run at a quality with a longer window.
