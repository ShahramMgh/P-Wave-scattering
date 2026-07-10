# RoomWave Studio — physics and numerics

This document specifies the mathematical model, the numerical method, the
physical calibration, the acoustic metrics, and the verification evidence.

## 1. Governing equation

Sound in air at room amplitudes is linear acoustics. With time convention
`e^{+iωt}` (all fields `X(x,t) = Re[X̂(x,ω) e^{iωt}]`), the pressure phasor
satisfies the **Helmholtz equation**

```
∇²p̂ + k² p̂ = 0,          k = ω / c,     c = 343 m/s,  ρ = 1.2 kg/m³
```

in the room interior, excited by monopole sources. The free-space Green's
function (outgoing for this time convention) is

```
G(x, y; ω) = e^{-ikr} / (4πr),      r = |x − y|
```

Self-test 1 verifies `∇²G + k²G = 0` by finite differences to ~3·10⁻⁹.

## 2. Boundary condition: locally-reacting impedance walls

Each wall is a locally-reacting surface with specific impedance `Z(f)`.
Momentum balance `iωρ v = −∇p` and the wall relation `v·n = p/Z` give

```
∂p/∂n + i k β(f) p = 0        on the wall,   β = ρc / Z   (normalised admittance)
```

with outward normal `n`. `β = 0` is a rigid wall (perfect reflection),
`β = 1` a matched (fully absorbing at normal incidence) surface.

### Materials → β(f)

Surfaces are described by published octave-band absorption coefficients
`α(f)` at 63/125/250/500 Hz (building-acoustics tables, e.g. Vorländer,
*Auralization*, Springer 2008; Long, *Architectural Acoustics*, 2nd ed.).
The normal-incidence energy relation converts α to admittance:

```
|R|² = 1 − α,     R = (1 − β)/(1 + β)   ⇒   β = (1 − √(1−α)) / (1 + √(1−α))
```

`α(f)` is interpolated in log-frequency between bands and clamped at 0.99.
Each of the six faces (or every mesh facet group) can carry its own material,
and β is re-evaluated at **every solved frequency** — frequency-dependent
absorption is exact within the model, not band-averaged.

*Approximations*: local reaction (no lateral wave propagation inside the
wall) and the normal-incidence α→β conversion. Both are standard in
engineering room acoustics; random-incidence corrections would shift β by
O(10–30 %).

## 3. Numerical method: Method of Fundamental Solutions (MFS)

The scattered field is represented by `N` fictitious monopoles at points
`y_s` placed **outside** the room, offset from the wall along the outward
normal:

```
p̂(x) = Σ_m a_m G(x, x0_m) + Σ_s f_s G(x, y_s)
        └─ real sources ─┘   └─ fictitious boundary sources ─┘
```

Because the fictitious sources are outside the domain, the representation
satisfies the Helmholtz equation exactly in the interior and **no singular
integrals ever arise** (the classic advantage of MFS over collocation BEM;
Fairweather & Karageorghis 1998; Alves 2009; Antunes 2018). The strengths
`f_s` are fitted by collocation of the impedance BC at `N` points on the
wall:

```
[∂G/∂n + ikβ G](x_c, y_s) f = −[∂G/∂n + ikβ G](x_c, x0) a
```

Implementation details that matter (all encode known MFS practice):

- **Source offset**: `y_s = x_c + 4·h·n`, with `h` the mean collocation
  spacing. Offsets of 4–6 spacings are the well-conditioned regime; smaller
  offsets reproduce the near-singular BEM behaviour, larger ones blow up the
  condition number.
- **Regularisation**: the square system is solved via Tikhonov-regularised
  normal equations, `(AᴴA + λI) f = Aᴴ b`, `λ = 10⁻¹⁰·tr(AᴴA)/N` — an LU
  solve, no SVD, fast on a laptop.
- **Honest residual**: after solving, the BC residual is evaluated on an
  **independent** set of wall points (different lattice) and reported for
  every run: `max|∂p/∂n + ikβp| / max|incident|`. Typical values 0.3–3 %.
- **Band limit**: a collocation lattice with spacing `h` resolves
  wavelengths down to ~6h, so the solver caps the band at
  `f_cap = c/(6h)`. Quality presets: preview ≈ 110 Hz, standard ≈ 155 Hz,
  high ≈ 190 Hz.

## 4. Time domain: frequency synthesis (no time-marching)

For transient signals `s(t)` (Ricker, tone burst, click, chirp — all
band-limited, no DC), the source spectrum `S(ω) = FFT[s]` is computed on the
`rfft` grid of the requested time window (`Nt`, `dt`). One Helmholtz problem
is solved per bin carrying significant energy (`|S| ≥ 3·10⁻³ max|S|`,
`f ≤ f_cap`), and the pressure history is recovered by inverse FFT.

This synthesis is **unconditionally stable** — there is no marching-on-in-time
scheme to go unstable, the classic pathology of time-domain BEM.

For the **steady tone** mode a single frequency is solved and the field is
the phasor animation `p(x,t) = Re[P̂(x) e^{iωt}]` — the room's standing-wave
response to a continuously playing speaker.

## 5. Physical calibration

The speaker is calibrated by its free-field level at 1 m. Since
`|G| = 1/(4πr)`, the source is scaled by

```
A = 4π · √2 · p_ref · 10^(SPL₁ₘ/20),      p_ref = 20 µPa
```

so the direct field has **peak** pressure at 1 m equal to the amplitude of a
sine at `SPL₁ₘ` dB SPL. All outputs are in pascals; SPL displays use
`20 log₁₀(p_rms/p_ref)` with `p_rms = |envelope|/√2` (envelope = magnitude of
the analytic signal, computed by one-sided spectrum inversion).

Sanity check (verified live): speaker 85 dB @ 1 m, mic at 3 m in a 6×5×3 m
brick/wood/drywall room → 78.6 dB SPL at the mic. Free-field direct alone
would give 75.5 dB (−9.5 dB for the distance); the reverberant field adds
~3 dB. Textbook behaviour.

## 6. Acoustic metrics (ISO 3382 style)

From the simulated impulse response `h(t)` at the receiver:

- **Schroeder decay**: `S(t) = 10 log₁₀ [ ∫ₜ^∞ h² dτ / ∫₀^∞ h² dτ ]`
- **RT60 (T20)**: −60 / slope of the linear fit of `S(t)` between −5 and
  −25 dB (reported only when the decay range is actually reached — a short
  window in a live room yields "window too short", not a fake number)
- **EDT**: −60 / slope between −0.5 and −10 dB
- **C50 / C80**: `10 log₁₀ (E_{0–50ms} / E_{50ms–∞})` (speech / music clarity)
- **D50**: `E_{0–50ms} / E_total`

Independent statistical estimates from the material data:

- **Sabine**: `T = 0.161 V / A`, `A = Σ Sᵢ αᵢ`
- **Eyring**: `T = 0.161 V / (−S ln(1 − ᾱ))`
- **Schroeder frequency**: `f_s = 2000 √(T/V)` — above it the modal picture
  transitions to statistical acoustics (and this solver's band typically
  sits below or around it, i.e. in the regime where wave simulation is the
  correct tool).

Analytic **room modes** of the rigid box are tabulated for reference:

```
f(n₁,n₂,n₃) = (c/2) √[(n₁/Lx)² + (n₂/Ly)² + (n₃/Lz)²]
```

labelled axial / tangential / oblique by the number of non-zero indices.

## 7. Verification summary

| # | Test | Result |
|---|------|--------|
| 1 | `∇²G + k²G = 0` (finite differences) | ~3·10⁻⁹ |
| 2 | Rigid-wall BC residual on independent points | ~1·10⁻² |
| 3 | Absorbing walls (β=0.5) vs rigid: late-window energy ratio | < 10⁻³ |
| 4 | Resonance at analytic axial mode `c/2Lx` = 42.875 Hz | peak 6× neighbours |
| — | FRF peaks vs analytic mode lines (28.6, 34.3, 44.6 Hz …) | aligned (visual, FRF tab) |
| — | RT60: simulated T20 vs Sabine/Eyring (6×5×3 brick/wood/drywall) | 0.79 s vs 0.94/0.88 s |
| — | SPL at 3 m for 85 dB @ 1 m speaker | 78.6 dB (theory ≈ 78–79) |

Run them: `python core/acoustics.py --selftest`.

## 8. Known limitations & roadmap

- Band limited to ~200 Hz on laptop budgets (dense O(N³) solve per
  frequency). Roadmap: FMM / ℋ-matrix compression for O(N log N), pushing
  the band to 1 kHz-class; hybrid coupling to a geometric solver above the
  Schroeder frequency (the standard commercial architecture).
- Local-reaction impedance model; normal-incidence α→β mapping.
- No air absorption (< 0.1 dB for these bands and distances).
- Empty rooms (box or watertight OBJ); no furniture/diffuser scattering yet.
- OBJ rooms: collocation at triangle centroids — mesh should be reasonably
  uniform; normals must point outward.

## 9. References

- Fairweather, G., Karageorghis, A. (1998). *The method of fundamental
  solutions for elliptic boundary value problems.* Adv. Comput. Math. 9.
- Alves, C.J.S. (2009). *On the choice of source points in the method of
  fundamental solutions.* Eng. Anal. Bound. Elem. 33.
- Antunes, P.R.S. (2018). *Numerical calculation of eigensolutions of 3D
  shapes using the MFS.* (modal MFS accuracy)
- Kuttruff, H. *Room Acoustics*, 6th ed., CRC Press 2016. (impedance BC,
  Sabine/Eyring, Schroeder frequency)
- Vorländer, M. *Auralization*, Springer 2008. (material data, IR metrics,
  auralization)
- ISO 3382-1/-2 — measurement of room acoustic parameters (RT, EDT, C50/80,
  D50 definitions).
- Schroeder, M.R. (1965). *New method of measuring reverberation time.*
  JASA 37. (backward integration)
