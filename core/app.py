"""
app.py — RoomWave Studio: wave-based 3D room acoustics analyzer (local web app).

Zero-dependency server (stdlib http.server + Plotly.js served from the
installed plotly package; no Flask). Physics: acoustics.py (scalar Helmholtz
MFS, per-surface frequency-dependent impedance materials, physically
calibrated speaker source, ISO 3382 metrics).

Run:
    python app.py            # then open http://localhost:8747

HTTP API (one simulation at a time; long solves run in a worker thread so no
request is ever held open for minutes — browsers drop those):

    GET  /            the single-page UI
    GET  /plotly.js   plotly.min.js from the local plotly package
    POST /simulate    start a job; body = JSON of the sidebar fields
                      -> 202 {"started": true} | 429 if one is running
    GET  /progress    {"state": idle|running|done|error, "done", "total", "error"}
    GET  /result      full result JSON of the last finished job (409 if none)

The result payload is documented by its two builders below: simulate()
(transient signals: field movie + IR + FRF + metrics) and _steady_response()
(speaker tone: complex phasor field; the client animates Re[P e^{iwt}]
locally so the loop is seamless and the payload small).

See ../docs/USER_GUIDE.md for the UI walkthrough and ../docs/PHYSICS.md for
the formulation.
"""
from __future__ import annotations
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import acoustics as ac

PORT = 8747
QUALITY = {
    #            n_wall  Nt   dt      ng
    'preview':  (450,    128, 0.0035, (36, 30)),
    'standard': (900,    224, 0.003,  (48, 40)),
    'high':     (1400,   256, 0.0026, (52, 44)),
}
# One job at a time; POST /simulate starts it, GET /progress polls, GET /result
# fetches. (A single long-held HTTP request breaks in browsers on ~1 min solves.)
JOB = {'state': 'idle', 'done': 0, 'total': 0, 'result': None, 'error': None}
_job_lock = threading.Lock()


def _plotly_js():
    import plotly
    return (Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js').read_bytes()


def _f(params, key, default):
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def simulate(params):
    n_wall, Nt, dt, ng = QUALITY.get(params.get('quality', 'preview'),
                                     QUALITY['preview'])
    geometry = None
    L = [_f(params, k, d) for k, d in (('Lx', 6.0), ('Ly', 5.0), ('Lz', 3.0))]
    obj = str(params.get('obj', '')).strip()
    if obj:
        if not os.path.isfile(obj):
            raise ValueError(f"OBJ file not found: '{obj}' — leave the field "
                             "empty to use the box room")
        pts, nrm, spacing = ac.load_obj(obj)
        lo, hi = pts.min(0), pts.max(0)
        L = (hi - lo).tolist()
        geometry = (pts - lo, nrm, spacing)

    clip = lambda p: np.clip(p, 0.08 * np.array(L), 0.92 * np.array(L))
    sources = [clip([_f(params, k, d) for k, d in
                     (('sx', 2.0), ('sy', 2.0), ('sz', 1.5))])]
    amps = [1.0]
    if str(params.get('s2on', '')) in ('1', 'true', 'on'):
        sources.append(clip([_f(params, k, d) for k, d in
                             (('s2x', 4.0), ('s2y', 2.0), ('s2z', 1.5))]))
        amps.append(_f(params, 's2pol', 1.0))
    rx = clip([_f(params, k, d) for k, d in
               (('rx', 4.5), ('ry', 3.5), ('rz', 1.2))])
    mats = (params.get('mat_walls', 'brick'),
            params.get('mat_floor', 'wood floor'),
            params.get('mat_ceiling', 'gypsum drywall'))
    zs = str(params.get('zs', '')).strip()
    show_walls = params.get('view', 'walls') == 'walls' and geometry is None
    scale = params.get('scale', 'pressure')
    spl_1m = min(max(_f(params, 'spl1m', 85.0), 40.0), 120.0)
    signal = params.get('signal', 'ricker')

    def _prog(done, total):
        JOB['done'], JOB['total'] = done, total

    common = dict(L=tuple(L), x0=np.array(sources), src_amps=amps,
                  f0=_f(params, 'f0', 80.0), materials=mats, spl_1m=spl_1m,
                  receivers=(tuple(rx),), n_wall=n_wall, ng=ng,
                  zslice=float(zs) if zs else None, geometry=geometry,
                  walls=show_walls, progress=_prog, verbose=True)
    if signal == 'tone':
        return _steady_response(ac.run_steady(**common), rx, scale)

    res = ac.run(signal=signal, Nt=Nt, dt=dt, spl=(scale == 'spl'), **common)

    t = res['t']
    stride = max(1, int(np.ceil(len(t) / 95)))       # <= ~95 animation frames
    idx = list(range(0, len(t), stride))
    rnd = lambda a: np.round(np.asarray(a, float), 4).tolist()

    db_hi = db_lo = 0.0
    if scale == 'spl':
        fields = [res['env']] + (res['env_walls'] or [])
        # absolute SPL of the instantaneous envelope: 20 log10(env/sqrt(2)/p_ref)
        ref = np.sqrt(2.0) * ac.P_REF
        db_hi = float(np.ceil(20 * np.log10(max(f.max() for f in fields) / ref)))
        db_lo = db_hi - 50.0
        todb = lambda a: np.round(np.clip(20 * np.log10(
            np.maximum(a, 1e-12) / ref), db_lo, db_hi), 2).tolist()
        frames = [todb(res['env'][i]) for i in idx]
        wall_frames = [[todb(e[i]) for i in idx] for e in (res['env_walls'] or [])]
        vmax = 0.0
    else:
        fields = [res['pt']] + [w['pt'] for w in (res['walls'] or [])]
        vmax = float(max(np.percentile(np.abs(f), 99.5) for f in fields))
        cl = lambda a: np.round(np.clip(a, -vmax, vmax), 5).tolist()
        frames = [cl(res['pt'][i]) for i in idx]
        wall_frames = [[cl(w['pt'][i]) for i in idx] for w in (res['walls'] or [])]

    m = res['metrics']
    return {
        'x': rnd(res['X'][0]), 'y': rnd(res['Y'][:, 0]), 'z': float(res['z']),
        'L': L, 'sources': rnd(res['x0']), 'rxp': rnd(rx),
        'vmax': vmax, 'scale': scale, 'db_hi': db_hi, 'db_lo': db_lo,
        't_ms': rnd(np.asarray(t)[idx] * 1000),
        'frames': frames,
        'walls': [{'name': w['name'], 'X': rnd(w['X']), 'Y': rnd(w['Y']),
                   'Z': rnd(w['Z']), 'frames': wf}
                  for w, wf in zip(res['walls'] or [], wall_frames)],
        'ir': {'t_ms': rnd(t * 1000),
               'h': np.round(res['rx_p'][:, 0], 6).tolist(),   # pascals
               'sig': rnd(res['sig']), 'dt': res['dt'],
               'schroeder': m.get('schroeder_db', [])},
        'frf': {'f': rnd(res['frf_f']),
                'HdB': rnd(20 * np.log10(np.maximum(res['frf_H'], 1e-12)
                                         / max(max(res['frf_H']), 1e-12)))},
        'metrics': {k: m.get(k) for k in
                    ('rt60_t20', 'rt60_sabine', 'rt60_eyring', 'edt', 'c50',
                     'c80', 'd50', 'mean_alpha', 'volume', 'area',
                     'f_schroeder', 'bands', 'spl_peak')},
        'modes': res['modes'][:25],
        'meta': {'bc_residual': res['resmax'], 'nsolved': res['nsolved'],
                 'f_cap': res['f_cap'], 'f0': res['f0'], 'signal': res['signal'],
                 'materials': [res['face_mats'][0], res['face_mats'][4],
                               res['face_mats'][5]]},
    }


def _steady_response(res, rx, scale):
    """Response for the continuous speaker tone: send the complex phasor
    (Re, Im) — the client animates p(t) = Re cos(wt) - Im sin(wt) locally,
    so the animation loops seamlessly with a tiny payload."""
    rnd = lambda a: np.round(np.asarray(a, float), 5).tolist()
    P = res['P']
    fields = [P] + [w['P'] for w in res['walls']]
    # 94th percentile: the 1/r near-field around the speaker must not wash out
    # the room's standing-wave contrast
    vmax = float(max(np.percentile(np.abs(f), 94) for f in fields))
    m = dict(res['metrics'])
    return {
        'steady': True, 'f0': res['f0'], 'period_ms': 1000.0 / res['f0'],
        'x': rnd(res['X'][0]), 'y': rnd(res['Y'][:, 0]), 'z': float(res['z']),
        'L': list(res['L']), 'sources': rnd(res['x0']), 'rxp': rnd(rx),
        'vmax': vmax, 'scale': scale,
        're': rnd(P.real), 'im': rnd(P.imag),
        'walls': [{'name': w['name'], 'X': rnd(w['X']), 'Y': rnd(w['Y']),
                   'Z': rnd(w['Z']), 're': rnd(w['P'].real),
                   'im': rnd(w['P'].imag)} for w in res['walls']],
        'ir': {'rx_re': float(res['P_rx'][0].real),
               'rx_im': float(res['P_rx'][0].imag)},
        'metrics': m, 'modes': res['modes'][:25],
        'meta': {'bc_residual': res['resmax'], 'nsolved': 1,
                 'f_cap': res['f_cap'], 'f0': res['f0'], 'signal': 'tone',
                 'materials': [res['face_mats'][0], res['face_mats'][4],
                               res['face_mats'][5]]},
    }


MAT_OPTIONS = ''.join(f'<option value="{m}">{m}</option>' for m in ac.MATERIALS)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>RoomWave Studio</title>
<script src="/plotly.js"></script>
<style>
 :root{--bg:#0d0f14;--panel:#161a22;--card:#1d2330;--acc:#3b82f6;--acc2:#22d3ee;
       --tx:#d6dae2;--mut:#8b93a3}
 *{box-sizing:border-box} body{margin:0;font:13px/1.45 system-ui;background:var(--bg);
   color:var(--tx);display:flex;height:100vh;overflow:hidden}
 #panel{width:276px;padding:14px;background:var(--panel);overflow-y:auto;flex-shrink:0;
   border-right:1px solid #262c38}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 h1{font-size:16px;margin:0 0 2px;letter-spacing:.3px}
 .sub{color:var(--mut);font-size:11px;margin-bottom:12px}
 h2{font-size:11px;color:var(--acc2);margin:15px 0 6px;text-transform:uppercase;
    letter-spacing:1px}
 label{display:block;margin:6px 0 2px;color:var(--mut);font-size:11.5px}
 input,select{width:100%;background:#10141c;color:var(--tx);border:1px solid #2b3242;
   border-radius:5px;padding:5px 7px;font-size:12.5px}
 input[type=checkbox]{width:auto;margin-right:6px}
 input[type=range]{padding:0}
 input:focus,select:focus{outline:none;border-color:var(--acc)}
 .row{display:flex;gap:6px}.row>div{flex:1}
 .chk{display:flex;align-items:center;margin-top:8px;color:var(--mut);font-size:12px}
 button.primary{width:100%;margin-top:13px;padding:10px;background:linear-gradient(90deg,var(--acc),#6366f1);
   color:#fff;border:0;border-radius:7px;font-size:14px;font-weight:600;cursor:pointer}
 button.primary:disabled{background:#3a4152;color:#9aa}
 #pbarw{margin-top:10px;height:7px;background:#242b3a;border-radius:4px;overflow:hidden;display:none}
 #pbar{height:100%;width:0%;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .4s}
 #status{margin-top:8px;font-size:11.5px;color:#9ec89e;white-space:pre-wrap;min-height:3em}
 #tabs{display:flex;gap:2px;background:var(--panel);padding:8px 10px 0;border-bottom:1px solid #262c38}
 .tab{padding:8px 16px;border-radius:8px 8px 0 0;cursor:pointer;color:var(--mut);
   background:transparent;border:0;font-size:13px}
 .tab.active{background:var(--bg);color:var(--tx);font-weight:600}
 .view{flex:1;display:none;min-height:0;position:relative}.view.active{display:block}
 .plotdiv{width:100%;height:100%}
 #ctrl3d{position:absolute;top:8px;left:12px;z-index:5;display:none;gap:10px;align-items:center;
   background:#161a22cc;padding:5px 10px;border-radius:8px;font-size:12px;color:var(--mut)}
 #ctrl3d select{width:auto}
 #irtools{position:absolute;top:10px;right:16px;z-index:5;display:flex;gap:8px;align-items:center}
 #frftools{position:absolute;top:10px;right:16px;z-index:5;display:flex;gap:12px;align-items:center;
   background:#161a22cc;padding:6px 12px;border-radius:8px;font-size:12px;color:var(--mut)}
 #metricsview{overflow-y:auto;padding:18px;height:100%}
 .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px}
 .card{background:var(--card);border-radius:10px;padding:12px 14px;border:1px solid #262c38}
 .card .v{font-size:21px;font-weight:700;color:#fff;margin-top:2px}
 .card .u{font-size:11px;color:var(--mut)}
 .card .n{font-size:11px;color:var(--acc2);text-transform:uppercase;letter-spacing:.5px}
 .card .d{font-size:11px;margin-top:2px}
 .dpos{color:#f87171}.dneg{color:#4ade80}
 table{border-collapse:collapse;margin-top:8px;font-size:12px}
 td,th{padding:4px 12px;border-bottom:1px solid #262c38;text-align:left}
 th{color:var(--mut);font-weight:600}
 .audiobtn{padding:7px 12px;background:var(--card);color:var(--tx);border:1px solid #2b3242;
   border-radius:7px;cursor:pointer;font-size:12px}
 .audiobtn:hover{border-color:var(--acc2)}
 #empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   color:var(--mut);font-size:15px;flex-direction:column;gap:8px}
 #irwrap{display:flex;flex-direction:column;height:100%}
 #plotir{flex:3;min-height:0}#plotspec{flex:2;min-height:0}
</style></head><body>
<div id="panel">
 <h1>&#127911; RoomWave Studio</h1>
 <div class="sub">wave-based room acoustics &middot; Helmholtz MFS</div>
 <h2>Preset</h2>
 <select id="preset" onchange="applyPreset()">
  <option value="">— custom —</option><option value="living">Living room</option>
  <option value="studio">Recording studio</option><option value="bathroom">Bathroom</option>
  <option value="hall">Small hall</option><option value="stereo">Stereo speaker pair</option></select>
 <h2>Room (m)</h2>
 <div class="row"><div><label>Lx</label><input id="Lx" value="6"></div>
  <div><label>Ly</label><input id="Ly" value="5"></div>
  <div><label>Lz</label><input id="Lz" value="3"></div></div>
 <label>OBJ mesh file (optional, overrides box)</label>
 <input id="obj" placeholder="leave empty for box room" autocomplete="off">
 <h2>Surface materials</h2>
 <label>Walls</label><select id="mat_walls">__MATS__</select>
 <label>Floor</label><select id="mat_floor">__MATS__</select>
 <label>Ceiling</label><select id="mat_ceiling">__MATS__</select>
 <h2>Source</h2>
 <div class="row"><div><label>x</label><input id="sx" value="2"></div>
  <div><label>y</label><input id="sy" value="2"></div>
  <div><label>z</label><input id="sz" value="1.5"></div></div>
 <div class="row"><div><label>Signal</label>
  <select id="signal"><option value="tone">Speaker tone (steady)</option>
   <option value="ricker">Ricker pulse</option>
   <option value="toneburst">Tone burst</option><option value="click">Click</option>
   <option value="chirp">Chirp sweep</option></select></div>
  <div><label>f0 (Hz)</label><input id="f0" value="80"></div></div>
 <label>Speaker level @ 1 m (dB SPL)</label>
 <input type="range" id="spl1m" min="50" max="110" step="1" value="85"
  oninput="spl1mV.textContent=this.value"><div style="text-align:center;color:var(--mut)">
  <span id="spl1mV">85</span> dB</div>
 <div class="chk"><input type="checkbox" id="s2on" onchange="s2f.style.display=this.checked?'block':'none'">
  <label style="margin:0">Second source (stereo pair)</label></div>
 <div id="s2f" style="display:none">
  <div class="row"><div><label>x</label><input id="s2x" value="4"></div>
   <div><label>y</label><input id="s2y" value="2"></div>
   <div><label>z</label><input id="s2z" value="1.5"></div></div>
  <label>Polarity</label><select id="s2pol">
   <option value="1">in phase (+)</option><option value="-1">anti-phase (&minus;)</option></select>
 </div>
 <h2>Receiver mic</h2>
 <div class="row"><div><label>x</label><input id="rx" value="4.5"></div>
  <div><label>y</label><input id="ry" value="3.5"></div>
  <div><label>z</label><input id="rz" value="1.2"></div></div>
 <h2>Visualisation</h2>
 <div class="row"><div><label>View</label>
  <select id="view"><option value="walls">Walls + slice</option>
   <option value="slice">Slice only</option></select></div>
  <div><label>Field scale</label>
  <select id="scale"><option value="pressure">Pressure</option>
   <option value="spl">SPL (dB)</option></select></div></div>
 <div class="row"><div><label>Colormap</label>
  <select id="cmap"><option value="RdBu">RdBu (classic)</option>
   <option value="Portland">Portland</option><option value="Viridis">Viridis</option>
   <option value="Inferno">Inferno</option></select></div>
  <div><label>Slice height (m)</label><input id="zs" placeholder="src height"></div></div>
 <label>Surface relief: <span id="reliefV">0.25</span> &times; Lz</label>
 <input type="range" id="relief" min="0" max="0.5" step="0.05" value="0.25"
  oninput="reliefV.textContent=this.value;if(D)render3D()">
 <h2>Compute</h2>
 <label>Quality</label>
 <select id="quality"><option value="preview">Preview (~10 s)</option>
  <option value="standard">Standard (~2 min)</option>
  <option value="high">High (~4 min)</option></select>
 <button class="primary" id="run" onclick="runSim()">&#9654;&nbsp; Simulate</button>
 <div id="pbarw"><div id="pbar"></div></div>
 <div id="status">Ready.</div>
</div>
<div id="main">
 <div id="tabs">
  <button class="tab active" onclick="showTab(0,this)">3D Wave Field</button>
  <button class="tab" onclick="showTab(1,this)">Impulse Response</button>
  <button class="tab" onclick="showTab(2,this)">Frequency Response</button>
  <button class="tab" onclick="showTab(3,this)">Acoustic Metrics</button>
 </div>
 <div class="view active" id="v0">
  <div id="ctrl3d">
   <span>speed</span><select id="speed"><option value="120">0.5&times;</option>
    <option value="60" selected>1&times;</option><option value="30">2&times;</option></select>
  </div>
  <div id="plot3d" class="plotdiv"></div>
  <div id="empty"><div style="font-size:34px">&#127925;</div>
   <div>Configure the room and press <b>Simulate</b></div></div>
 </div>
 <div class="view" id="v1"><div id="irwrap">
  <div id="plotir"></div><div id="plotspec"></div></div>
  <div id="irtools">
   <button class="audiobtn" onclick="playIR()">&#128266; IR</button>
   <button class="audiobtn" onclick="playKick()">&#129345; Kick</button>
   <button class="audiobtn" onclick="playClap()">&#128079; Clap</button>
   <button class="audiobtn" onclick="stopAudio()">&#9632;</button>
   <span style="color:var(--mut);font-size:11px">wet</span>
   <input type="range" id="wet" min="0" max="1" step="0.1" value="0.9" style="width:70px">
   <button class="audiobtn" onclick="downloadWAV()">&#11015; WAV</button>
  </div></div>
 <div class="view" id="v2"><div id="plotfrf" class="plotdiv"></div>
  <div id="frftools">
   <label style="margin:0"><input type="checkbox" id="fModes" checked onchange="renderFRF()">modes</label>
   <label style="margin:0"><input type="checkbox" id="fSmooth" onchange="renderFRF()">&#8531;-oct smooth</label>
   <label style="margin:0"><input type="checkbox" id="fPrev" checked onchange="renderFRF()">previous run</label>
  </div></div>
 <div class="view" id="v3"><div id="metricsview">
  <div style="color:var(--mut)">Run a simulation to see metrics.</div></div></div>
</div>
<script>
let D=null,prevD=null;
const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
window.addEventListener('DOMContentLoaded',()=>{   // sensible material defaults
  $('mat_walls').value='brick';$('mat_floor').value='wood floor';
  $('mat_ceiling').value='gypsum drywall';
  $('obj').value='';   // Firefox restores form values by POSITION on reload —
                       // stale text from an older layout must not become a path
});
const PRESETS={
 living:{Lx:6,Ly:5,Lz:3,mat_walls:'brick',mat_floor:'wood floor',mat_ceiling:'gypsum drywall',
         sx:1,sy:1,sz:1.2,rx:4.5,ry:3.5,rz:1.2,f0:80,signal:'ricker'},
 studio:{Lx:5,Ly:4,Lz:2.8,mat_walls:'acoustic panel',mat_floor:'carpet',mat_ceiling:'acoustic panel',
         sx:2.5,sy:1,sz:1.4,rx:2.5,ry:3,rz:1.4,f0:90,signal:'click'},
 bathroom:{Lx:2.5,Ly:2,Lz:2.4,mat_walls:'ceramic tile',mat_floor:'ceramic tile',
         mat_ceiling:'gypsum drywall',sx:.6,sy:.5,sz:1.2,rx:1.9,ry:1.5,rz:1.5,f0:110,signal:'toneburst'},
 hall:{Lx:10,Ly:7,Lz:4.5,mat_walls:'concrete',mat_floor:'wood floor',mat_ceiling:'concrete',
         sx:2,sy:3.5,sz:1.5,rx:8,ry:3.5,rz:1.5,f0:60,signal:'ricker'},
 stereo:{Lx:6,Ly:5,Lz:3,mat_walls:'gypsum drywall',mat_floor:'carpet',mat_ceiling:'gypsum drywall',
         sx:1.8,sy:1,sz:1.2,rx:3,ry:3.8,rz:1.2,f0:80,signal:'toneburst',_s2:{x:4.2,y:1,z:1.2}}};
function applyPreset(){const p=PRESETS[$('preset').value];if(!p)return;
  for(const k in p){if(k==='_s2')continue;const el=$(k);if(el)el.value=p[k];}
  if(p._s2){$('s2on').checked=true;$('s2f').style.display='block';
    $('s2x').value=p._s2.x;$('s2y').value=p._s2.y;$('s2z').value=p._s2.z;}
  else{$('s2on').checked=false;$('s2f').style.display='none';}}
function showTab(i,btn){document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  btn.classList.add('active');$('v'+i).classList.add('active');
  if(!D)return;
  if(i===0)Plotly.Plots.resize($('plot3d'));
  if(i===1){Plotly.Plots.resize($('plotir'));Plotly.Plots.resize($('plotspec'));}
  if(i===2)Plotly.Plots.resize($('plotfrf'));}

async function runSim(){
  const btn=$('run'),st=$('status');
  btn.disabled=true;st.textContent='Starting solver\\u2026';
  $('pbarw').style.display='block';$('pbar').style.width='0%';
  const ids=['Lx','Ly','Lz','sx','sy','sz','s2x','s2y','s2z','s2pol','rx','ry','rz',
             'f0','signal','quality','obj','view','scale','zs','spl1m',
             'mat_walls','mat_floor','mat_ceiling'];
  const p={};ids.forEach(i=>p[i]=$(i).value);
  p.s2on=$('s2on').checked?'1':'0';
  const t0=performance.now();
  try{
    const r=await fetch('/simulate',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    if(!r.ok)throw new Error(await r.text());
    while(true){
      await sleep(600);
      const q=await(await fetch('/progress')).json();
      if(q.state==='error')throw new Error(q.error);
      if(q.state==='done')break;
      if(q.total>0){$('pbar').style.width=(100*q.done/q.total).toFixed(0)+'%';
        st.textContent=`Solving Helmholtz: frequency ${q.done} / ${q.total}\\u2026`;}
    }
    $('pbar').style.width='100%';st.textContent='Downloading result\\u2026';
    const d=await(await fetch('/result')).json();
    prevD=D;D=d;
    st.textContent=`Done in ${((performance.now()-t0)/1000).toFixed(1)} s\\n`+
      `${D.meta.nsolved} frequencies, band \\u2264 ${D.meta.f_cap.toFixed(0)} Hz\\n`+
      `BC residual ${D.meta.bc_residual.toExponential(1)}`;
    $('empty').style.display='none';$('ctrl3d').style.display='flex';
    if(D.steady){renderSteady3D();renderSteadyIR();renderFRF();renderMetrics();}
    else{render3D();renderIR();renderSpec();renderFRF();renderMetrics();}
  }catch(e){st.textContent='Error: '+e.message;}
  $('pbarw').style.display='none';btn.disabled=false;
}

function wire(L){
  const c=[[0,0,0],[L[0],0,0],[0,L[1],0],[L[0],L[1],0],[0,0,L[2]],[L[0],0,L[2]],[0,L[1],L[2]],L];
  const ex=[],ey=[],ez=[];
  for(let i=0;i<8;i++)for(let j=i+1;j<8;j++){
    let dif=0;for(let a=0;a<3;a++)if(c[i][a]!==c[j][a])dif++;
    if(dif===1){ex.push(c[i][0],c[j][0],null);ey.push(c[i][1],c[j][1],null);
                ez.push(c[i][2],c[j][2],null);}}
  return{type:'scatter3d',x:ex,y:ey,z:ez,mode:'lines',line:{color:'#4a5266',width:3},
         hoverinfo:'skip',showlegend:false};
}
function render3D(){
  const spl=D.scale==='spl',v=spl?1:D.vmax,relief=parseFloat($('relief').value)*D.L[2];
  const cmap=$('cmap').value;
  const amp=p=>spl?(p-D.db_lo)/(D.db_hi-D.db_lo):p/v;
  const zOf=f=>f.map(r=>r.map(p=>D.z+amp(p)*relief));
  const cargs=spl?{cmin:D.db_lo,cmax:D.db_hi,colorscale:cmap==='RdBu'?'Inferno':cmap}
                 :{cmin:-v,cmax:v,colorscale:cmap,reversescale:cmap==='RdBu'};
  const surf=i=>Object.assign({type:'surface',x:D.x,y:D.y,z:zOf(D.frames[i]),
    surfacecolor:D.frames[i],colorbar:{title:spl?'dB SPL':'Pa',len:0.55}},cargs);
  const wsurf=(w,i)=>Object.assign({type:'surface',x:w.X,y:w.Y,z:w.Z,
    surfacecolor:w.frames[i],showscale:false,name:w.name},cargs);
  const walls=D.walls||[];
  const animated=i=>[surf(i)].concat(walls.map(w=>wsurf(w,i)));
  const spk={type:'scatter3d',x:D.sources.map(s=>s[0]),y:D.sources.map(s=>s[1]),
    z:D.sources.map(s=>s[2]),mode:'markers+text',
    marker:{size:7,color:'#facc15',symbol:'diamond'},
    text:D.sources.map((s,i)=>'S'+(i+1)),textposition:'top center',
    textfont:{color:'#facc15'},showlegend:false};
  const mic={type:'scatter3d',x:[D.rxp[0]],y:[D.rxp[1]],z:[D.rxp[2]],mode:'markers+text',
    marker:{size:6,color:'#22d3ee'},text:['R'],textposition:'top center',
    textfont:{color:'#22d3ee'},showlegend:false};
  const frames=D.frames.map((f,i)=>({name:''+i,data:animated(i)}));
  const steps=D.t_ms.map((t,i)=>({method:'animate',label:t.toFixed(0),
    args:[[''+i],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}}]}));
  Plotly.newPlot('plot3d',animated(0).concat([wire(D.L),spk,mic]),{
    template:'plotly_dark',paper_bgcolor:'#0d0f14',
    scene:{aspectmode:'data',xaxis:{title:'x (m)'},yaxis:{title:'y (m)'},
           zaxis:{title:'z (m)'},camera:{eye:{x:1.45,y:1.45,z:0.85}},bgcolor:'#0d0f14'},
    margin:{l:0,r:0,t:8,b:0},
    updatemenus:[{type:'buttons',x:0.04,y:0.04,bgcolor:'#1d2330',
      font:{color:'#d6dae2'},buttons:[
      {label:'\\u25B6 Play',method:'animate',args:[null,{frame:{duration:+$('speed').value,
        redraw:true},fromcurrent:true,transition:{duration:0}}]},
      {label:'\\u275A\\u275A',method:'animate',
       args:[[null],{mode:'immediate',frame:{duration:0,redraw:false}}]}]}],
    sliders:[{x:0.17,len:0.79,y:0.03,currentvalue:{prefix:'t = ',suffix:' ms',
      font:{size:12}},steps:steps}]
  },{responsive:true}).then(gd=>Plotly.addFrames(gd,frames));
}
function renderSteady3D(){
  // continuous speaker: p(t) = Re cos(wt) - Im sin(wt); loops seamlessly.
  const spl=D.scale==='spl',relief=parseFloat($('relief').value)*D.L[2];
  const cmap=$('cmap').value,v=D.vmax;
  const pref=Math.sqrt(2)*2e-5;
  const nph=24,ncyc=3;
  const phase=(re,im,c,s)=>re.map((row,i)=>row.map((x,j)=>x*c-im[i][j]*s));
  const toSPL=(re,im)=>re.map((row,i)=>row.map((x,j)=>
    20*Math.log10(Math.max(Math.hypot(x,im[i][j])/pref,1e-3))));
  if(spl){    // steady SPL map is time-invariant: show the standing-wave map
    const Zs=toSPL(D.re,D.im);let hi=-1e9;Zs.forEach(r=>r.forEach(x=>hi=Math.max(hi,x)));
    hi=Math.ceil(hi);const lo=hi-40;
    const cargs={cmin:lo,cmax:hi,colorscale:cmap==='RdBu'?'Inferno':cmap};
    const mk=(x,y,z,sc,scale)=>Object.assign({type:'surface',x:x,y:y,z:z,
      surfacecolor:sc,showscale:scale,colorbar:scale?{title:'dB SPL',len:0.55}:undefined},cargs);
    const tr=[mk(D.x,D.y,D.re.map(r=>r.map(_=>D.z)),Zs,true)];
    (D.walls||[]).forEach(w=>tr.push(mk(w.X,w.Y,w.Z,toSPL(w.re,w.im),false)));
    Plotly.newPlot('plot3d',tr.concat([wire(D.L),steadyMarkers().spk,steadyMarkers().mic]),
      steadyLayout(`Standing-wave SPL map \\u00B7 ${D.f0.toFixed(0)} Hz tone (time-invariant)`),
      {responsive:true});
    return;
  }
  const cargs={cmin:-v,cmax:v,colorscale:cmap,reversescale:cmap==='RdBu'};
  const frameField=k=>{const ph=2*Math.PI*k/nph;
    return phase(D.re,D.im,Math.cos(ph),Math.sin(ph));};
  const surfAt=(f,first)=>Object.assign({type:'surface',x:D.x,y:D.y,
    z:f.map(r=>r.map(p=>D.z+p/v*relief)),surfacecolor:f,
    showscale:first,colorbar:first?{title:'Pa',len:0.55}:undefined},cargs);
  const wallAt=(w,k)=>{const ph=2*Math.PI*k/nph;
    const f=phase(w.re,w.im,Math.cos(ph),Math.sin(ph));
    return Object.assign({type:'surface',x:w.X,y:w.Y,z:w.Z,surfacecolor:f,
      showscale:false},cargs);};
  const animated=k=>[surfAt(frameField(k),true)].concat((D.walls||[]).map(w=>wallAt(w,k)));
  const N=nph*ncyc;
  const frames=[];for(let k=0;k<N;k++)frames.push({name:''+k,data:animated(k)});
  const steps=[];for(let k=0;k<N;k++)steps.push({method:'animate',
    label:(k%nph*D.period_ms/nph).toFixed(1),
    args:[[''+k],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}}]});
  const m=steadyMarkers();
  Plotly.newPlot('plot3d',animated(0).concat([wire(D.L),m.spk,m.mic]),
    Object.assign(steadyLayout(''),{
    updatemenus:[{type:'buttons',x:0.04,y:0.04,bgcolor:'#1d2330',font:{color:'#d6dae2'},
      buttons:[{label:'\\u25B6 Play',method:'animate',
        args:[null,{frame:{duration:Math.max(+$('speed').value/2,20),redraw:true},
        fromcurrent:true,transition:{duration:0}}]},
      {label:'\\u275A\\u275A',method:'animate',
       args:[[null],{mode:'immediate',frame:{duration:0,redraw:false}}]}]}],
    sliders:[{x:0.17,len:0.79,y:0.03,currentvalue:{prefix:'phase t = ',suffix:' ms',
      font:{size:12}},steps:steps}]}),
    {responsive:true}).then(gd=>Plotly.addFrames(gd,frames));
}
function steadyMarkers(){
  return{spk:{type:'scatter3d',x:D.sources.map(s=>s[0]),y:D.sources.map(s=>s[1]),
    z:D.sources.map(s=>s[2]),mode:'markers+text',
    marker:{size:7,color:'#facc15',symbol:'diamond'},
    text:D.sources.map((s,i)=>'S'+(i+1)),textposition:'top center',
    textfont:{color:'#facc15'},showlegend:false},
  mic:{type:'scatter3d',x:[D.rxp[0]],y:[D.rxp[1]],z:[D.rxp[2]],mode:'markers+text',
    marker:{size:6,color:'#22d3ee'},text:['R'],textposition:'top center',
    textfont:{color:'#22d3ee'},showlegend:false}};
}
function steadyLayout(title){
  return{template:'plotly_dark',paper_bgcolor:'#0d0f14',
    title:title?{text:title,font:{size:13}}:undefined,
    scene:{aspectmode:'data',xaxis:{title:'x (m)'},yaxis:{title:'y (m)'},
      zaxis:{title:'z (m)'},camera:{eye:{x:1.45,y:1.45,z:0.85}},bgcolor:'#0d0f14'},
    margin:{l:0,r:0,t:title?30:8,b:0}};
}
function renderSteadyIR(){
  const A=Math.hypot(D.ir.rx_re,D.ir.rx_im),ph0=Math.atan2(D.ir.rx_im,D.ir.rx_re);
  const T=D.period_ms,n=200,ts=[],ps=[];
  for(let i=0;i<n;i++){const t=3*T*i/n;ts.push(t);
    ps.push(A*Math.cos(2*Math.PI*t/T+ph0));}
  Plotly.newPlot('plotir',[{x:ts,y:ps,mode:'lines',name:'p at mic',
    line:{color:'#22d3ee',width:2}}],
   {template:'plotly_dark',paper_bgcolor:'#0d0f14',plot_bgcolor:'#0d0f14',
    title:{text:`Steady tone at receiver \\u00B7 amplitude ${A.toFixed(3)} Pa \\u00B7 `+
      `${D.metrics.spl_mic.toFixed(1)} dB SPL`,font:{size:13}},
    xaxis:{title:'time (ms)'},yaxis:{title:'p (Pa)'},margin:{t:40,b:36}},{responsive:true});
  Plotly.newPlot('plotspec',[],{template:'plotly_dark',paper_bgcolor:'#0d0f14',
    plot_bgcolor:'#0d0f14',xaxis:{visible:false},yaxis:{visible:false},
    annotations:[{text:'Spectrogram & auralization need a broadband signal '+
      '(Ricker / click / chirp)',showarrow:false,font:{color:'#8b93a3',size:13}}]},
    {responsive:true});
}
function renderIR(){
  const ir=D.ir;
  const tr=[{x:ir.t_ms,y:ir.h,name:'pressure at mic',line:{color:'#22d3ee',width:1.5}},
            {x:ir.t_ms,y:ir.sig,name:'source signal',yaxis:'y3',
             line:{color:'#facc15',width:1},opacity:0.6}];
  if(ir.schroeder&&ir.schroeder.length)
    tr.push({x:ir.t_ms,y:ir.schroeder,name:'Schroeder decay',yaxis:'y2',
             line:{color:'#f472b6',width:2}});
  Plotly.newPlot('plotir',tr,{template:'plotly_dark',paper_bgcolor:'#0d0f14',
    plot_bgcolor:'#0d0f14',title:{text:'Room response at receiver',font:{size:13}},
    xaxis:{title:'time (ms)'},yaxis:{title:'p (Pa)'},
    yaxis2:{title:'decay (dB)',overlaying:'y',side:'right',range:[-60,3],showgrid:false},
    yaxis3:{overlaying:'y',visible:false},
    legend:{x:0.68,y:1.08,orientation:'h'},margin:{t:40,b:36}},{responsive:true});
}
function renderSpec(){                       // client-side STFT waterfall
  const h=D.ir.h,dt=D.ir.dt,fs=1/dt,W=48,hop=6,nb=W/2;
  const win=i=>0.5-0.5*Math.cos(2*Math.PI*i/(W-1));
  const nf=Math.max(1,Math.floor((h.length-W)/hop)+1);
  const T=[],F=[],Z=[];
  for(let fr=0;fr<nf;fr++)T.push((fr*hop+W/2)*dt*1000);
  for(let b=0;b<nb;b++)F.push(b*fs/W);
  let mx=-1e9;
  for(let b=0;b<nb;b++){const row=[];
    for(let fr=0;fr<nf;fr++){let re=0,im=0;
      for(let i=0;i<W;i++){const v=(h[fr*hop+i]||0)*win(i),ph=-2*Math.PI*b*i/W;
        re+=v*Math.cos(ph);im+=v*Math.sin(ph);}
      const db=20*Math.log10(Math.hypot(re,im)+1e-9);
      mx=Math.max(mx,db);row.push(db);}
    Z.push(row);}
  const Zn=Z.map(r=>r.map(v=>Math.max(v-mx,-45)));
  Plotly.newPlot('plotspec',[{type:'heatmap',x:T,y:F,z:Zn,colorscale:'Inferno',
    zmin:-45,zmax:0,colorbar:{title:'dB',len:0.9}}],
   {template:'plotly_dark',paper_bgcolor:'#0d0f14',plot_bgcolor:'#0d0f14',
    title:{text:'Spectrogram (decay per frequency)',font:{size:13}},
    xaxis:{title:'time (ms)'},yaxis:{title:'f (Hz)',range:[0,Math.min(fs/2,D.meta.f_cap*1.15)]},
    margin:{t:36,b:36}},{responsive:true});
}
function smooth3rd(f,HdB){                    // 1/3-octave smoothing
  const Hl=HdB.map(v=>Math.pow(10,v/20));
  return f.map(fc=>{const lo=fc/Math.pow(2,1/6),hi=fc*Math.pow(2,1/6);
    let s=0,n=0;for(let j=0;j<f.length;j++)if(f[j]>=lo&&f[j]<=hi){s+=Hl[j];n++;}
    return 20*Math.log10(s/Math.max(n,1));});
}
function renderFRF(){
  if(!D)return;
  if(D.steady){
    Plotly.newPlot('plotfrf',[],{template:'plotly_dark',paper_bgcolor:'#0d0f14',
      plot_bgcolor:'#0d0f14',xaxis:{visible:false},yaxis:{visible:false},
      annotations:[{text:'The transfer function needs a broadband signal \\u2014 '+
        'switch Signal to Ricker / click / chirp',showarrow:false,
        font:{color:'#8b93a3',size:13}}]},{responsive:true});
    return;}
  const tr=[];
  const y=$('fSmooth').checked?smooth3rd(D.frf.f,D.frf.HdB):D.frf.HdB;
  if($('fPrev').checked&&prevD)
    tr.push({x:prevD.frf.f,y:$('fSmooth').checked?smooth3rd(prevD.frf.f,prevD.frf.HdB)
      :prevD.frf.HdB,mode:'lines',name:'previous run',
      line:{color:'#8b93a3',width:1.5,dash:'dash'}});
  tr.push({x:D.frf.f,y:y,mode:'lines',name:'|H(f)| current',
    line:{color:'#22d3ee',width:2},fill:'tozeroy',fillcolor:'rgba(34,211,238,0.07)'});
  const shapes=$('fModes').checked?(D.modes||[]).map(m=>({type:'line',x0:m.f,x1:m.f,
    y0:0,y1:1,yref:'paper',line:{color:m.type==='axial'?'#f87171':
    (m.type==='tangential'?'#fbbf24':'#4a5266'),width:1,dash:'dot'},opacity:0.5})):[];
  Plotly.newPlot('plotfrf',tr,
   {template:'plotly_dark',paper_bgcolor:'#0d0f14',plot_bgcolor:'#0d0f14',
    title:{text:'Transfer function source \\u2192 mic (red=axial, amber=tangential, '+
      'grey=oblique modes)',font:{size:13}},
    xaxis:{title:'frequency (Hz)'},yaxis:{title:'|H| (dB re max)',range:[-45,3]},
    shapes:shapes,legend:{x:0.72,y:1.06},margin:{t:44}},{responsive:true});
}
function card(n,v,u,delta){
  let d='';
  if(delta!=null&&isFinite(delta)&&Math.abs(delta)>1e-4){
    const s=delta>0?'+':'\\u2212';
    d=`<div class="d ${delta>0?'dpos':'dneg'}">${s}${Math.abs(delta).toFixed(2)} vs previous</div>`;}
  return `<div class="card"><div class="n">${n}</div><div class="v">${v}</div>`+
         `<div class="u">${u}</div>${d}</div>`;
}
function renderMetrics(){
  const m=D.metrics,pm=prevD?prevD.metrics:{};
  const f=(x,d)=>x==null?'\\u2014':x.toFixed(d);
  const dd=(k)=>(m[k]!=null&&pm&&pm[k]!=null)?m[k]-pm[k]:null;
  let h='<h2 style="margin-top:0">Reverberation &amp; clarity (ISO 3382)</h2><div class="cards">';
  if(m.spl_mic!=null)h+=card('SPL at mic',f(m.spl_mic,1),'dB \\u00B7 steady tone',dd('spl_mic'));
  if(m.spl_peak!=null)h+=card('Peak SPL at mic',f(m.spl_peak,1),'dB \\u00B7 pulse peak',dd('spl_peak'));
  if(!D.steady)
  h+=card('RT60 \\u00B7 T20 (measured)',f(m.rt60_t20,2),'s \\u00B7 Schroeder integration'+
    (m.rt60_t20==null?' \\u00B7 window too short':''),dd('rt60_t20'));
  h+=card('RT60 \\u00B7 Sabine',f(m.rt60_sabine,2),'s \\u00B7 0.161V/A',dd('rt60_sabine'));
  h+=card('RT60 \\u00B7 Eyring',f(m.rt60_eyring,2),'s',dd('rt60_eyring'));
  if(!D.steady){
  h+=card('EDT',f(m.edt,2),'s \\u00B7 early decay time',dd('edt'));
  h+=card('C50 clarity',f(m.c50,1),'dB \\u00B7 speech',dd('c50'));
  h+=card('C80 clarity',f(m.c80,1),'dB \\u00B7 music',dd('c80'));
  h+=card('D50 definition',m.d50==null?'\\u2014':(100*m.d50).toFixed(0)+'%',
    'early/total energy',dd('d50')==null?null:100*dd('d50'));}
  h+=card('Schroeder frequency',f(m.f_schroeder,0),'Hz \\u00B7 modal\\u2192statistical',null);
  h+=card('Volume',f(m.volume,1),'m\\u00B3',null);
  h+=card('Mean absorption',f(m.mean_alpha,3),'\\u0101 over '+f(m.area,0)+' m\\u00B2',null);
  h+='</div>';
  h+='<div style="margin-top:14px"><button class="audiobtn" onclick="downloadCSV()">'+
     '&#11015; Export metrics CSV</button> <button class="audiobtn" '+
     'onclick="downloadJSON()">&#11015; Export full result JSON</button></div>';
  if(m.bands&&m.bands.length)
    h+='<h2>Predicted RT60 per octave band</h2><div id="bandplot" style="height:220px"></div>';
  h+='<h2>Room modes (analytic, rigid walls)</h2><table><tr><th>#</th><th>f (Hz)</th>'+
     '<th>(nx,ny,nz)</th><th>type</th></tr>';
  (D.modes||[]).slice(0,14).forEach((mo,i)=>{h+=`<tr><td>${i+1}</td><td>${mo.f.toFixed(1)}</td>`+
    `<td>(${mo.n.join(',')})</td><td>${mo.type}</td></tr>`;});
  h+='</table>';
  $('metricsview').innerHTML=h;
  if(m.bands&&m.bands.length)
    Plotly.newPlot('bandplot',[
      {x:m.bands.map(b=>b.f+' Hz'),y:m.bands.map(b=>b.sabine),name:'Sabine',type:'bar',
       marker:{color:'#3b82f6'}},
      {x:m.bands.map(b=>b.f+' Hz'),y:m.bands.map(b=>b.eyring),name:'Eyring',type:'bar',
       marker:{color:'#22d3ee'}}],
      {template:'plotly_dark',paper_bgcolor:'#0d0f14',plot_bgcolor:'#0d0f14',
       yaxis:{title:'RT60 (s)'},barmode:'group',margin:{t:10}},{responsive:true});
}

// ---- exports ----
function dl(blob,name){const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download=name;a.click();}
function downloadCSV(){if(!D)return;
  let s='metric,value\\n';
  for(const k in D.metrics){const v=D.metrics[k];
    if(v==null||typeof v==='object')continue;s+=`${k},${v}\\n`;}
  (D.metrics.bands||[]).forEach(b=>{s+=`rt60_sabine_${b.f}Hz,${b.sabine}\\n`
    +`rt60_eyring_${b.f}Hz,${b.eyring}\\n`;});
  (D.modes||[]).forEach((mo,i)=>{s+=`mode_${i+1}_${mo.type},${mo.f}\\n`;});
  dl(new Blob([s],{type:'text/csv'}),'room_metrics.csv');}
function downloadJSON(){if(!D)return;
  dl(new Blob([JSON.stringify(D)],{type:'application/json'}),'roomwave_result.json');}
function downloadWAV(){if(!D||D.steady)return;
  const buf=irBuffer(),ch=buf.getChannelData(0),sr=buf.sampleRate,n=ch.length;
  const b=new ArrayBuffer(44+2*n),v=new DataView(b);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF');v.setUint32(4,36+2*n,true);ws(8,'WAVE');ws(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sr,true);v.setUint32(28,2*sr,true);v.setUint16(32,2,true);
  v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,2*n,true);
  for(let i=0;i<n;i++)v.setInt16(44+2*i,Math.max(-1,Math.min(1,ch[i]))*32767,true);
  dl(new Blob([b],{type:'audio/wav'}),'room_impulse_response.wav');}

// ---- auralization (WebAudio) ----
let actx=null,playing=[];
function irBuffer(){
  actx=actx||new (window.AudioContext||window.webkitAudioContext)();
  const sr=actx.sampleRate,h=D.ir.h,dt=D.ir.dt,n=Math.floor(h.length*dt*sr);
  const buf=actx.createBuffer(1,n,sr),ch=buf.getChannelData(0);
  let peak=1e-9;for(const x of h)peak=Math.max(peak,Math.abs(x));
  for(let i=0;i<n;i++){const x=i/(sr*dt),j=Math.floor(x),fr=x-j;
    ch[i]=((h[j]||0)*(1-fr)+(h[j+1]||0)*fr)/peak*0.8;}
  return buf;}
function stopAudio(){playing.forEach(s=>{try{s.stop()}catch(e){}});playing=[];}
function playIR(){if(!D||D.steady)return;
  const buf=irBuffer(),src=actx.createBufferSource();
  src.buffer=buf;src.connect(actx.destination);src.start();playing.push(src);}
function playThroughRoom(mkSrc){if(!D||D.steady)return;
  const buf=irBuffer(),conv=actx.createConvolver();conv.normalize=true;conv.buffer=buf;
  const s=mkSrc(actx),wet=actx.createGain(),dry=actx.createGain();
  wet.gain.value=parseFloat($('wet').value);dry.gain.value=1-0.7*wet.gain.value;
  s.connect(dry);dry.connect(actx.destination);
  s.connect(conv);conv.connect(wet);wet.connect(actx.destination);
  s.start();playing.push(s);}
function playKick(){playThroughRoom(ctx=>{
  const sr=ctx.sampleRate,n=Math.floor(0.4*sr),b=ctx.createBuffer(1,n,sr),c=b.getChannelData(0);
  for(let i=0;i<n;i++){const t=i/sr;
    c[i]=Math.sin(2*Math.PI*(40*t+110/8*(1-Math.exp(-8*t))))*Math.exp(-7*t);}
  const s=ctx.createBufferSource();s.buffer=b;return s;});}
function playClap(){playThroughRoom(ctx=>{
  const sr=ctx.sampleRate,n=Math.floor(0.12*sr),b=ctx.createBuffer(1,n,sr),c=b.getChannelData(0);
  for(let i=0;i<n;i++)c[i]=(Math.random()*2-1)*Math.exp(-25*i/sr);
  const s=ctx.createBufferSource();s.buffer=b;return s;});}
</script></body></html>""".replace('__MATS__', MAT_OPTIONS)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='text/html'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self._send(200, PAGE.encode())
        elif self.path == '/plotly.js':
            self._send(200, _plotly_js(), 'application/javascript')
        elif self.path == '/progress':
            self._send(200, json.dumps({k: JOB[k] for k in
                                        ('state', 'done', 'total', 'error')}).encode(),
                       'application/json')
        elif self.path == '/result':
            if JOB['state'] != 'done' or JOB['result'] is None:
                return self._send(409, b'no result available', 'text/plain')
            self._send(200, JOB['result'], 'application/json')
        else:
            self._send(404, b'not found')

    def do_POST(self):
        if self.path != '/simulate':
            return self._send(404, b'not found')
        with _job_lock:
            if JOB['state'] == 'running':
                return self._send(429, b'a simulation is already running', 'text/plain')
            JOB.update(state='running', done=0, total=0, result=None, error=None)
        n = int(self.headers.get('Content-Length', 0))
        params = json.loads(self.rfile.read(n) or b'{}')

        def work():
            try:
                JOB['result'] = json.dumps(simulate(params)).encode()
                JOB['state'] = 'done'
            except Exception as e:
                traceback.print_exc()
                JOB.update(state='error', error=str(e))

        threading.Thread(target=work, daemon=True).start()
        self._send(202, b'{"started": true}', 'application/json')

    def log_message(self, fmt, *args):  # quieter console
        if '/simulate' in (args[0] if args else ''):
            super().log_message(fmt, *args)


def main():
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f"RoomWave Studio  ->  http://localhost:{PORT}")
    srv.serve_forever()


if __name__ == '__main__':
    main()
