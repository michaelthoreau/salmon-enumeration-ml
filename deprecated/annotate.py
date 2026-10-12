"""
Minimal annotation viewer/editor + inference UI.
Usage:
    python annotate.py               # uses data/ next to this file
    python annotate.py --split=val   # start on val split
    python annotate.py --port=8080
"""
import base64
import glob
import io
import json
import logging
import os
import sys
import tempfile
import time

import fire
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
DATA_ROOT  = None   # set in main()

log = logging.getLogger('annotate')
RUNS_ROOT  = None   # set in main()
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# lazy-loaded inference model
_infer_model      = None
_infer_model_path = None

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a1a; color: #eee; display: flex; height: 100vh; }
#sidebar {
  width: 220px; flex-shrink: 0; background: #111; padding: 10px;
  display: flex; flex-direction: column; gap: 8px;
}
#sidebar h2 { font-size: 13px; color: #aaa; margin-bottom: 4px; }
select, button, input[type=number] { width: 100%; padding: 5px; background: #222; color: #eee; border: 1px solid #444; border-radius: 3px; cursor: pointer; }
button:hover { background: #333; }
#save-btn { background: #1a5; border-color: #2b6; }
#save-btn:hover { background: #2b6; }
#info { font-size: 11px; color: #888; line-height: 1.8; }
#status { font-size: 11px; color: #fa0; min-height: 16px; }
#canvas-wrap { flex: 1; overflow: auto; }
canvas { display: block; cursor: crosshair; }
</style>
</head>
<body>
<div id="sidebar">
  <h2>Split</h2>
  <select id="split-sel">
    <option value="train">train</option>
    <option value="val">val</option>
    <option value="test">test</option>
  </select>
  <h2>Navigate</h2>
  <button onclick="prevImg()">&#8592; Prev</button>
  <button onclick="nextImg()">Next &#8594;</button>
  <input type="number" id="idx-input" min="0" placeholder="index" title="Jump to index">
  <hr style="border-color:#333">
  <div id="info"></div>
  <div id="status"></div>
  <button id="save-btn" onclick="save()">Save (Ctrl+S)</button>
  <a href="/infer" style="font-size:12px;color:#5af;text-decoration:none">&#x1F52C; Run Inference</a>
  <a href="/manage" style="font-size:12px;color:#5af;text-decoration:none">&#x1F4C2; Data Manager</a>
  <a href="/label" style="font-size:12px;color:#5af;text-decoration:none">&#x270F; Labeler</a>
  <hr style="border-color:#333">
  <div style="font-size:11px;color:#666;line-height:1.6">
    Left-click: add point<br>
    Right-click: remove nearest<br>
    Scroll: zoom
  </div>
</div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<script>
const splitSel  = document.getElementById('split-sel');
const idxInput  = document.getElementById('idx-input');
const canvas    = document.getElementById('c');
const ctx       = canvas.getContext('2d');
const infoDiv   = document.getElementById('info');
const statusDiv = document.getElementById('status');

let ann = { points: [] }, imgEl = new Image();
let scale = 1.0, dirty = false;
let idx = 0, total = 0, currentName = '';

// ---------- load by index ----------
async function loadIdx(i) {
  if (dirty && !confirm('Unsaved changes — discard?')) return;
  dirty = false;
  statusDiv.textContent = 'Loading…';

  const split = splitSel.value;
  const r = await fetch(`/api/nav?split=${split}&idx=${i}`);
  if (!r.ok) { statusDiv.textContent = 'Error loading'; return; }
  const data = await r.json();

  idx         = data.idx;
  total       = data.total;
  currentName = data.name;
  ann         = data.annotation;
  idxInput.value = idx;

  imgEl = new Image();
  imgEl.onload = () => { fitScale(); draw(); statusDiv.textContent = ''; };
  imgEl.src = `/api/image?split=${split}&name=${currentName}`;
}

// ---------- fit to viewport ----------
function fitScale() {
  const wrap = document.getElementById('canvas-wrap');
  scale = Math.min((wrap.clientWidth - 4) / imgEl.naturalWidth,
                   (wrap.clientHeight - 4) / imgEl.naturalHeight, 1.0);
  canvas.width  = Math.round(imgEl.naturalWidth  * scale);
  canvas.height = Math.round(imgEl.naturalHeight * scale);
}

// ---------- draw ----------
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
  const r = Math.max(3, 5 * scale);
  ann.points.forEach(([x, y]) => {
    const cx = x * scale, cy = y * scale;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle   = 'rgba(255,50,50,0.55)';
    ctx.strokeStyle = '#ff3';
    ctx.lineWidth   = 1.5;
    ctx.fill(); ctx.stroke();
  });
  infoDiv.innerHTML =
    `${idx + 1} / ${total}<br>` +
    `<span style="color:#ccc">${currentName.slice(0,8)}…</span><br>` +
    `${ann.width}×${ann.height}<br>Points: <b>${ann.points.length}</b>`;
}

// ---------- click: add ----------
canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  ann.points.push([
    Math.round((e.clientX - rect.left) / scale),
    Math.round((e.clientY - rect.top)  / scale),
  ]);
  dirty = true; draw();
});

// ---------- right-click: remove nearest ----------
canvas.addEventListener('contextmenu', e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) / scale;
  const my = (e.clientY - rect.top)  / scale;
  let best = -1, bestD = Infinity;
  ann.points.forEach(([x, y], i) => {
    const d = Math.hypot(x - mx, y - my);
    if (d < bestD) { bestD = d; best = i; }
  });
  if (best >= 0 && bestD < 20 / scale) {
    ann.points.splice(best, 1);
    dirty = true; draw();
  }
});

// ---------- scroll: zoom ----------
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  scale *= e.deltaY < 0 ? 1.1 : 0.9;
  scale = Math.max(0.1, Math.min(scale, 4));
  canvas.width  = Math.round(imgEl.naturalWidth  * scale);
  canvas.height = Math.round(imgEl.naturalHeight * scale);
  draw();
}, { passive: false });

// ---------- save ----------
async function save() {
  const split = splitSel.value;
  const r = await fetch('/api/annotation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ split, name: currentName, points: ann.points }),
  });
  const j = await r.json();
  statusDiv.textContent = j.ok ? 'Saved.' : ('Error: ' + j.error);
  dirty = false;
}

// ---------- nav ----------
function prevImg() { if (idx > 0) loadIdx(idx - 1); }
function nextImg() { if (idx < total - 1) loadIdx(idx + 1); }

idxInput.addEventListener('change', () => {
  const v = parseInt(idxInput.value);
  if (!isNaN(v)) loadIdx(Math.max(0, Math.min(v, total - 1)));
});

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  prevImg();
  if (e.key === 'ArrowRight') nextImg();
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); save(); }
});

splitSel.addEventListener('change', () => loadIdx(0));

// ---------- init ----------
splitSel.value = new URLSearchParams(location.search).get('split') || 'train';
loadIdx(0);
</script>
</body>
</html>
"""

# ---- API ----

_image_cache = {}

def _get_images(split: str) -> list:
    if split not in _image_cache:
        images_dir = os.path.join(DATA_ROOT, split, "images")
        _image_cache[split] = sorted(
            f for f in os.listdir(images_dir) if f.endswith(".jpg")
        )
    return _image_cache[split]

@app.get("/")
def index():
    return HTML, 200, {"Content-Type": "text/html"}

@app.get("/api/nav")
def api_nav():
    split = request.args.get("split", "train")
    idx   = int(request.args.get("idx", 0))
    names = _get_images(split)
    idx   = max(0, min(idx, len(names) - 1))
    name  = names[idx]
    uuid  = name.replace(".jpg", "")
    ann_path = os.path.join(DATA_ROOT, split, "annotations", f"{uuid}.json")
    with open(ann_path) as f:
        ann = json.load(f)
    return jsonify({"idx": idx, "total": len(names), "name": name, "annotation": ann})

@app.get("/api/image")
def api_image():
    split = request.args.get("split", "train")
    name  = request.args.get("name")
    path  = os.path.join(DATA_ROOT, split, "images", name)
    resp = send_file(path, mimetype="image/jpeg", conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp

@app.get("/api/annotation")
def api_annotation_get():
    split = request.args.get("split", "train")
    name  = request.args.get("name")
    uuid  = name.replace(".jpg", "")
    path  = os.path.join(DATA_ROOT, split, "annotations", f"{uuid}.json")
    with open(path) as f:
        return jsonify(json.load(f))

@app.post("/api/annotation")
def api_annotation_post():
    body  = request.get_json()
    split = body.get("split")
    name  = body.get("name")
    uuid  = name.replace(".jpg", "")
    path  = os.path.join(DATA_ROOT, split, "annotations", f"{uuid}.json")
    try:
        with open(path) as f:
            ann = json.load(f)
        ann["points"] = body["points"]
        with open(path, "w") as f:
            json.dump(ann, f)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


MANAGE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Manager</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a1a; color: #eee; display: flex; height: 100vh; overflow: hidden; }
#sidebar {
  width: 240px; flex-shrink: 0; background: #111; padding: 12px;
  display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}
#sidebar h2 { font-size: 13px; color: #aaa; margin-bottom: 2px; }
button, select, input[type=number] { width: 100%; padding: 6px; background: #222; color: #eee;
  border: 1px solid #444; border-radius: 3px; cursor: pointer; }
button:hover { background: #333; }
.go-btn  { background: #1a5; border-color: #2b6; }
.go-btn:hover  { background: #2b6; }
.sel-btn { background: #224; border-color: #446; }
#status { font-size: 11px; color: #fa0; min-height: 16px; word-break: break-all; }
a.back { font-size: 12px; color: #5af; text-decoration: none; }
a.back:hover { text-decoration: underline; }
#main { flex: 1; overflow-y: auto; padding: 12px; }
#stats { font-size: 12px; color: #888; margin-bottom: 10px; }
#grid { display: flex; flex-wrap: wrap; gap: 8px; }
.card {
  width: 160px; background: #222; border: 2px solid #333; border-radius: 4px;
  overflow: hidden; cursor: pointer; user-select: none;
}
.card.selected { border-color: #2b6; }
.card.queued   { border-color: #f80; opacity: 0.6; }
.card img { width: 100%; height: 90px; object-fit: cover; display: block; }
.card .label { font-size: 10px; padding: 3px 5px; color: #aaa; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.card .badge { font-size: 9px; padding: 1px 4px; border-radius: 2px;
  background: #f80; color: #000; float: right; margin-top: -1px; }
</style>
</head>
<body>
<div id="sidebar">
  <a class="back" href="/">&#8592; Annotator</a>
  <h2>Data Manager</h2>
  <hr style="border-color:#333">
  <h2>Tile size (px)</h2>
  <input type="number" id="tile-size" value="640" min="128" max="2048" step="64">
  <h2>Overlap (px)</h2>
  <input type="number" id="tile-overlap" value="64" min="0" max="512" step="32">
  <hr style="border-color:#333">
  <button class="sel-btn" onclick="selectAll()">Select all unqueued</button>
  <button class="sel-btn" onclick="selectNone()">Deselect all</button>
  <button class="go-btn" onclick="queueSelected()">Queue selected &#8594; labeling</button>
  <button style="background:#422;border-color:#644;width:100%;padding:6px;color:#eee;border:1px solid #644;border-radius:3px;cursor:pointer" onmouseover="this.style.background='#533'" onmouseout="this.style.background='#422'" onclick="clearQueue()">&#x2715; Clear queue &amp; tiles</button>
  <hr style="border-color:#333">
  <a href="/label" style="font-size:12px;color:#5af;text-decoration:none">&#x270F; Go to Labeling &#8594;</a>
  <hr style="border-color:#333">
  <div id="status"></div>
</div>
<div id="main">
  <div id="stats"></div>
  <div id="grid"></div>
</div>

<script>
const statusDiv = document.getElementById('status');
const statsDiv  = document.getElementById('stats');
const grid      = document.getElementById('grid');
let images = [];   // [{name, queued}]

async function load() {
  statusDiv.textContent = 'Loading…';
  const r = await fetch('/api/raw/images');
  images = await r.json();
  const nQueued = images.filter(x => x.queued).length;
  statsDiv.textContent = `${images.length} raw images  |  ${nQueued} already queued  |  ${images.length - nQueued} available`;
  statusDiv.textContent = '';
  renderGrid();
}

function renderGrid() {
  grid.innerHTML = '';
  images.forEach((img, i) => {
    const card = document.createElement('div');
    card.className = 'card' + (img.queued ? ' queued' : '');
    card.dataset.idx = i;
    card.innerHTML = `
      <img src="/api/raw/thumb?name=${encodeURIComponent(img.name)}" loading="lazy">
      <div class="label">
        ${img.queued ? '<span class="badge">queued</span>' : ''}
        ${img.name}
      </div>`;
    if (!img.queued) {
      card.addEventListener('click', () => {
        card.classList.toggle('selected');
      });
    }
    grid.appendChild(card);
  });
}

function selectAll() {
  grid.querySelectorAll('.card:not(.queued)').forEach(c => c.classList.add('selected'));
}
function selectNone() {
  grid.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
}

async function queueSelected() {
  const selected = [...grid.querySelectorAll('.card.selected')].map(c => images[c.dataset.idx].name);
  if (!selected.length) { statusDiv.textContent = 'Nothing selected.'; return; }
  const tileSize    = parseInt(document.getElementById('tile-size').value);
  const tileOverlap = parseInt(document.getElementById('tile-overlap').value);
  statusDiv.textContent = `Queuing ${selected.length} image(s)…`;
  const r = await fetch('/api/raw/queue', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names: selected, tile_size: tileSize, overlap: tileOverlap }),
  });
  const j = await r.json();
  if (j.ok) {
    statusDiv.textContent = `Done — ${j.tiles} tiles created at ${tileSize}px.`;
    load();
  } else {
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

async function clearQueue() {
  if (!confirm('Delete all tiles in labeling/images and reset the queue? Committed annotations are kept.')) return;
  statusDiv.textContent = 'Clearing…';
  const r = await fetch('/api/raw/clear', { method: 'POST' });
  const j = await r.json();
  if (j.ok) {
    statusDiv.textContent = `Cleared — ${j.deleted} tiles removed.`;
    load();
  } else {
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

load();
</script>
</body>
</html>
"""

LABEL_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Labeler</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a1a; color: #eee; display: flex; height: 100vh; overflow: hidden; }
#sidebar {
  width: 220px; flex-shrink: 0; background: #111; padding: 10px;
  display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}
#sidebar h2 { font-size: 13px; color: #aaa; margin-bottom: 4px; }
select, button, input[type=number] { width: 100%; padding: 5px; background: #222; color: #eee;
  border: 1px solid #444; border-radius: 3px; cursor: pointer; }
button:hover { background: #333; }
.commit-btn { background: #1a5; border-color: #2b6; }
.commit-btn:hover { background: #2b6; }
.save-btn { background: #225; border-color: #447; }
.save-btn:hover { background: #336; }
.skip-btn { background: #422; border-color: #644; }
.skip-btn:hover { background: #533; }
#info { font-size: 11px; color: #888; line-height: 1.8; }
#status { font-size: 11px; color: #fa0; min-height: 16px; }
#viewport { flex: 1; overflow: hidden; position: relative; cursor: crosshair; }
canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
a.back { font-size: 12px; color: #5af; text-decoration: none; }
a.back:hover { text-decoration: underline; }
#zoom-info { font-size: 10px; color: #555; }
</style>
</head>
<body>
<div id="sidebar">
  <a class="back" href="/manage">&#8592; Data Manager</a>
  <h2>Labeler</h2>
  <h2>Navigate</h2>
  <button onclick="prevImg()">&#8592; Prev</button>
  <button onclick="nextImg()">Next &#8594;</button>
  <input type="number" id="idx-input" min="0" placeholder="index">
  <button onclick="fitView()">Fit view</button>
  <div id="zoom-info"></div>
  <hr style="border-color:#333">
  <div id="info"></div>
  <div id="status"></div>
  <hr style="border-color:#333">
  <button class="save-btn" onclick="savePoints()">&#128190; Save annotations</button>
  <hr style="border-color:#333">
  <h2>Commit to</h2>
  <select id="split-sel">
    <option value="train">train</option>
    <option value="val">val</option>
    <option value="test">test</option>
  </select>
  <button class="commit-btn" onclick="commit()">&#10003; Commit (Ctrl+Enter)</button>
  <button class="skip-btn" onclick="skip()">Skip (no fish)</button>
  <hr style="border-color:#333">
  <div style="font-size:11px;color:#666;line-height:1.8">
    Click: add point<br>
    Right-click: remove nearest<br>
    Scroll: zoom (cursor anchor)<br>
    Middle-drag / Space+drag: pan<br>
    Ctrl+S: save<br>
    ←/→: prev/next image
  </div>
</div>
<div id="viewport">
  <canvas id="c"></canvas>
</div>

<script>
const idxInput  = document.getElementById('idx-input');
const canvas    = document.getElementById('c');
const ctx       = canvas.getContext('2d');
const infoDiv   = document.getElementById('info');
const statusDiv = document.getElementById('status');
const viewport  = document.getElementById('viewport');
const zoomInfo  = document.getElementById('zoom-info');

let points = [], imgEl = new Image();
// transform state: canvas is rendered at natural size, CSS transform scales/translates
let scale = 1.0, panX = 0, panY = 0;
let dirty = false;
let idx = 0, total = 0, currentName = '', committed = false;

// ---- transform helpers ----
function applyTransform() {
  canvas.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
  zoomInfo.textContent = `zoom: ${Math.round(scale * 100)}%`;
}

function viewportToCanvas(vx, vy) {
  return [(vx - panX) / scale, (vy - panY) / scale];
}

// ---- load ----
async function loadIdx(i) {
  if (dirty && !confirm('Unsaved changes — discard?')) return;
  dirty = false;
  statusDiv.textContent = 'Loading…';

  const r = await fetch(`/api/label/nav?idx=${i}`);
  if (!r.ok) { statusDiv.textContent = 'Error'; return; }
  const data = await r.json();

  idx         = data.idx;
  total       = data.total;
  currentName = data.name;
  committed   = data.committed;
  points      = data.points || [];
  idxInput.value = idx;

  imgEl = new Image();
  imgEl.onload = () => {
    canvas.width  = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    fitView();
    draw();
    statusDiv.textContent = committed ? 'Already committed.' : '';
  };
  imgEl.src = `/api/label/image?name=${encodeURIComponent(currentName)}&t=${Date.now()}`;
}

function fitView() {
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  scale = Math.min(vw / canvas.width, vh / canvas.height, 1.0);
  panX  = Math.round((vw - canvas.width  * scale) / 2);
  panY  = Math.round((vh - canvas.height * scale) / 2);
  applyTransform();
}

// ---- draw ----
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = committed ? 0.45 : 1.0;
  ctx.drawImage(imgEl, 0, 0);
  ctx.globalAlpha = 1.0;
  const r = Math.max(4, 6 / scale);
  points.forEach(([x, y]) => {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,50,50,0.6)'; ctx.strokeStyle = '#ff3';
    ctx.lineWidth = Math.max(1, 1.5 / scale);
    ctx.fill(); ctx.stroke();
  });
  infoDiv.innerHTML =
    `${idx + 1} / ${total}<br>` +
    `<span style="color:#ccc">${currentName.replace(/(.{12}).+/, '$1…')}</span><br>` +
    `${canvas.width}×${canvas.height}<br>` +
    `Points: <b>${points.length}</b>` +
    (committed ? `<br><span style="color:#f80">committed</span>` : '');
}

// ---- pan state ----
let isPanning = false, spaceDown = false, panStart = null;

viewport.addEventListener('mousedown', e => {
  if (e.button === 1 || (e.button === 0 && spaceDown)) {
    e.preventDefault();
    isPanning = true;
    panStart  = { x: e.clientX - panX, y: e.clientY - panY };
    viewport.style.cursor = 'grabbing';
    return;
  }
  if (e.button === 0) {
    if (committed) return;
    const rect = viewport.getBoundingClientRect();
    const [cx, cy] = viewportToCanvas(e.clientX - rect.left, e.clientY - rect.top);
    points.push([Math.round(cx), Math.round(cy)]);
    dirty = true; draw();
  }
});

window.addEventListener('mousemove', e => {
  if (!isPanning) return;
  panX = e.clientX - panStart.x;
  panY = e.clientY - panStart.y;
  applyTransform();
});

window.addEventListener('mouseup', e => {
  if (isPanning) {
    isPanning = false;
    viewport.style.cursor = spaceDown ? 'grab' : 'crosshair';
  }
});

viewport.addEventListener('contextmenu', e => {
  e.preventDefault();
  if (committed) return;
  const rect = viewport.getBoundingClientRect();
  const [mx, my] = viewportToCanvas(e.clientX - rect.left, e.clientY - rect.top);
  let best = -1, bestD = Infinity;
  points.forEach(([x, y], i) => { const d = Math.hypot(x - mx, y - my); if (d < bestD) { bestD = d; best = i; } });
  const pickRadius = 20 / scale;
  if (best >= 0 && bestD < pickRadius) { points.splice(best, 1); dirty = true; draw(); }
});

// ---- zoom centred on cursor ----
viewport.addEventListener('wheel', e => {
  e.preventDefault();
  const rect   = viewport.getBoundingClientRect();
  const vx     = e.clientX - rect.left;
  const vy     = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newScale = Math.max(0.05, Math.min(scale * factor, 16));
  // adjust pan so the point under the cursor stays fixed
  panX = vx - (vx - panX) * (newScale / scale);
  panY = vy - (vy - panY) * (newScale / scale);
  scale = newScale;
  applyTransform();
}, { passive: false });

// ---- spacebar pan mode ----
document.addEventListener('keydown', e => {
  if (e.key === ' ' && !e.target.matches('input,select,textarea')) {
    e.preventDefault();
    if (!spaceDown) { spaceDown = true; viewport.style.cursor = 'grab'; }
  }
  if (e.key === 'ArrowLeft')  prevImg();
  if (e.key === 'ArrowRight') nextImg();
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); savePoints(); }
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); commit(); }
});
document.addEventListener('keyup', e => {
  if (e.key === ' ') { spaceDown = false; viewport.style.cursor = 'crosshair'; }
});

// ---- save (without committing) ----
async function savePoints() {
  statusDiv.textContent = 'Saving…';
  const r = await fetch('/api/label/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: currentName, points }),
  });
  const j = await r.json();
  if (j.ok) { dirty = false; statusDiv.textContent = `Saved (${points.length} pts).`; }
  else       { statusDiv.textContent = 'Error: ' + j.error; }
}

// ---- commit ----
async function commit() {
  const split = document.getElementById('split-sel').value;
  statusDiv.textContent = 'Committing…';
  const r = await fetch('/api/label/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: currentName, split, points }),
  });
  const j = await r.json();
  if (j.ok) {
    statusDiv.textContent = `Committed to ${split}.`;
    dirty = false; committed = true; draw();
    setTimeout(() => { if (idx < total - 1) loadIdx(idx + 1); }, 600);
  } else {
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

async function skip() {
  statusDiv.textContent = 'Skipping…';
  const r = await fetch('/api/label/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: currentName, split: null, points: [] }),
  });
  const j = await r.json();
  if (j.ok) {
    dirty = false; committed = true; draw();
    if (idx < total - 1) loadIdx(idx + 1);
  } else {
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

function prevImg() { if (idx > 0) loadIdx(idx - 1); }
function nextImg() { if (idx < total - 1) loadIdx(idx + 1); }
idxInput.addEventListener('change', () => {
  const v = parseInt(idxInput.value);
  if (!isNaN(v)) loadIdx(Math.max(0, Math.min(v, total - 1)));
});

loadIdx(0);
</script>
</body>
</html>
"""

INFER_HTML = r"""
<!DOCTYPE html>
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

async function skip() {
  statusDiv.textContent = 'Skipping…';
  const r = await fetch('/api/label/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: currentName, split: null, points: [] }),
  });
  const j = await r.json();
  if (j.ok) {
    dirty = false; committed = true;
    if (idx < total - 1) loadIdx(idx + 1);
  } else {
    statusDiv.textContent = 'Error: ' + j.error;
  }
}

function prevImg() { if (idx > 0) loadIdx(idx - 1); }
function nextImg() { if (idx < total - 1) loadIdx(idx + 1); }
idxInput.addEventListener('change', () => {
  const v = parseInt(idxInput.value);
  if (!isNaN(v)) loadIdx(Math.max(0, Math.min(v, total - 1)));
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  prevImg();
  if (e.key === 'ArrowRight') nextImg();
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); commit(); }
});

loadIdx(0);
</script>
</body>
</html>
"""

INFER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Inference</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #1a1a1a; color: #eee; display: flex; height: 100vh; }
#sidebar {
  width: 260px; flex-shrink: 0; background: #111; padding: 12px;
  display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}
#sidebar h2 { font-size: 13px; color: #aaa; margin-bottom: 2px; }
select, button { width: 100%; padding: 6px; background: #222; color: #eee;
  border: 1px solid #444; border-radius: 3px; cursor: pointer; }
button:hover { background: #333; }
#run-btn { background: #1a5; border-color: #2b6; }
#run-btn:hover { background: #2b6; }
#drop-zone {
  border: 2px dashed #555; border-radius: 4px; padding: 16px;
  text-align: center; color: #666; font-size: 12px; cursor: pointer;
  transition: border-color 0.2s;
}
#drop-zone.over { border-color: #2b6; color: #eee; }
#info { font-size: 11px; color: #888; line-height: 1.8; }
#status { font-size: 11px; color: #fa0; min-height: 16px; }
a.back { font-size: 12px; color: #5af; text-decoration: none; }
a.back:hover { text-decoration: underline; }
#zoom-hint { font-size: 10px; color: #555; }
#main { flex: 1; overflow: hidden; position: relative; cursor: grab; }
#main.grabbing { cursor: grabbing; }
#viewport { position: absolute; top: 0; left: 0;
            transform-origin: 0 0; display: inline-block; }
#viewport img { display: block; max-width: none; }
</style>
</head>
<body>
<div id="sidebar">
  <a class="back" href="/">&#8592; Annotator</a>
  <h2>Inference</h2>
  <h2>Model</h2>
  <select id="model-sel"><option value="">Loading…</option></select>
  <div id="info"></div>
  <h2>Image</h2>
  <div id="drop-zone" onclick="document.getElementById('file-in').click()">
    Drop image here<br>or click to browse
  </div>
  <input type="file" id="file-in" accept="image/*" style="display:none">
  <h2>Threshold</h2>
  <div style="display:flex;gap:6px;align-items:center">
    <input type="range" id="thr-range" min="0" max="1" step="0.005" value="0.35"
           style="flex:1" oninput="document.getElementById('thr-num').value=parseFloat(this.value).toFixed(3)">
    <input type="number" id="thr-num" min="0" max="1" step="0.005" value="0.35"
           style="width:62px;padding:4px;background:#222;color:#eee;border:1px solid #444;border-radius:3px"
           oninput="document.getElementById('thr-range').value=this.value">
  </div>
  <div id="status"></div>
  <button id="run-btn" onclick="runInference()">Run Inference</button>
  <div id="zoom-hint">Scroll to zoom &middot; drag to pan</div>
</div>
<div id="main">
  <div id="viewport">
    <img id="preview" src="" alt="">
  </div>
</div>

<script>
const modelSel  = document.getElementById('model-sel');
const dropZone  = document.getElementById('drop-zone');
const fileIn    = document.getElementById('file-in');
const mainDiv   = document.getElementById('main');
const viewport  = document.getElementById('viewport');
const preview   = document.getElementById('preview');
const infoDiv   = document.getElementById('info');
const statusDiv = document.getElementById('status');

let uploadedFile = null;
let zoom = 1.0;
let panX = 0, panY = 0;
let dragStart = null;

function applyTransform() {
  viewport.style.transform = `translate(${panX}px,${panY}px) scale(${zoom})`;
}

function fitToWindow(natW, natH) {
  const mw = mainDiv.clientWidth, mh = mainDiv.clientHeight;
  zoom = Math.min(mw / natW, mh / natH, 1.0);
  panX = Math.round((mw - natW * zoom) / 2);
  panY = Math.round((mh - natH * zoom) / 2);
  applyTransform();
}

// Scroll to zoom around cursor
mainDiv.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = mainDiv.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newZoom = Math.min(Math.max(zoom * factor, 0.05), 20);
  panX = mx - (mx - panX) * (newZoom / zoom);
  panY = my - (my - panY) * (newZoom / zoom);
  zoom = newZoom;
  applyTransform();
}, { passive: false });

// Drag to pan
mainDiv.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  dragStart = { x: e.clientX - panX, y: e.clientY - panY };
  mainDiv.classList.add('grabbing');
});
window.addEventListener('mousemove', e => {
  if (!dragStart) return;
  panX = e.clientX - dragStart.x;
  panY = e.clientY - dragStart.y;
  applyTransform();
});
window.addEventListener('mouseup', () => { dragStart = null; mainDiv.classList.remove('grabbing'); });

// Load model list
async function loadModels() {
  const r = await fetch('/api/models');
  const models = await r.json();
  modelSel.innerHTML = '';
  if (!models.length) {
    modelSel.innerHTML = '<option value="">No models found</option>';
    return;
  }
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.path; o.textContent = m.label;
    modelSel.appendChild(o);
  });
  updateModelInfo();
}
function updateModelInfo() {
  const opt = modelSel.options[modelSel.selectedIndex];
  infoDiv.textContent = opt ? opt.value.replace(/.*\/runs\//, 'runs/') : '';
}
modelSel.addEventListener('change', updateModelInfo);

// Drag & drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});
fileIn.addEventListener('change', () => { if (fileIn.files[0]) setFile(fileIn.files[0]); });

function setFile(f) {
  uploadedFile = f;
  dropZone.textContent = f.name;
  const url = URL.createObjectURL(f);
  preview.onload = () => fitToWindow(preview.naturalWidth, preview.naturalHeight);
  preview.src = url;
  statusDiv.textContent = '';
}

async function runInference() {
  if (!uploadedFile) { statusDiv.textContent = 'Upload an image first.'; return; }
  const modelPath = modelSel.value;
  if (!modelPath) { statusDiv.textContent = 'Select a model first.'; return; }

  statusDiv.textContent = 'Running…';
  const fd = new FormData();
  fd.append('image', uploadedFile);
  fd.append('model', modelPath);
  fd.append('threshold', document.getElementById('thr-num').value);

  const r = await fetch('/api/infer', { method: 'POST', body: fd });
  const j = await r.json();

  if (!j.ok) { statusDiv.textContent = 'Error: ' + j.error; return; }

  // Swap in overlay image — keep current zoom/pan
  const natW = preview.naturalWidth, natH = preview.naturalHeight;
  preview.onload = () => {
    // restore dimensions (JPEG from server is same size as input)
    if (natW && preview.naturalWidth !== natW) fitToWindow(preview.naturalWidth, preview.naturalHeight);
  };
  preview.src = 'data:image/jpeg;base64,' + j.image;
  statusDiv.textContent = `Count: ${j.count}  (threshold ${j.threshold})`;
}

loadModels();
</script>
</body>
</html>
"""

# ---- inference helpers ----

def _load_model(model_path: str):
    global _infer_model, _infer_model_path
    if _infer_model_path == model_path and _infer_model is not None:
        return _infer_model

    sys.path.insert(0, BASE_DIR)
    from types import SimpleNamespace
    import torch
    from Networks.CDETR import build_model

    cfg = SimpleNamespace(
        backbone='resnet50', dilation=True, lr_backbone=1e-4,
        position_embedding='sine', masks=False, frozen_weights=None,
        enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=256,
        dropout=0.1, nheads=8, num_queries=700, pre_norm=False,
        transformer_flag='merge3', dm_count=True, branch_merge=True,
        branch_merge_way=2, channel_point=3, only_dm=False, with_weights=False,
        aux_loss=True, cls_loss_coef=2, point_loss_coef=5, giou_loss_coef=2,
        mask_loss_coef=1, dice_loss_coef=1, set_cost_class=2, set_cost_point=5,
        set_cost_giou=2, focal_alpha=0.25, dataset_file='crowd_data', device='cuda',
    )
    model, _, _ = build_model(cfg)
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    # strip DataParallel prefix if present
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    _infer_model = model
    _infer_model_path = model_path
    return model


def _run_infer(model, img_path: str, threshold: float = 0.35, batch_size: int = 8):
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    t_total = time.perf_counter()

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    CROP = 256

    device = next(model.parameters()).device
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

    t0 = time.perf_counter()
    pil = Image.open(img_path).convert('RGB')
    img_t = tf(pil)  # [3, H, W]
    print(f"[infer] load+transform  {time.perf_counter()-t0:.3f}s")

    # Pad so both dims are multiples of CROP (matches dataset.py test mode)
    t0 = time.perf_counter()
    _, H, W = img_t.shape
    pad_h = (CROP - H % CROP) % CROP
    pad_w = (CROP - W % CROP) % CROP
    img_pad = F.pad(img_t, (0, pad_w, 0, pad_h), 'constant', 0)  # [3, H_pad, W_pad]
    _, H_pad, W_pad = img_pad.shape
    num_h = H_pad // CROP
    num_w = W_pad // CROP
    n_patches = num_h * num_w

    # Tile into (P, 3, CROP, CROP) patches
    all_patches = (img_pad
                   .view(3, num_h, CROP, W_pad)
                   .view(3, num_h, CROP, num_w, CROP)
                   .permute(0, 1, 3, 2, 4)
                   .contiguous()
                   .view(3, n_patches, CROP, CROP)
                   .permute(1, 0, 2, 3))  # [P, 3, CROP, CROP]  (CPU)
    print(f"[infer] tiling          {time.perf_counter()-t0:.3f}s  ({n_patches} patches: {num_h}×{num_w})")

    # Run model in batches to avoid OOM on large images
    t0 = time.perf_counter()
    all_logits = []
    all_points = []
    count = 0.0
    for start in range(0, n_patches, batch_size):
        batch = all_patches[start:start + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
        if isinstance(out, list):
            d6, out_dm = out
            count += float(out_dm[1].sum().item())
        else:
            d6 = out
        all_logits.append(d6['pred_logits'].cpu())
        all_points.append(d6['pred_points'].cpu())
    all_logits = torch.cat(all_logits, dim=0)  # [P, Q, 2]
    all_points = torch.cat(all_points, dim=0)  # [P, Q, 3]
    scores = all_logits.sigmoid()[:, :, 1]     # [P, Q]
    print(f"[infer] model forward   {time.perf_counter()-t0:.3f}s  (batch_size={batch_size}, dm_count={count:.1f})")

    # Map patch-local normalised coords back to full-image pixel coords
    t0 = time.perf_counter()
    import cv2
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Build origin offsets for every patch: shape [P, 2] (origin_y, origin_x)
    patch_rows = torch.arange(num_h).repeat_interleave(num_w)
    patch_cols = torch.arange(num_w).repeat(num_h)
    origins_y = patch_rows * CROP  # [P]
    origins_x = patch_cols * CROP  # [P]

    n_pts = 0
    for patch_idx in range(n_patches):
        keep = scores[patch_idx] > threshold
        pts  = all_points[patch_idx][keep].numpy()[:, :2]  # [N, 2] (y_norm, x_norm)
        oy, ox = int(origins_y[patch_idx]), int(origins_x[patch_idx])
        for (yn, xn) in pts:
            py = oy + yn * CROP
            px = ox + xn * CROP
            if py >= H or px >= W:
                continue
            cv2.circle(arr, (int(px), int(py)), 6, (0, 60, 255), -1)
            cv2.circle(arr, (int(px), int(py)), 6, (0, 255, 255), 1)
            n_pts += 1
    print(f"[infer] draw points     {time.perf_counter()-t0:.3f}s  ({n_pts} points, thr={threshold})")

    t0 = time.perf_counter()
    _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    debug_path = os.path.join(debug_dir, f"{stem}_overlay.jpg")
    cv2.imwrite(debug_path, arr)
    print(f"[infer] encode+save     {time.perf_counter()-t0:.3f}s  -> {debug_path}")
    print(f"[infer] TOTAL           {time.perf_counter()-t_total:.3f}s")

    img_b64 = base64.b64encode(buf).decode()
    return img_b64, n_pts, threshold


# ---- Manage / Label globals ----

RAW_ROOT     = None   # set in main() — train/raw/images/
LABELING_ROOT = None  # set in main() — labeling/

def _raw_queued_path():
    return os.path.join(RAW_ROOT, 'queued.json')

def _load_queued():
    p = _raw_queued_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

def _save_queued(q):
    with open(_raw_queued_path(), 'w') as f:
        json.dump(q, f, indent=2)

def _label_index_path():
    return os.path.join(LABELING_ROOT, 'index.json')

def _load_label_index():
    p = _label_index_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}   # name -> {committed: bool, split: str|null}

def _save_label_index(idx):
    with open(_label_index_path(), 'w') as f:
        json.dump(idx, f, indent=2)

def _label_images():
    d = os.path.join(LABELING_ROOT, 'images')
    if not os.path.exists(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.jpg'))


# ---- Manage routes ----

@app.get("/manage")
def manage_page():
    return MANAGE_HTML, 200, {"Content-Type": "text/html"}

@app.get("/api/raw/images")
def api_raw_images():
    raw_images_dir = os.path.join(RAW_ROOT, 'images')
    if not os.path.exists(raw_images_dir):
        return jsonify([])
    queued = _load_queued()
    names = sorted(f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return jsonify([{"name": n, "queued": n in queued} for n in names])

@app.get("/api/raw/thumb")
def api_raw_thumb():
    from PIL import Image as PILImage
    name = request.args.get("name")
    path = os.path.join(RAW_ROOT, 'images', name)
    pil  = PILImage.open(path).convert('RGB')
    pil.thumbnail((320, 180))
    buf = io.BytesIO()
    pil.save(buf, 'JPEG', quality=70)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.post("/api/raw/queue")
def api_raw_queue():
    from PIL import Image as PILImage
    body      = request.get_json()
    names     = body.get("names", [])
    tile_size = int(body.get("tile_size", 640))
    overlap   = int(body.get("overlap", 64))
    stride    = max(tile_size - overlap, 1)
    log.info("queue request: %d images, tile_size=%d, overlap=%d", len(names), tile_size, overlap)

    queued      = _load_queued()
    label_idx   = _load_label_index()
    images_out  = os.path.join(LABELING_ROOT, 'images')
    os.makedirs(images_out, exist_ok=True)

    total_tiles = 0
    try:
        for name in names:
            if name in queued:
                continue
            src = os.path.join(RAW_ROOT, 'images', name)
            pil = PILImage.open(src).convert('RGB')
            W, H = pil.size
            stem = os.path.splitext(name)[0]
            tile_names = []

            # generate overlapping tiles
            y = 0
            row = 0
            while y < H:
                x = 0
                col = 0
                while x < W:
                    box = (x, y, min(x + tile_size, W), min(y + tile_size, H))
                    tile = pil.crop(box)
                    tile_name = f"{stem}_r{row:03d}_c{col:03d}.jpg"
                    tile.save(os.path.join(images_out, tile_name), 'JPEG', quality=92)
                    tile_names.append(tile_name)
                    if tile_name not in label_idx:
                        label_idx[tile_name] = {"committed": False, "split": None, "points": []}
                    # store tile metadata for annotation
                    label_idx[tile_name]["source"] = name
                    label_idx[tile_name]["box"] = list(box)
                    label_idx[tile_name]["size"] = [tile.width, tile.height]
                    total_tiles += 1
                    x += stride
                    col += 1
                    if x >= W: break
                y += stride
                row += 1
                if y >= H: break

            queued[name] = {"tiles": tile_names, "tile_size": tile_size, "overlap": overlap}
            log.info("queued %s -> %d tiles", name, len(tile_names))

        _save_queued(queued)
        _save_label_index(label_idx)
        return jsonify({"ok": True, "tiles": total_tiles})
    except Exception as e:
        log.exception("queue error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/raw/clear")
def api_raw_clear():
    import shutil
    try:
        images_dir = os.path.join(LABELING_ROOT, 'images')
        deleted = 0
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir):
                if f.lower().endswith('.jpg'):
                    os.remove(os.path.join(images_dir, f))
                    deleted += 1
        # reset queued.json and label index (keep committed entries)
        label_idx = _load_label_index()
        label_idx = {k: v for k, v in label_idx.items() if v.get('committed')}
        _save_label_index(label_idx)
        _save_queued({})
        log.info("cleared queue — %d tiles deleted", deleted)
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        log.exception("clear error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- Label routes ----

@app.get("/label")
def label_page():
    return LABEL_HTML, 200, {"Content-Type": "text/html"}

@app.get("/api/label/nav")
def api_label_nav():
    images    = _label_images()
    label_idx = _load_label_index()
    i = int(request.args.get("idx", 0))
    i = max(0, min(i, len(images) - 1))
    if not images:
        return jsonify({"idx": 0, "total": 0, "name": "", "committed": False, "points": []})
    name = images[i]
    info = label_idx.get(name, {})
    return jsonify({
        "idx": i,
        "total": len(images),
        "name": name,
        "committed": info.get("committed", False),
        "split": info.get("split"),
        "points": info.get("points", []),
    })

@app.get("/api/label/image")
def api_label_image():
    name = request.args.get("name")
    path = os.path.join(LABELING_ROOT, 'images', name)
    resp = send_file(path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.post("/api/label/save")
def api_label_save():
    body   = request.get_json()
    name   = body.get("name")
    points = body.get("points", [])
    label_idx = _load_label_index()
    info   = label_idx.get(name, {})
    info["points"] = points
    label_idx[name] = info
    _save_label_index(label_idx)
    return jsonify({"ok": True})

@app.post("/api/label/commit")
def api_label_commit():
    import shutil
    from PIL import Image as PILImage
    body   = request.get_json()
    name   = body.get("name")
    split  = body.get("split")    # None = skip
    points = body.get("points", [])

    label_idx = _load_label_index()
    info      = label_idx.get(name, {})

    # mark as committed regardless
    info["committed"] = True
    info["split"]     = split
    info["points"]    = points
    label_idx[name]   = info
    _save_label_index(label_idx)

    if split:
        import uuid as _uuid
        # copy image + write annotation into data/{split}/
        new_id   = str(_uuid.uuid4())
        src_img  = os.path.join(LABELING_ROOT, 'images', name)
        dst_dir_img = os.path.join(DATA_ROOT, split, 'images')
        dst_dir_ann = os.path.join(DATA_ROOT, split, 'annotations')
        os.makedirs(dst_dir_img, exist_ok=True)
        os.makedirs(dst_dir_ann, exist_ok=True)

        tile_size = info.get("size", [0, 0])
        W, H = tile_size if tile_size else [0, 0]
        if not W or not H:
            pil = PILImage.open(src_img)
            W, H = pil.size

        shutil.copy2(src_img, os.path.join(dst_dir_img, f"{new_id}.jpg"))
        ann = {"filename": f"{new_id}.jpg", "width": W, "height": H, "points": points}
        with open(os.path.join(dst_dir_ann, f"{new_id}.json"), 'w') as f:
            json.dump(ann, f)

        # invalidate image cache so new image shows up
        _image_cache.pop(split, None)
        log.info("committed %s -> %s/%s (%d pts)", name, split, new_id, len(points))

    return jsonify({"ok": True})


# ---- API: inference page ----

@app.get("/infer")
def infer_page():
    return INFER_HTML, 200, {"Content-Type": "text/html"}

@app.get("/api/models")
def api_models():
    models = []
    pattern = os.path.join(RUNS_ROOT, "**", "*.pth")
    for path in sorted(glob.glob(pattern, recursive=True)):
        label = os.path.relpath(path, RUNS_ROOT)
        models.append({"path": path, "label": label})
    return jsonify(models)

@app.post("/api/infer")
def api_infer():
    try:
        model_path = request.form.get("model")
        img_file   = request.files.get("image")
        threshold  = float(request.form.get("threshold", 0.35))
        filename   = img_file.filename or "<unknown>"

        log.info("infer  image=%s  model=%s  thr=%.4f", filename,
                 os.path.relpath(model_path) if model_path else None, threshold)
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img_file.save(tmp.name)
            tmp_path = tmp.name

        model   = _load_model(model_path)
        img_b64, count, thr = _run_infer(model, tmp_path, threshold)
        os.unlink(tmp_path)
        elapsed = time.perf_counter() - t0
        log.info("infer  done  count=%d  thr=%.4f  elapsed=%.2fs", count, thr, elapsed)
        return jsonify({"ok": True, "image": img_b64, "count": count, "threshold": thr})
    except Exception as e:
        log.exception("infer  error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


def main(split: str = "train", port: int = 5000,
         data_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
         runs_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
         raw_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train", "raw"),
         labeling_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "labeling")):
    global DATA_ROOT, RUNS_ROOT, RAW_ROOT, LABELING_ROOT
    DATA_ROOT     = data_root
    RUNS_ROOT     = runs_root
    RAW_ROOT      = raw_root
    LABELING_ROOT = labeling_root
    os.makedirs(os.path.join(raw_root, 'images'),  exist_ok=True)
    os.makedirs(os.path.join(data_root, 'input', 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'input', 'videos'), exist_ok=True)
    os.makedirs(os.path.join(labeling_root, 'images'), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)-7s  %(message)s',
                        datefmt='%H:%M:%S')
    import socket
    host = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = None

    # Patch werkzeug's _log to inject our hostname URL into the startup output
    import werkzeug.serving as _ws
    _original_log = _ws._log
    _injected = False
    def _patched_log(type_, message, *args, **kwargs):
        nonlocal _injected
        if not _injected and 'Running on' in (message % args if args else message):
            _injected = True
            _original_log(type_, f' * Running on http://{host}:{port}')
        _original_log(type_, message, *args, **kwargs)
    _ws._log = _patched_log

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    fire.Fire(main)
