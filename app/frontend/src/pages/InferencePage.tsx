import { useState, useRef, useEffect, useCallback } from 'react'

interface Model { path: string; label: string }

type Tab = 'overlay' | 'density'

export default function InferencePage() {
  const [models, setModels]         = useState<Model[]>([])
  const [selectedModel, setSelected] = useState('')
  const [threshold, setThreshold]   = useState(0.35)
  const [status, setStatus]         = useState('')
  const [running, setRunning]       = useState(false)
  const [tab, setTab]               = useState<Tab>('overlay')

  // Results
  const [overlayB64, setOverlayB64]   = useState<string | null>(null)
  const [dmB64, setDmB64]             = useState<string | null>(null)
  const [count, setCount]             = useState<number | null>(null)
  const [dmCount, setDmCount]         = useState<number | null>(null)
  const [runId, setRunId]             = useState(0)  // bumped each run to bust img cache

  const fileRef    = useRef<HTMLInputElement>(null)
  const dropRef    = useRef<HTMLDivElement>(null)
  const fileObject = useRef<File | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  // Zoom/pan state
  const viewRef   = useRef<HTMLDivElement>(null)
  const vpRef     = useRef<HTMLDivElement>(null)
  const zoom      = useRef(1)
  const panX      = useRef(0)
  const panY      = useRef(0)
  const dragStart = useRef<{ x: number; y: number } | null>(null)

  function applyTransform() {
    if (vpRef.current)
      vpRef.current.style.transform = `translate(${panX.current}px,${panY.current}px) scale(${zoom.current})`
  }

  function fitImage(natW: number, natH: number) {
    if (!viewRef.current) return
    const { width: vw, height: vh } = viewRef.current.getBoundingClientRect()
    zoom.current = Math.min(vw / natW, vh / natH, 1)
    panX.current = Math.round((vw - natW * zoom.current) / 2)
    panY.current = Math.round((vh - natH * zoom.current) / 2)
    applyTransform()
  }

  useEffect(() => {
    const el = viewRef.current
    if (!el) return
    function onWheel(e: WheelEvent) {
      e.preventDefault()
      const rect = el!.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      const nz = Math.max(0.05, Math.min(20, zoom.current * factor))
      panX.current = mx - (mx - panX.current) * (nz / zoom.current)
      panY.current = my - (my - panY.current) * (nz / zoom.current)
      zoom.current = nz
      applyTransform()
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  function onMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return
    dragStart.current = { x: e.clientX - panX.current, y: e.clientY - panY.current }
  }
  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!dragStart.current) return
      panX.current = e.clientX - dragStart.current.x
      panY.current = e.clientY - dragStart.current.y
      applyTransform()
    }
    function onUp() { dragStart.current = null }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [])

  // Load model list on mount
  useEffect(() => {
    fetch('/api/infer/models')
      .then(r => r.json())
      .then((ms: Model[]) => {
        setModels(ms)
        if (ms.length) setSelected(ms[0].path)
      })
      .catch(() => setStatus('Failed to load models'))
  }, [])

  function setFile(f: File) {
    fileObject.current = f
    setFileName(f.name)
    // preview upload before inference
    const url = URL.createObjectURL(f)
    setOverlayB64(null)
    setDmB64(null)
    setCount(null)
    setDmCount(null)
    // Show preview via object URL temporarily
    const img = new Image()
    img.onload = () => fitImage(img.naturalWidth, img.naturalHeight)
    img.src = url
    // show in viewport
    if (vpRef.current) {
      const el = vpRef.current.querySelector('img') as HTMLImageElement | null
      if (el) { el.src = url; el.onload = () => fitImage(el.naturalWidth, el.naturalHeight) }
    }
  }

  async function run() {
    if (!fileObject.current) { setStatus('Upload an image first.'); return }
    if (!selectedModel) { setStatus('Select a model first.'); return }
    setRunning(true)
    setStatus('Running inference…')
    try {
      const fd = new FormData()
      fd.append('image', fileObject.current)
      fd.append('model', selectedModel)
      fd.append('threshold', String(threshold))
      const r = await fetch('/api/infer/run', { method: 'POST', body: fd })
      const j = await r.json()
      if (!j.ok) throw new Error(j.detail || j.error || 'Unknown error')
      setOverlayB64(j.overlay)
      setDmB64(j.density_map ?? null)
      setCount(j.count)
      setDmCount(j.dm_count ?? null)
      setRunId(id => id + 1)
      setTab('overlay')
      setStatus('')
    } catch (e: any) {
      setStatus('Error: ' + e.message)
    } finally {
      setRunning(false)
    }
  }

  const currentB64 = tab === 'density' ? dmB64 : overlayB64

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 bg-[#111] border-r border-[#2a2a2a] flex flex-col gap-3 p-4 overflow-y-auto">
        <span className="text-[10px] text-gray-600 uppercase tracking-widest">Inference</span>

        {/* Model */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Model</p>
          {models.length === 0 ? (
            <p className="text-xs text-gray-600">No .pth files found in runs/</p>
          ) : (
            <select
              value={selectedModel}
              onChange={e => setSelected(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-[#333] rounded px-2 py-1.5 text-xs text-gray-200"
            >
              {models.map(m => (
                <option key={m.path} value={m.path}>{m.label}</option>
              ))}
            </select>
          )}
        </div>

        {/* Image upload */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Image</p>
          <div
            ref={dropRef}
            className="border-2 border-dashed border-[#333] rounded p-4 text-center text-xs text-gray-600 cursor-pointer hover:border-[#555] transition-colors"
            onClick={() => fileRef.current?.click()}
            onDragOver={e => { e.preventDefault(); dropRef.current?.classList.add('border-blue-500') }}
            onDragLeave={() => dropRef.current?.classList.remove('border-blue-500')}
            onDrop={e => {
              e.preventDefault()
              dropRef.current?.classList.remove('border-blue-500')
              const f = e.dataTransfer.files[0]
              if (f?.type.startsWith('image/')) setFile(f)
            }}
          >
            {fileName ?? 'Drop image or click to browse'}
          </div>
          <input ref={fileRef} type="file" accept="image/*" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) setFile(f) }} />
        </div>

        {/* Threshold */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Threshold</p>
          <div className="flex items-center gap-2">
            <input
              type="range" min={0} max={1} step={0.005} value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="flex-1 accent-blue-500"
            />
            <input
              type="number" min={0} max={1} step={0.005} value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="w-16 bg-[#1e1e1e] border border-[#333] rounded px-2 py-1 text-xs text-gray-200"
            />
          </div>
        </div>

        {/* Status */}
        {status && <p className="text-xs text-amber-400">{status}</p>}

        {/* Results */}
        {count !== null && (
          <div className="text-xs text-gray-400 leading-relaxed">
            <p>Point count: <span className="text-white">{count}</span></p>
            {dmCount !== null && <p>DM count: <span className="text-white">{dmCount}</span></p>}
          </div>
        )}

        {/* Run button */}
        <button
          onClick={run}
          disabled={running || !selectedModel}
          className="mt-auto px-3 py-2 rounded border border-green-700 text-green-400 text-xs hover:bg-green-900/30 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {running ? 'Running…' : 'Run Inference'}
        </button>
      </div>

      {/* Main view */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Tab bar */}
        {(overlayB64 || dmB64) && (
          <div className="flex gap-1 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0">
            <button
              onClick={() => setTab('overlay')}
              className={`px-3 py-1 rounded text-xs border transition-colors ${
                tab === 'overlay'
                  ? 'border-blue-500 bg-blue-900/40 text-blue-300'
                  : 'border-[#333] text-gray-500 hover:text-gray-300'
              }`}
            >
              Overlay
            </button>
            {dmB64 && (
              <button
                onClick={() => setTab('density')}
                className={`px-3 py-1 rounded text-xs border transition-colors ${
                  tab === 'density'
                    ? 'border-blue-500 bg-blue-900/40 text-blue-300'
                    : 'border-[#333] text-gray-500 hover:text-gray-300'
                }`}
              >
                Density Map
              </button>
            )}
          </div>
        )}

        {/* Image viewport */}
        <div
          ref={viewRef}
          className="flex-1 relative overflow-hidden bg-[#0d0d0d] cursor-grab active:cursor-grabbing"
          onMouseDown={onMouseDown}
        >
          {currentB64 ? (
            <div
              ref={vpRef}
              className="absolute top-0 left-0 origin-top-left inline-block"
            >
              <img
                key={`${tab}-${runId}`}
                src={`data:image/jpeg;base64,${currentB64}`}
                alt=""
                className="block max-w-none select-none"
                draggable={false}
                onLoad={e => {
                  const img = e.currentTarget
                  fitImage(img.naturalWidth, img.naturalHeight)
                }}
              />
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-700">
              {fileName
                ? 'Press "Run Inference" to see results'
                : 'Upload an image and select a model'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
