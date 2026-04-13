import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useBlocker } from 'react-router-dom'
import { api } from '../api/client'
import type { DBCrop, Split } from '../api/types'

const SPLITS: Split[] = ['train', 'val', 'test']

interface Transform {
  scale: number
  offsetX: number
  offsetY: number
}

function computeTransform(img: HTMLImageElement, container: HTMLDivElement): Transform | null {
  if (!img.naturalWidth || !img.naturalHeight) return null
  const imgRect = img.getBoundingClientRect()
  const containerRect = container.getBoundingClientRect()
  if (!imgRect.width || !imgRect.height) return null
  return {
    scale: imgRect.width / img.naturalWidth,
    offsetX: imgRect.left - containerRect.left,
    offsetY: imgRect.top - containerRect.top,
  }
}

export default function LabelPage() {
  const [split, setSplit] = useState<Split>('train')
  const [queueIndex, setQueueIndex] = useState(0)
  const [points, setPoints] = useState<[number, number][]>([])
  const [dirty, setDirty] = useState(false)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [transform, setTransform] = useState<Transform | null>(null)

  // Drag state: which point index is being dragged, and its current screen pos
  const draggingRef = useRef<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const qc = useQueryClient()

  // Block in-app navigation (tab switches) when dirty and there's something to lose
  const blocker = useBlocker(dirty && points.length > 0)

  // Confirm or cancel the blocked navigation
  useEffect(() => {
    if (blocker.state === 'blocked') {
      if (window.confirm('You have unsaved changes. Leave without saving?')) {
        blocker.proceed()
      } else {
        blocker.reset()
      }
    }
  }, [blocker.state])

  // Warn on browser close/refresh when dirty
  useEffect(() => {
    function onBeforeUnload(e: BeforeUnloadEvent) {
      if (dirty && points.length > 0) {
        e.preventDefault()
        e.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [dirty])

  const updateTransform = useCallback(() => {
    if (imgRef.current && containerRef.current && imgLoaded) {
      setTransform(computeTransform(imgRef.current, containerRef.current))
    }
  }, [imgLoaded])

  useEffect(() => {
    window.addEventListener('resize', updateTransform)
    return () => window.removeEventListener('resize', updateTransform)
  }, [updateTransform])

  useEffect(() => { updateTransform() }, [updateTransform])

  const { data: queue = [] } = useQuery({
    queryKey: ['crop-label-queue', split],
    queryFn: () => api.cropLabelQueue(split),
  })

  const currentCrop: DBCrop | undefined = queue[queueIndex]

  const { data: ann } = useQuery({
    queryKey: ['crop-annotation', currentCrop?.id],
    queryFn: () => api.getCropAnnotation(currentCrop!.id),
    enabled: !!currentCrop,
  })

  useEffect(() => {
    setPoints(ann ? ann.points : [])
    setDirty(false)
    setImgLoaded(false)
    setTransform(null)
  }, [currentCrop?.id, ann])

  // Warn when navigating away within the SPA (changing split or crop)
  function guardedGoTo(idx: number) {
    if (dirty && !window.confirm('You have unsaved changes. Leave without saving?')) return
    setQueueIndex(Math.max(0, Math.min(queue.length - 1, idx)))
  }

  function guardedSetSplit(s: Split) {
    if (dirty && !window.confirm('You have unsaved changes. Leave without saving?')) return
    setSplit(s)
    setQueueIndex(0)
    setPoints([])
    setDirty(false)
  }

  const saveMut = useMutation({
    mutationFn: () => api.saveCropAnnotation(currentCrop!.id, points),
    onSuccess: () => setDirty(false),
  })

  const commitMut = useMutation({
    mutationFn: async () => {
      await api.saveCropAnnotation(currentCrop!.id, points)
      return api.commitCropAnnotation(currentCrop!.id)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['crop-label-queue', split] })
      setPoints([])
      setDirty(false)
      setImgLoaded(false)
      setTransform(null)
    },
  })

  // Click on empty canvas area -> add point
  function handleCanvasClick(e: React.MouseEvent<SVGSVGElement>) {
    // Ignore if we just finished a drag
    if (draggingRef.current !== null) return
    if (!imgRef.current || !containerRef.current || !imgLoaded) return
    const rect = containerRef.current.getBoundingClientRect()
    const t = computeTransform(imgRef.current, containerRef.current)
    if (!t) return
    const imgX = Math.round((e.clientX - rect.left - t.offsetX) / t.scale)
    const imgY = Math.round((e.clientY - rect.top  - t.offsetY) / t.scale)
    console.log(`[click] imgX=${imgX} imgY=${imgY} scale=${t.scale.toFixed(4)} offsetX=${t.offsetX.toFixed(1)} offsetY=${t.offsetY.toFixed(1)}`)
    if (imgX < 0 || imgY < 0 || imgX > imgRef.current.naturalWidth || imgY > imgRef.current.naturalHeight) return
    setPoints((prev) => [...prev, [imgX, imgY]])
    setDirty(true)
  }

  function removePoint(i: number, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    setPoints((prev) => prev.filter((_, j) => j !== i))
    setDirty(true)
  }

  // Drag handlers on individual point circles
  function handlePointMouseDown(i: number, e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    draggingRef.current = i
  }

  useEffect(() => {
    function onMouseMove(e: MouseEvent) {
      const i = draggingRef.current
      if (i === null || !containerRef.current || !imgRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const t = computeTransform(imgRef.current, containerRef.current)
      if (!t) return
      const imgX = Math.round((e.clientX - rect.left - t.offsetX) / t.scale)
      const imgY = Math.round((e.clientY - rect.top  - t.offsetY) / t.scale)
      console.log(`[drag]  imgX=${imgX} imgY=${imgY}`)
      const clampedX = Math.max(0, Math.min(imgRef.current.naturalWidth,  imgX))
      const clampedY = Math.max(0, Math.min(imgRef.current.naturalHeight, imgY))
      setPoints((prev) => {
        const next = [...prev]
        next[i] = [clampedX, clampedY]
        return next
      })
    }

    function onMouseUp() {
      if (draggingRef.current !== null) {
        draggingRef.current = null
        setDirty(true)
      }
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  const remaining = queue.length

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0 flex-wrap">
        <span className="text-xs text-gray-500 uppercase tracking-widest">Label Queue</span>
        <div className="flex gap-1">
          {SPLITS.map((s) => (
            <button
              key={s}
              onClick={() => guardedSetSplit(s)}
              className={`px-2.5 py-1 rounded text-xs border transition-colors ${
                split === s
                  ? 'border-blue-500 bg-blue-900/40 text-blue-300'
                  : 'border-[#333] text-gray-500 hover:text-gray-300'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {currentCrop && (
          <>
            <div className="flex items-center gap-1 ml-2">
              <button
                onClick={() => guardedGoTo(queueIndex - 1)}
                disabled={queueIndex === 0}
                className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
              >&#8249;</button>
              <span className="text-xs text-gray-500 tabular-nums">
                {queueIndex + 1} / {remaining}
              </span>
              <button
                onClick={() => guardedGoTo(queueIndex + 1)}
                disabled={queueIndex >= remaining - 1}
                className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
              >&#8250;</button>
            </div>

            <span className={`text-xs ${dirty ? 'text-amber-400' : 'text-gray-600'}`}>
              {points.length} fish{dirty ? ' · unsaved' : ''}
            </span>

            <div className="ml-auto flex gap-2">
              <button
                onClick={() => { setPoints((p) => p.slice(0, -1)); setDirty(true) }}
                disabled={points.length === 0}
                className="px-3 py-1.5 text-xs rounded border border-[#333] hover:bg-[#222] disabled:opacity-30"
              >Undo</button>
              <button
                onClick={() => { setPoints([]); setDirty(true) }}
                disabled={points.length === 0}
                className="px-3 py-1.5 text-xs rounded border border-[#333] hover:bg-[#222] disabled:opacity-30"
              >Clear</button>
              <button
                onClick={() => saveMut.mutate()}
                disabled={!dirty || saveMut.isPending}
                className="px-3 py-1.5 text-xs rounded border border-[#444] hover:bg-[#222] disabled:opacity-30"
              >{saveMut.isPending ? 'Saving...' : 'Save'}</button>
              <button
                onClick={() => commitMut.mutate()}
                disabled={commitMut.isPending}
                className="px-3 py-1.5 text-xs rounded border border-green-700 text-green-400 hover:bg-green-900/30 disabled:opacity-30"
              >{commitMut.isPending ? 'Committing...' : `Commit (${points.length})`}</button>
            </div>
          </>
        )}
      </div>

      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar: strip of crops */}
        <div className="w-28 flex-shrink-0 border-r border-[#2a2a2a] overflow-y-auto bg-[#111]">
          {queue.length === 0 ? (
            <p className="p-3 text-[10px] text-gray-600 leading-relaxed">
              No unlabeled crops in {split}.<br />Generate crops from the Raw page first.
            </p>
          ) : (
            queue.map((c, idx) => (
              <button
                key={c.id}
                onClick={() => guardedGoTo(idx)}
                className={`w-full p-1.5 border-b border-[#1a1a1a] transition-colors ${
                  idx === queueIndex
                    ? 'bg-blue-950/60 border-l-2 border-l-blue-500'
                    : 'hover:bg-[#1c1c1c]'
                }`}
              >
                <div className={`rounded overflow-hidden ${idx === queueIndex ? 'ring-2 ring-blue-500' : ''}`}>
                  <img
                    src={api.cropThumbUrl(c.uuid, 120)}
                    alt=""
                    className="w-full aspect-square object-cover bg-[#0d0d0d]"
                  />
                </div>
                <p className={`text-[9px] mt-1 text-center tabular-nums ${idx === queueIndex ? 'text-blue-400 font-semibold' : 'text-gray-600'}`}>{idx + 1}</p>
              </button>
            ))
          )}
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-hidden bg-[#0d0d0d] relative">
          {!currentCrop ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="text-xs text-gray-600">
                {queue.length === 0 ? `No unlabeled crops in ${split}.` : 'Select a crop from the sidebar.'}
              </p>
            </div>
          ) : (
            <div ref={containerRef} className="absolute inset-0 flex items-center justify-center">
              <img
                key={currentCrop.id}
                ref={imgRef}
                src={api.cropImgUrl(currentCrop.uuid)}
                alt=""
                className="max-w-full max-h-full object-contain select-none"
                draggable={false}
                onLoad={() => {
                  setImgLoaded(true)
                  if (imgRef.current && containerRef.current) {
                    setTransform(computeTransform(imgRef.current, containerRef.current))
                  }
                }}
              />
              {imgLoaded && (
                <svg
                  ref={svgRef}
                  className="absolute inset-0 w-full h-full"
                  style={{ cursor: 'crosshair' }}
                  onClick={handleCanvasClick}
                >
                  {transform && points.map(([px, py], i) => {
                    const sx = transform.offsetX + px * transform.scale
                    const sy = transform.offsetY + py * transform.scale
                    return (
                      <g
                        key={i}
                        onContextMenu={(e) => removePoint(i, e)}
                        onMouseDown={(e) => handlePointMouseDown(i, e)}
                        onClick={(e) => e.stopPropagation()}
                      >
                        {/* Large invisible hit area for easy grabbing */}
                        <circle
                          cx={sx} cy={sy} r={14}
                          fill="transparent"
                          style={{ cursor: 'grab' }}
                        />
                        <circle
                          cx={sx} cy={sy} r={9}
                          fill="rgba(239,68,68,0.25)"
                          stroke="rgb(239,68,68)"
                          strokeWidth={1.5}
                          style={{ cursor: 'grab', pointerEvents: 'none' }}
                        />
                        <circle cx={sx} cy={sy} r={2.5} fill="rgb(239,68,68)" style={{ pointerEvents: 'none' }} />
                        <text x={sx + 11} y={sy + 4} fontSize={10} fill="rgb(248,113,113)" style={{ pointerEvents: 'none', userSelect: 'none' }}>{i + 1}</text>
                      </g>
                    )
                  })}
                </svg>
              )}
              {!imgLoaded && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <span className="text-xs text-gray-600">Loading...</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Help */}
      <div className="px-4 py-1.5 border-t border-[#2a2a2a] flex-shrink-0 text-[10px] text-gray-700">
        Click to add a fish point · Drag to move · Right-click to remove · Save to keep progress · Commit to export and finalize
      </div>
    </div>
  )
}
