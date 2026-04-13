import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { DBCrop, AnnotationData, Split } from '../api/types'

const SPLITS: Split[] = ['train', 'val', 'test']

interface Transform { scale: number; offsetX: number; offsetY: number }

function computeTransform(img: HTMLImageElement, container: HTMLDivElement): Transform | null {
  if (!img.naturalWidth || !img.naturalHeight) return null
  const { width: cw, height: ch } = container.getBoundingClientRect()
  if (!cw || !ch) return null
  const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight)
  return {
    scale,
    offsetX: (cw - img.naturalWidth * scale) / 2,
    offsetY: (ch - img.naturalHeight * scale) / 2,
  }
}

function CropViewer({ crop, onClose }: { crop: DBCrop; onClose: () => void }) {
  const [imgLoaded, setImgLoaded] = useState(false)
  const [transform, setTransform] = useState<Transform | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const { data: ann } = useQuery({
    queryKey: ['crop-annotation', crop.id],
    queryFn: () => api.getCropAnnotation(crop.id),
  })
  const points: [number, number][] = ann?.points ?? []

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center"
      onClick={onClose}
    >
      <div
        className="relative bg-[#111] rounded-lg overflow-hidden shadow-2xl flex flex-col"
        style={{ width: 'min(90vw, 900px)', height: 'min(90vh, 900px)' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0">
          <span className="text-xs text-gray-400">
            {points.length} fish &middot; {crop.w}&times;{crop.h} &middot; offset ({crop.x},{crop.y})
          </span>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-200 text-xs px-2 py-1 rounded hover:bg-[#222]">
            Close (Esc)
          </button>
        </div>

        {/* Canvas — same pattern as LabelPage */}
        <div ref={containerRef} className="flex-1 bg-[#0d0d0d] relative flex items-center justify-center">
          <img
            key={crop.id}
            ref={imgRef}
            src={api.cropImgUrl(crop.uuid)}
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
            <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: 'none' }}>
              {transform && points.map(([px, py], i) => {
                const sx = transform.offsetX + px * transform.scale
                const sy = transform.offsetY + py * transform.scale
                return (
                  <g key={i}>
                    <circle cx={sx} cy={sy} r={9} fill="rgba(239,68,68,0.25)" stroke="rgb(239,68,68)" strokeWidth={1.5} />
                    <circle cx={sx} cy={sy} r={2.5} fill="rgb(239,68,68)" />
                    <text x={sx + 11} y={sy + 4} fontSize={10} fill="rgb(248,113,113)" style={{ userSelect: 'none' }}>{i + 1}</text>
                  </g>
                )
              })}
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}

export default function LabeledPage() {
  const [split, setSplit] = useState<Split>('train')
  const [viewing, setViewing] = useState<DBCrop | null>(null)

  const { data: crops = [], isFetching } = useQuery({
    queryKey: ['labeled-crops', split],
    queryFn: () => api.labeledCrops(split),
  })

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0">
        <span className="text-xs text-gray-500 uppercase tracking-widest">Labeled Images</span>
        <div className="flex gap-1">
          {SPLITS.map((s) => (
            <button
              key={s}
              onClick={() => { setSplit(s); setViewing(null) }}
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
        <span className="text-xs text-gray-600 ml-auto">
          {isFetching ? 'loading...' : `${crops.length} labeled`}
        </span>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-auto">
        {crops.length === 0 ? (
          <div className="p-12 text-center text-gray-600 text-xs">
            No labeled crops in {split} yet. Commit annotations from the Label page.
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-3 p-4 content-start">
            {crops.map((c) => (
              <CropCard key={c.id} crop={c} onClick={() => setViewing(c)} />
            ))}
          </div>
        )}
      </div>

      {viewing && <CropViewer crop={viewing} onClose={() => setViewing(null)} />}
    </div>
  )
}

function CropCard({ crop, onClick }: { crop: DBCrop; onClick: () => void }) {
  const { data: ann } = useQuery({
    queryKey: ['crop-annotation', crop.id],
    queryFn: () => api.getCropAnnotation(crop.id),
  })
  const count = ann?.points.length ?? '...'

  return (
    <div
      className="relative rounded border border-[#2a2a2a] overflow-hidden cursor-pointer hover:border-[#555] transition-colors select-none"
      onClick={onClick}
    >
      <img
        src={api.cropThumbUrl(crop.uuid, 200)}
        alt=""
        className="w-full aspect-square object-cover bg-[#111]"
        loading="lazy"
      />
      <div className="px-2 py-1 bg-[#141414] flex items-center justify-between">
        <p className="text-[10px] text-gray-600">{crop.w}&times;{crop.h}</p>
        <span className="text-[10px] text-red-400 font-semibold">{count} fish</span>
      </div>
      <div className="absolute top-1.5 left-1.5 bg-green-700/80 text-green-200 text-[9px] px-1.5 py-0.5 rounded font-semibold">
        labeled
      </div>
    </div>
  )
}
