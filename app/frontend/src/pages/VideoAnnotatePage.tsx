import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import type { Split } from '../api/types'
const SPLITS: Split[] = ['train', 'val', 'test']
const STEP_SMALL = 1
const STEP_LARGE = 10

interface Transform {
  scale: number
  offsetX: number
  offsetY: number
}

function computeTransform(media: HTMLVideoElement, container: HTMLDivElement): Transform | null {
  if (!media.videoWidth || !media.videoHeight) return null
  const rect  = media.getBoundingClientRect()
  const cRect = container.getBoundingClientRect()
  if (!rect.width || !rect.height) return null
  return {
    scale:   rect.width / media.videoWidth,
    offsetX: rect.left - cRect.left,
    offsetY: rect.top  - cRect.top,
  }
}

export default function VideoAnnotatePage() {
  const [split, setSplit] = useState<Split>('train')
  const [selectedVideoId, setSelectedVideoId] = useState<number | null>(null)
  const [frameNumber, setFrameNumber] = useState(0)
  const [frameInput, setFrameInput] = useState('0')
  const [points, setPoints] = useState<[number, number][]>([])
  const [bgSub, setBgSub] = useState(false)
  const [imgLoaded, setImgLoaded] = useState(false)
  const [transform, setTransform] = useState<Transform | null>(null)
  const [lastCommit, setLastCommit] = useState<string | null>(null)
  const [isPlaying, setIsPlaying]   = useState(false)
  const videoOverlayRef = useRef<HTMLVideoElement>(null)  // unused, kept for TS
  const playEndTimeRef  = useRef<number>(0)

  const mediaRef     = useRef<HTMLVideoElement>(null)   // the seekable still player
  const containerRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef<number | null>(null)

  const { data: videos = [] } = useQuery({
    queryKey: ['videos'],
    queryFn: api.listVideos,
  })

  const selectedVideo = videos.find(v => v.id === selectedVideoId) ?? null
  const totalFrames = selectedVideo?.total_frames ?? 0

  // Reset when video changes
  useEffect(() => {
    setFrameNumber(0)
    setFrameInput('0')
    setPoints([])
    setImgLoaded(false)
    setLastCommit(null)
  }, [selectedVideoId])

  // Reset points when frame changes
  useEffect(() => {
    setPoints([])
    setImgLoaded(false)
    setLastCommit(null)
  }, [frameNumber])

  const updateTransform = useCallback(() => {
    if (mediaRef.current && containerRef.current && imgLoaded) {
      setTransform(computeTransform(mediaRef.current, containerRef.current))
    }
  }, [imgLoaded])

  useEffect(() => {
    window.addEventListener('resize', updateTransform)
    return () => window.removeEventListener('resize', updateTransform)
  }, [updateTransform])
  useEffect(() => { updateTransform() }, [updateTransform])

  // Seek the native video element to the current frame (skip when playing clip)
  useEffect(() => {
    const el = mediaRef.current
    if (!el || !selectedVideo?.fps || isPlaying) return
    setImgLoaded(false)
    el.currentTime = frameNumber / selectedVideo.fps
  }, [frameNumber, selectedVideoId, selectedVideo?.fps, isPlaying])

  const streamUrl  = selectedVideoId !== null ? api.videoStreamUrl(selectedVideoId) : null
  const bgSubUrl   = (selectedVideoId !== null && bgSub)
    ? api.videoFrameUrl(selectedVideoId, frameNumber, 'bg_sub')
    : null

  const commitMutation = useMutation({
    mutationFn: () =>
      api.commitVideoFrame(selectedVideoId!, frameNumber, split, points),
    onSuccess: (data) => {
      setLastCommit(`Saved: ${data.frame}  (motion: ${data.motion ?? 'n/a'})  — ${data.points} pts`)
    },
  })

  // --- navigation helpers ---
  const clampedFrame = (n: number) => Math.max(0, Math.min(n, totalFrames - 1))

  const go = (delta: number) => {
    const n = clampedFrame(frameNumber + delta)
    setFrameNumber(n)
    setFrameInput(String(n))
  }

  const goToInput = () => {
    const n = clampedFrame(parseInt(frameInput, 10) || 0)
    setFrameNumber(n)
    setFrameInput(String(n))
  }

  // --- 1-second playback via native seek+play ---
  const playClip = useCallback(() => {
    const el  = mediaRef.current
    const fps = selectedVideo?.fps
    if (!el || !fps || isPlaying) return
    const endTime   = (frameNumber + 1) / fps
    const startTime = Math.max(0, endTime - 1)
    playEndTimeRef.current = endTime
    el.currentTime = startTime
    el.play()
    setIsPlaying(true)
  }, [selectedVideo?.fps, frameNumber, isPlaying])

  const stopPlayback = useCallback(() => {
    const el  = mediaRef.current
    const fps = selectedVideo?.fps ?? 30
    if (!el) return
    el.pause()
    el.currentTime = frameNumber / fps
    setIsPlaying(false)
  }, [frameNumber, selectedVideo?.fps])

  const handleTimeUpdate = useCallback(() => {
    const el = mediaRef.current
    if (!el || !isPlaying) return
    if (el.currentTime >= playEndTimeRef.current) stopPlayback()
  }, [isPlaying, stopPlayback])

  // --- annotation interaction ---
  const screenToImage = (sx: number, sy: number): [number, number] | null => {
    if (!transform || !mediaRef.current) return null
    const ix = Math.round((sx - transform.offsetX) / transform.scale)
    const iy = Math.round((sy - transform.offsetY) / transform.scale)
    if (ix < 0 || iy < 0 || ix > mediaRef.current.videoWidth || iy > mediaRef.current.videoHeight)
      return null
    return [ix, iy]
  }

  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (draggingRef.current !== null) return
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const pt = screenToImage(e.clientX - rect.left, e.clientY - rect.top)
    if (!pt) return
    setPoints(prev => [...prev, pt])
  }

  const handleDotMouseDown = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation()
    if (e.button === 2) {
      setPoints(prev => prev.filter((_, i) => i !== idx))
      return
    }
    draggingRef.current = idx
    const onMove = (me: MouseEvent) => {
      if (draggingRef.current === null || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const pt = screenToImage(me.clientX - rect.left, me.clientY - rect.top)
      if (!pt) return
      setPoints(prev => prev.map((p, i) => i === draggingRef.current! ? pt : p))
    }
    const onUp = () => {
      draggingRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft')              go(-STEP_SMALL)
    if (e.key === 'ArrowRight')             go(+STEP_SMALL)
    if (e.key === 'ArrowUp')                go(+STEP_LARGE)
    if (e.key === 'ArrowDown')              go(-STEP_LARGE)
    if (e.key === 'r' || e.key === 'R')    { e.preventDefault(); playClip() }
    if (e.key === 'Escape' && isPlaying)    stopPlayback()
  }

  return (
    <div className="flex flex-col h-full overflow-hidden" onKeyDown={handleKeyDown} tabIndex={0}>
      {/* ---- toolbar ---- */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0 flex-wrap">

        {/* video picker */}
        <select
          className="bg-[#222] border border-[#333] text-gray-200 text-xs px-2 py-1 rounded"
          value={selectedVideoId ?? ''}
          onChange={e => setSelectedVideoId(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">— pick video —</option>
          {videos.map(v => (
            <option key={v.id} value={v.id}>
              {v.original_name} ({v.total_frames} frames @ {v.fps.toFixed(2)} fps)
            </option>
          ))}
        </select>

        {/* split */}
        <div className="flex gap-1">
          {SPLITS.map(s => (
            <button
              key={s}
              onClick={() => setSplit(s)}
              className={`px-2 py-1 rounded text-xs border ${
                split === s
                  ? 'bg-blue-700 border-blue-600 text-white'
                  : 'border-[#333] text-gray-400 hover:text-gray-200'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {/* frame navigation */}
        <div className="flex items-center gap-1 ml-2">
          <button onClick={() => go(-STEP_LARGE)} className="nav-btn" title="-10">«</button>
          <button onClick={() => go(-STEP_SMALL)} className="nav-btn" title="-1">‹</button>
          <input
            className="w-20 bg-[#222] border border-[#333] text-gray-200 text-xs px-2 py-1 rounded text-center"
            value={frameInput}
            onChange={e => setFrameInput(e.target.value)}
            onBlur={goToInput}
            onKeyDown={e => { if (e.key === 'Enter') goToInput() }}
          />
          <span className="text-gray-500 text-xs">/ {totalFrames > 0 ? totalFrames - 1 : '?'}</span>
          <button onClick={() => go(+STEP_SMALL)} className="nav-btn" title="+1">›</button>
          <button onClick={() => go(+STEP_LARGE)} className="nav-btn" title="+10">»</button>
        </div>

        {/* bg subtract toggle */}
        <button
          onClick={() => setBgSub(v => !v)}
          title="Subtract 5-second window average (1-in-10 frames) to reveal motion"
          className={`px-3 py-1 rounded text-xs border transition-colors ${
            bgSub
              ? 'bg-amber-700 border-amber-600 text-white'
              : 'border-[#333] text-gray-400 hover:text-white'
          }`}
        >
          BG−
        </button>

        {/* point count */}
        <span className="text-gray-400 text-xs ml-2">{points.length} pts</span>

        {/* play last second */}
        <button
          onClick={playClip}
          disabled={!selectedVideoId || isPlaying}
          title="Play last 1 second up to this frame (R)"
          className="px-3 py-1 rounded text-xs border border-[#333] text-gray-400 hover:text-white disabled:opacity-40 transition-colors"
        >
          {isPlaying ? '⏸ playing' : '▶ 1s [R]'}
        </button>

        {/* actions */}
        <button
          onClick={() => setPoints([])}
          className="px-3 py-1 rounded text-xs border border-[#333] text-gray-400 hover:text-white ml-auto"
        >
          Clear
        </button>
        <button
          onClick={() => commitMutation.mutate()}
          disabled={!selectedVideoId || commitMutation.isPending}
          className="px-3 py-1 rounded text-xs bg-green-700 hover:bg-green-600 text-white disabled:opacity-40"
        >
          {commitMutation.isPending ? 'Saving…' : 'Save'}
        </button>
      </div>

      {/* ---- status bar ---- */}
      {(lastCommit || commitMutation.isError) && (
        <div className={`px-4 py-1 text-xs flex-shrink-0 ${
          commitMutation.isError ? 'bg-red-900/40 text-red-300' : 'bg-green-900/30 text-green-300'
        }`}>
          {commitMutation.isError
            ? String((commitMutation.error as Error).message)
            : lastCommit}
        </div>
      )}

      {/* ---- canvas ---- */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto flex items-start justify-center bg-[#111] relative select-none"
        onClick={handleContainerClick}
        onContextMenu={e => e.preventDefault()}
      >
        {!selectedVideoId ? (
          <p className="text-gray-600 text-sm mt-20">Select a video above</p>
        ) : (
          <>
            {/* Native video element — browser H.264 decoder, seeked to current frame */}
            <video
              ref={mediaRef}
              key={streamUrl!}
              src={streamUrl!}
              muted
              preload="auto"
              className="max-h-full max-w-full object-contain"
              style={{ visibility: isPlaying ? 'visible' : undefined }}
              onLoadedMetadata={() => {
                if (mediaRef.current && selectedVideo?.fps)
                  mediaRef.current.currentTime = frameNumber / selectedVideo.fps
              }}
              onSeeked={() => { setImgLoaded(true); updateTransform() }}
              onTimeUpdate={handleTimeUpdate}
              onEnded={stopPlayback}
              draggable={false}
            />

            {/* BG-sub processed overlay — JPEG from server, shown on top when enabled */}
            {bgSub && bgSubUrl && imgLoaded && transform && (
              <img
                key={bgSubUrl}
                src={bgSubUrl}
                alt="bg-sub"
                className="absolute object-contain pointer-events-none"
                style={{
                  left:   transform.offsetX,
                  top:    transform.offsetY,
                  width:  mediaRef.current?.getBoundingClientRect().width  ?? '100%',
                  height: mediaRef.current?.getBoundingClientRect().height ?? '100%',
                  zIndex: 4,
                }}
                draggable={false}
              />
            )}

            {/* Clip playback uses the same <video> element above — no second element needed */}

            {/* SVG dot overlay — always on top */}
            {imgLoaded && transform && (
              <svg
                className="absolute inset-0 pointer-events-none"
                style={{ left: 0, top: 0, width: '100%', height: '100%', zIndex: 10 }}
              >
                {points.map(([x, y], i) => {
                  const sx = transform.offsetX + x * transform.scale
                  const sy = transform.offsetY + y * transform.scale
                  return (
                    <g
                      key={i}
                      style={{ pointerEvents: 'all', cursor: 'grab' }}
                      onMouseDown={e => handleDotMouseDown(e, i)}
                    >
                      <circle cx={sx} cy={sy} r={7} fill="rgba(0,200,80,0.25)" stroke="none" />
                      <circle cx={sx} cy={sy} r={4} fill="#00dc50" stroke="#004a1a" strokeWidth={1} />
                    </g>
                  )
                })}
              </svg>
            )}
          </>
        )}
      </div>

      {/* ---- help ---- */}
      <div className="px-4 py-1 border-t border-[#2a2a2a] text-[10px] text-gray-600 flex gap-4 flex-shrink-0">
        <span>Click → add point</span>
        <span>Right-click dot → remove</span>
        <span>Drag dot → move</span>
        <span>← → ±1 frame  |  ↑ ↓ ±10 frames</span>
        <span><strong>R</strong> → play last 1 s (dots stay visible)</span>
        <span><strong>BG−</strong> foreground overlay</span>
        <span>Motion-cue frame (−10) saved as <code>_prev10.jpg</code></span>
      </div>
    </div>
  )
}
