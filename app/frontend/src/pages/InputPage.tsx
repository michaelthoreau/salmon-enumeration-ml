import { useRef, useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, type UploadResult } from '../api/client'
import type { DBFile, Split } from '../api/types'
import FileGrid from '../components/FileGrid'

export default function InputPage() {
  const qc = useQueryClient()
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [lastClicked, setLastClicked] = useState<number | null>(null)
  const [targetSplit, setTargetSplit] = useState<Split>('train')
  const [extractFps, setExtractFps] = useState(1)
  const [extractTarget, setExtracting] = useState<DBFile | null>(null)
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { data: files = [], isFetching } = useQuery({
    queryKey: ['files', 'input'],
    queryFn: () => api.listFiles({ status: 'input' }),
  })

  const { data: status } = useQuery({
    queryKey: ['input-status'],
    queryFn: () => api.inputStatus(),
  })

  const scanMut = useMutation({
    mutationFn: api.scan,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['files'] })
      qc.invalidateQueries({ queryKey: ['input-status'] })
    },
  })

  const uploadMut = useMutation({
    mutationFn: (files: File[]) => api.upload(files),
    onSuccess: async (result) => {
      setUploadResult(result)
      await api.scan()
      qc.invalidateQueries({ queryKey: ['input-status'] })
      qc.invalidateQueries({ queryKey: ['files'] })
    },
  })

  const assignMut = useMutation({
    mutationFn: ({ ids, split }: { ids: number[]; split: Split }) =>
      api.assignFiles(ids, split),
    onSuccess: () => {
      setSelected(new Set())
      qc.invalidateQueries({ queryKey: ['files'] })
    },
  })

  const extractMut = useMutation({
    mutationFn: ({ file, split, fps }: { file: DBFile; split: Split; fps: number }) =>
      api.extractFrames(file.id, split, fps),
    onSuccess: () => {
      setExtracting(null)
      qc.invalidateQueries({ queryKey: ['files'] })
    },
  })

  const toggle = useCallback((id: number, shift: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (shift && lastClicked !== null) {
        const ids = files.map((f) => f.id)
        const a = ids.indexOf(lastClicked)
        const b = ids.indexOf(id)
        const [lo, hi] = [Math.min(a, b), Math.max(a, b)]
        ids.slice(lo, hi + 1).forEach((x) => next.add(x))
      } else {
        next.has(id) ? next.delete(id) : next.add(id)
      }
      return next
    })
    setLastClicked(id)
  }, [files, lastClicked])

  function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return
    setUploadResult(null)
    uploadMut.mutate(Array.from(fileList))
  }

  const selCount = selected.size

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0 flex-wrap">
        {/* Status badge */}
        {status && (
          <span className="text-xs text-gray-500 border border-[#2a2a2a] rounded px-2 py-1">
            {status.on_disk} on disk · {status.ingested} ingested
          </span>
        )}

        <button
          onClick={() => scanMut.mutate()}
          disabled={scanMut.isPending}
          className="px-3 py-1.5 bg-[#222] hover:bg-[#2a2a2a] rounded text-xs border border-[#333] disabled:opacity-50"
        >
          {scanMut.isPending ? 'Scanning…' : 'Scan'}
        </button>

        {scanMut.data && (
          <span className="text-xs text-gray-500">
            +{scanMut.data.added} added, {scanMut.data.skipped} skipped
          </span>
        )}

        {/* Upload */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadMut.isPending}
          className="px-3 py-1.5 bg-[#222] hover:bg-[#2a2a2a] rounded text-xs border border-[#333] disabled:opacity-50"
        >
          {uploadMut.isPending ? 'Uploading…' : 'Upload images'}
        </button>

        <div className="ml-auto flex items-center gap-2">
          {selCount > 0 && (
            <>
              <span className="text-xs text-gray-500">{selCount} selected</span>
              <select
                value={targetSplit}
                onChange={(e) => setTargetSplit(e.target.value as Split)}
                className="bg-[#222] border border-[#333] rounded px-2 py-1 text-xs"
              >
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
              </select>
              <button
                onClick={() => assignMut.mutate({ ids: [...selected], split: targetSplit })}
                disabled={assignMut.isPending}
                className="px-3 py-1.5 bg-blue-700 hover:bg-blue-600 rounded text-xs disabled:opacity-50"
              >
                Assign →
              </button>
            </>
          )}
          <button
            onClick={() => setSelected(new Set())}
            disabled={selCount === 0}
            className="px-3 py-1.5 text-xs rounded border border-[#333] hover:bg-[#222] disabled:opacity-30"
          >
            Deselect all
          </button>
          {isFetching && <span className="text-xs text-gray-600">loading…</span>}
        </div>
      </div>

      {/* Upload result banner */}
      {uploadResult && (
        <div className="flex flex-col gap-1 px-4 py-2 border-b border-[#2a2a2a] bg-[#141414] flex-shrink-0">
          {uploadResult.saved.length > 0 && (
            <p className="text-xs text-green-400">
              ✓ {uploadResult.saved.length} file{uploadResult.saved.length !== 1 ? 's' : ''} uploaded — click Scan to ingest
            </p>
          )}
          {uploadResult.duplicates.length > 0 && (
            <div className="text-xs text-amber-400">
              <p>⚠ {uploadResult.duplicates.length} duplicate{uploadResult.duplicates.length !== 1 ? 's' : ''} detected (saved anyway):</p>
              <ul className="mt-1 ml-3 list-disc text-amber-500">
                {uploadResult.duplicates.map((d, i) => (
                  <li key={i}>{d.original_name} → matches {d.duplicate_of}</li>
                ))}
              </ul>
            </div>
          )}
          <button
            onClick={() => setUploadResult(null)}
            className="text-[10px] text-gray-600 hover:text-gray-400 self-start"
          >
            dismiss
          </button>
        </div>
      )}

      {/* Drop zone + grid */}
      <div
        className={`flex-1 overflow-auto relative ${dragOver ? 'ring-2 ring-inset ring-blue-500' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer.files)
        }}
      >
        {dragOver && (
          <div className="absolute inset-0 z-10 bg-blue-900/20 flex items-center justify-center pointer-events-none">
            <p className="text-blue-300 text-sm">Drop to upload</p>
          </div>
        )}
        {!isFetching && files.length === 0 ? (
          <div className="p-12 text-center text-gray-600 text-xs flex flex-col items-center gap-3">
            <p>No ingested files. Upload images or drop them here, then click Scan.</p>
          </div>
        ) : (
          <FileGrid
            files={files}
            selected={selected}
            onToggle={toggle}
            onExtractFrames={setExtracting}
          />
        )}
      </div>

      {/* Extract frames dialog */}
      {extractTarget && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-[#333] rounded-lg p-6 w-80 flex flex-col gap-4">
            <p className="text-sm font-medium">Extract frames</p>
            <p className="text-xs text-gray-500 truncate">{extractTarget.original_name}</p>
            <label className="flex flex-col gap-1 text-xs text-gray-400">
              FPS
              <input
                type="number"
                min={0.1}
                max={30}
                step={0.1}
                value={extractFps}
                onChange={(e) => setExtractFps(parseFloat(e.target.value))}
                className="bg-[#111] border border-[#333] rounded px-2 py-1 text-white"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-gray-400">
              Destination split
              <select
                value={targetSplit}
                onChange={(e) => setTargetSplit(e.target.value as Split)}
                className="bg-[#111] border border-[#333] rounded px-2 py-1 text-white"
              >
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
              </select>
            </label>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setExtracting(null)}
                className="px-3 py-1.5 text-xs rounded border border-[#333] hover:bg-[#222]"
              >
                Cancel
              </button>
              <button
                onClick={() => extractMut.mutate({ file: extractTarget, split: targetSplit, fps: extractFps })}
                disabled={extractMut.isPending}
                className="px-3 py-1.5 text-xs rounded bg-blue-700 hover:bg-blue-600 disabled:opacity-50"
              >
                {extractMut.isPending ? 'Extracting…' : 'Extract'}
              </button>
            </div>
            {extractMut.data && (
              <p className="text-xs text-green-400">{extractMut.data.framesExtracted} frames extracted</p>
            )}
            {extractMut.data?.errors.length ? (
              <p className="text-xs text-red-400">{extractMut.data.errors[0]}</p>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}
