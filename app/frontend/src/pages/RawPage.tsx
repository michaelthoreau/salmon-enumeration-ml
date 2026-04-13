import { useState, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { Split } from '../api/types'
import FileGrid from '../components/FileGrid'

export default function RawPage() {
  const { split } = useParams<{ split: string }>()
  const qc = useQueryClient()
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [lastClicked, setLastClicked] = useState<number | null>(null)
  const [overlap, setOverlap] = useState(0.2)
  const [cropSize, setCropSize] = useState(640)

  const { data: files = [], isFetching } = useQuery({
    queryKey: ['files', 'raw', split],
    queryFn: () => api.listFiles({ split }),
    enabled: !!split,
  })

  const cropMut = useMutation({
    mutationFn: (ids: number[]) => api.generateCrops(ids, overlap, cropSize),
    onSuccess: () => {
      setSelected(new Set())
      qc.invalidateQueries({ queryKey: ['crops'] })
      qc.invalidateQueries({ queryKey: ['files', 'raw', split] })
    },
  })

  const unassignMut = useMutation({
    mutationFn: (ids: number[]) => api.unassignFiles(ids),
    onSuccess: () => {
      setSelected(new Set())
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

  const selCount = selected.size
  const croppableSelected = [...selected].filter(
    (id) => files.find((f) => f.id === id)?.status !== 'cropped'
  )

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0">
        <span className="text-xs text-gray-500 uppercase tracking-widest">{split} raw</span>
        <span className="text-xs text-gray-600">{files.length} images</span>

        <div className="ml-auto flex items-center gap-2">
          {selCount > 0 && (
            <>
              <span className="text-xs text-gray-500">{selCount} selected</span>
              <button
                onClick={() => unassignMut.mutate([...selected])}
                disabled={unassignMut.isPending}
                className="px-3 py-1.5 bg-[#222] hover:bg-[#2a2a2a] border border-[#444] rounded text-xs disabled:opacity-50"
              >
                {unassignMut.isPending ? 'Unassigning…' : 'Unassign'}
              </button>
              <label className="flex items-center gap-1.5 text-xs text-gray-400">
                size
                <input
                  type="number"
                  min={64}
                  max={2048}
                  step={64}
                  value={cropSize}
                  onChange={(e) => setCropSize(parseInt(e.target.value))}
                  className="w-16 bg-[#222] border border-[#333] rounded px-1.5 py-1 text-white"
                />
              </label>
              <label className="flex items-center gap-1.5 text-xs text-gray-400">
                overlap
                <input
                  type="number"
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={overlap}
                  onChange={(e) => setOverlap(parseFloat(e.target.value))}
                  className="w-16 bg-[#222] border border-[#333] rounded px-1.5 py-1 text-white"
                />
              </label>
              <button
                onClick={() => cropMut.mutate(croppableSelected)}
                disabled={cropMut.isPending || croppableSelected.length === 0}
                className="px-3 py-1.5 bg-green-800 hover:bg-green-700 rounded text-xs disabled:opacity-50"
              >
                {cropMut.isPending ? 'Cropping…' : 'Generate crops'}
              </button>
            </>
          )}
          {cropMut.data && (
            <span className="text-xs text-green-400">
              {cropMut.data.cropsGenerated} crops
            </span>
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

      {/* Grid */}
      <div className="flex-1 overflow-auto">
        <FileGrid files={files} selected={selected} onToggle={toggle} />
      </div>
    </div>
  )
}
