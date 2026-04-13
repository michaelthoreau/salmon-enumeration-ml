import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { DBCrop, Split } from '../api/types'

const SPLITS: Split[] = ['train', 'val', 'test']
const PAGE_SIZE = 100

export default function UnlabeledPage() {
  const [activeSplit, setActiveSplit] = useState<Split>('train')
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [page, setPage] = useState(0)

  function toggleCrop(c: DBCrop) {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(c.id) ? next.delete(c.id) : next.add(c.id)
      return next
    })
  }

  const { data, isFetching } = useQuery({
    queryKey: ['crops', activeSplit, page],
    queryFn: () => api.listCrops(activeSplit, page * PAGE_SIZE, PAGE_SIZE),
  })

  const crops = data?.items ?? []
  const total = data?.total ?? 0
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-[#2a2a2a] flex-shrink-0">
        <span className="text-xs text-gray-500 uppercase tracking-widest">Unlabeled crops</span>
        <div className="flex gap-1">
          {SPLITS.map((s) => (
            <button
              key={s}
              onClick={() => { setActiveSplit(s); setSelected(new Set()); setPage(0) }}
              className={`px-2.5 py-1 rounded text-xs border transition-colors ${
                activeSplit === s
                  ? 'border-blue-500 bg-blue-900/40 text-blue-300'
                  : 'border-[#333] text-gray-500 hover:text-gray-300'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-600 ml-auto">
          {isFetching ? 'loading…' : `${total} crops`}
        </span>
        {selected.size > 0 && (
          <button
            onClick={() => setSelected(new Set())}
            className="px-3 py-1.5 text-xs rounded border border-[#333] hover:bg-[#222]"
          >
            Deselect all ({selected.size})
          </button>
        )}
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-auto">
        {total === 0 && !isFetching ? (
          <div className="p-12 text-center text-gray-600 text-xs">
            No crops yet — assign images then generate crops from the {activeSplit} raw page.
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-3 p-4 content-start">
            {crops.map((c) => (
              <div
                key={c.id}
                className={`relative rounded border overflow-hidden cursor-pointer select-none transition-all ${
                  selected.has(c.id)
                    ? 'border-blue-500 ring-1 ring-blue-500/50'
                    : 'border-[#2a2a2a] hover:border-[#444]'
                }`}
                onClick={() => toggleCrop(c)}
              >
                <img
                  src={api.cropThumbUrl(c.uuid, 200)}
                  alt={`crop ${c.id}`}
                  className="w-full aspect-square object-cover bg-[#111]"
                  loading="lazy"
                />
                <div className="px-2 py-1 bg-[#141414]">
                  <p className="text-[10px] text-gray-600">
                    {c.x},{c.y} · {c.w}×{c.h}
                  </p>
                </div>
                {selected.has(c.id) && (
                  <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white text-[10px] font-bold shadow">
                    ✓
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {pageCount > 1 && (
        <div className="flex items-center justify-center gap-2 px-4 py-2 border-t border-[#2a2a2a] flex-shrink-0">
          <button
            disabled={page === 0}
            onClick={() => setPage(0)}
            className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
          >
            «
          </button>
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
          >
            ‹
          </button>
          <span className="text-xs text-gray-500">
            Page {page + 1} of {pageCount}
          </span>
          <button
            disabled={page >= pageCount - 1}
            onClick={() => setPage((p) => p + 1)}
            className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
          >
            ›
          </button>
          <button
            disabled={page >= pageCount - 1}
            onClick={() => setPage(pageCount - 1)}
            className="px-2 py-1 text-xs rounded border border-[#333] disabled:opacity-30 hover:bg-[#222] disabled:cursor-default"
          >
            »
          </button>
        </div>
      )}
    </div>
  )
}
