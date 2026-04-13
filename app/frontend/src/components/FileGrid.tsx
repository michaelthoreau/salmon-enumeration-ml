import type { DBFile } from '../api/types'
import { api } from '../api/client'

interface Props {
  files: DBFile[]
  selected: Set<number>
  onToggle: (id: number, shift: boolean) => void
  onExtractFrames?: (file: DBFile) => void
}

export default function FileGrid({ files, selected, onToggle, onExtractFrames }: Props) {
  if (!files.length)
    return <div className="p-12 text-center text-gray-600">No files</div>

  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(148px,1fr))] gap-3 p-4 overflow-y-auto h-full content-start">
      {files.map((f) => {
        const sel = selected.has(f.id)
        return (
          <div
            key={f.id}
            onClick={(e) => onToggle(f.id, e.shiftKey)}
            className={`relative rounded cursor-pointer select-none transition-all border ${
              sel
                ? 'border-blue-500 ring-1 ring-blue-500/50'
                : 'border-[#2a2a2a] hover:border-[#444]'
            }`}
          >
            {/* Thumbnail */}
            {f.type === 'image' ? (
              <img
                src={api.fileThumbUrl(f.id, 200)}
                alt={f.original_name}
                className="w-full aspect-square object-cover rounded-t bg-[#111]"
                loading="lazy"
              />
            ) : (
              <div className="w-full aspect-square bg-[#111] rounded-t flex flex-col items-center justify-center gap-1 text-gray-600">
                <span className="text-2xl">🎬</span>
                <span className="text-[10px]">video</span>
              </div>
            )}

            {/* Info */}
            <div className="px-2 py-1.5 bg-[#141414] rounded-b">
              <p className="truncate text-[11px] text-gray-300" title={f.original_name}>
                {f.original_name}
              </p>
              <p className="text-[10px] text-gray-700">{f.status}</p>
            </div>

            {/* Cropped badge */}
            {f.status === 'cropped' && (
              <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[9px] bg-green-900/80 text-green-300 border border-green-700/50">
                cropped
              </div>
            )}

            {/* Selection badge */}
            {sel && (
              <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center text-white text-[10px] font-bold shadow">
                ✓
              </div>
            )}

            {/* Video extract button */}
            {f.type === 'video' && onExtractFrames && (
              <button
                onClick={(e) => { e.stopPropagation(); onExtractFrames(f) }}
                className="absolute bottom-9 left-1.5 right-1.5 py-0.5 text-[10px] bg-[#225]/80 hover:bg-[#336] text-blue-200 rounded"
                title="Extract frames"
              >
                Extract frames
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}
