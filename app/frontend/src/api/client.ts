import type {
  DBFile, DBCrop, CropsPage, AnnotationData,
  ScanResult, AssignResult, CropResult, ExtractResult, Split,
} from './types'

export interface UploadResult {
  saved: { original_name: string; saved_as: string }[]
  duplicates: { original_name: string; duplicate_of: string; saved_as: string }[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init)
  if (!r.ok) throw new Error(await r.text())
  return r.json() as Promise<T>
}

const json = (body: unknown) => ({
  method:  'POST',
  headers: { 'Content-Type': 'application/json' },
  body:    JSON.stringify(body),
})

export const api = {
  scan: () =>
    req<ScanResult>('/scan', { method: 'POST' }),

  inputStatus: () =>
    req<{ on_disk: number; ingested: number }>('/input-status'),

  upload: (files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return req<UploadResult>('/upload', { method: 'POST', body: form })
  },

  listFiles: (params?: { status?: string; split?: string; type?: string }) => {
    const qs = new URLSearchParams()
    if (params?.status) qs.set('status', params.status)
    if (params?.split)  qs.set('split',  params.split)
    if (params?.type)   qs.set('type',   params.type)
    const q = qs.toString()
    return req<DBFile[]>(`/files${q ? `?${q}` : ''}`)
  },

  assignFiles: (fileIds: number[], split: Split) =>
    req<AssignResult>('/files/assign', json({ fileIds, split })),

  unassignFiles: (fileIds: number[]) =>
    req<{ unassigned: number; errors: string[] }>('/files/unassign', json({ fileIds })),

  extractFrames: (fileId: number, split: Split, fps = 1) =>
    req<ExtractResult>('/files/extract-frames', json({ fileId, split, fps })),

  generateCrops: (fileIds: number[], overlap = 0.2, cropSize = 640) =>
    req<CropResult>('/crops', json({ fileIds, overlap, cropSize })),

  listCrops: (split?: Split, offset = 0, limit = 100) => {
    const qs = new URLSearchParams()
    if (split) qs.set('split', split)
    qs.set('offset', String(offset))
    qs.set('limit', String(limit))
    return req<CropsPage>(`/crops?${qs}`)
  },

  fileImgUrl:   (id: number) => `/api/images/files/${id}`,
  cropImgUrl:   (uuid: string) => `/api/images/crops/uuid/${uuid}`,
  fileThumbUrl: (id: number, size = 200) => `/api/thumbnails/files/${id}?size=${size}`,
  cropThumbUrl: (uuid: string, size = 200) => `/api/thumbnails/crops/uuid/${uuid}?size=${size}`,

  labelQueue: (split: Split) =>
    req<DBFile[]>(`/label/queue?split=${split}`),

  cropLabelQueue: (split: Split) =>
    req<DBCrop[]>(`/label/crop-queue?split=${split}`),

  labeledCrops: (split: Split) =>
    req<DBCrop[]>(`/label/labeled-crops?split=${split}`),

  getAnnotation: (fileId: number) =>
    req<AnnotationData>(`/label/${fileId}`),

  getCropAnnotation: (cropId: number) =>
    req<AnnotationData>(`/label/crops/${cropId}`),

  saveAnnotation: (fileId: number, points: [number, number][]) =>
    req<{ ok: boolean; count: number }>(`/label/${fileId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points }),
    }),

  saveCropAnnotation: (cropId: number, points: [number, number][]) =>
    req<{ ok: boolean; count: number }>(`/label/crops/${cropId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points }),
    }),

  commitAnnotation: (fileId: number) =>
    req<{ ok: boolean; points: number; path: string }>(`/label/${fileId}/commit`, { method: 'POST' }),

  commitCropAnnotation: (cropId: number) =>
    req<{ ok: boolean; points: number; path: string }>(`/label/crops/${cropId}/commit`, { method: 'POST' }),
}
