export type Split = 'train' | 'val' | 'test'

export interface DBFile {
  id:            number
  uuid:          string
  type:          'image' | 'video'
  original_name: string
  sha256:        string
  status:        'input' | 'assigned' | 'extracted' | 'labeled' | 'cropped'
  split:         Split | null
  rel_path:      string
  created_at:    string
}

export interface DBCrop {
  id:          number
  uuid:        string
  source_uuid: string
  split:       Split
  rel_path:    string
  x:           number
  y:           number
  w:           number
  h:           number
  created_at:  string
}

export interface CropsPage     { items: DBCrop[]; total: number; offset: number; limit: number }
export interface AnnotationData { file_id?: number; crop_id?: number; points: [number, number][] }
export interface ScanResult    { added: number; skipped: number }
export interface AssignResult  { assigned: number; errors: string[] }
export interface CropResult    { cropsGenerated: number; errors: string[] }
export interface ExtractResult { extracted: number; errors: string[] }
export interface ExtractResult { framesExtracted: number; errors: string[] }
