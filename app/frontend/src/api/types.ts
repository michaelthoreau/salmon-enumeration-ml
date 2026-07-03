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

export interface DBVideoFrame {
  id:               number
  uuid:             string
  source_file_id:   number
  frame_number:     number
  split:            Split
  frame_rel_path:   string | null
  motion_rel_path:  string | null
  labeled_at:       string | null
  created_at:       string
}

export interface VideoInfo {
  id:            number
  original_name: string
  total_frames:  number
  fps:           number
}

export interface VideoFile extends DBFile {
  total_frames: number
  fps:          number
}

export interface CropsPage     { items: DBCrop[]; total: number; offset: number; limit: number }
export interface AnnotationData { file_id?: number; crop_id?: number; points: [number, number][] }
export interface ScanResult    { added: number; skipped: number }
export interface AssignResult  { assigned: number; errors: string[] }
export interface CropResult    { cropsGenerated: number; errors: string[] }
export interface ExtractResult { extracted: number; errors: string[] }
export interface ExtractResult { framesExtracted: number; errors: string[] }
