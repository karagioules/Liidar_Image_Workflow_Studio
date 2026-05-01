export type AdultAgeCategory = "adult_18_plus" | "adult_21_plus" | "adult_25_plus" | "adult_30_plus";
export type GenerationMode = "portrait" | "full_body" | "lifestyle_post" | "studio" | "reference_match";
export type QualityPreset = "fast" | "balanced" | "high" | "ultra";
export type DatasetType = "body_part" | "body_shape" | "pose" | "style" | "fictional_face_identity";
export type FacePolicy = "reject_faces" | "redact_faces" | "body_part_crops_only";
export type SourceRights = "synthetic" | "owned" | "licensed" | "consented";
export type VisibilityLevel = "clear" | "partial" | "covered" | "not_visible" | "unclear";

export interface CharacterProfile {
  id: string;
  display_name: string;
  age_category: AdultAgeCategory;
  face_summary: string;
  hair: string;
  eyes: string;
  skin_tone: string;
  body_shape: string;
  chest: string;
  grooming: string;
  style_notes: string;
  negative_notes: string;
  reference_images: string[];
  lora_files: string[];
  seed_strategy: "locked" | "vary" | "reuse_last";
  locked_seed: number | null;
  created_at?: string;
  updated_at?: string;
}

export interface ReferenceAnalysisImage {
  path: string;
  file_name: string;
  width: number;
  height: number;
  orientation: string;
  brightness: string;
  tone: string;
  texture: string;
  caption: string | null;
  body_attributes: {
    coverage: string;
    chest_visibility: string;
    pose_framing: string;
    confidence: number;
    evidence: string;
  };
  image_details: ReferenceImageDetails;
  adult_content: AdultContentSignals;
  vision_tags: VisionTag[];
}

export interface AttributeDetail {
  visibility: VisibilityLevel;
  confidence: number;
  summary: string;
  evidence: string;
}

export interface ReferenceImageDetails {
  face: AttributeDetail;
  hair: AttributeDetail;
  skin: AttributeDetail;
  body_shape: AttributeDetail;
  chest: AttributeDetail;
  waist_hips: AttributeDetail;
  pose: AttributeDetail;
  clothing: AttributeDetail;
  lighting: AttributeDetail;
  camera: AttributeDetail;
  background: AttributeDetail;
  quality: AttributeDetail;
}

export interface AdultContentSignals {
  nudity_level: string;
  breast_visibility: string;
  nipple_areola_visibility: string;
  genital_visibility: string;
  buttocks_visibility: string;
  sexual_activity: string;
  confidence: number;
  evidence: string;
}

export interface VisionTag {
  name: string;
  confidence: number;
  source: string;
}

export interface AggregateReferenceIntelligence extends ReferenceImageDetails {
  adult_content: AdultContentSignals;
  strong_tags: VisionTag[];
  prompt_summary: string;
  uncertainty_notes: string[];
}

export interface ReferenceAnalysisResponse extends CharacterProfile {
  consistency_score: number;
  analysis_warnings: string[];
  analysis_images: ReferenceAnalysisImage[];
  aggregate_intelligence: AggregateReferenceIntelligence;
}

export interface RuntimeStatus {
  os_name: string;
  python_version: string;
  cpu_name: string;
  total_ram_gb: number;
  gpu_names: string[];
  amd_driver_version: string | null;
  comfyui_path_exists: boolean;
  warnings: string[];
}

export interface SystemLiveMetrics {
  timestamp: string;
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  gpu_percent: number | null;
  gpu_memory_used_gb: number | null;
  gpu_memory_total_gb: number | null;
  gpu_provider: string | null;
  process_memory_mb: number;
  warnings: string[];
}

export interface PathBrowserEntry {
  name: string;
  path: string;
  kind: "directory" | "file" | "drive";
}

export interface PathBrowserResponse {
  current_path: string | null;
  parent_path: string | null;
  entries: PathBrowserEntry[];
}

export interface SelectedReferenceImagesResponse {
  selected_paths: string[];
}

export interface GenerationRequest {
  character_id: string;
  mode: GenerationMode;
  quality: QualityPreset;
  scene_prompt: string;
  extra_negative: string;
  seed: number | null;
}

export interface PromptRecipe {
  positive: string;
  negative: string;
  seed: number;
  width: number;
  height: number;
  steps: number;
  cfg: number;
}

export interface GenerationJobResponse {
  prompt_id: string;
  recipe: PromptRecipe;
}

export interface DatasetScanRequest {
  name: string;
  source_folder: string;
  dataset_type: DatasetType;
  face_policy: FacePolicy;
  character_id?: string | null;
  source_rights?: SourceRights | null;
  tags: string[];
}

export interface DatasetScanReport {
  dataset_id: string;
  name: string;
  dataset_type: DatasetType;
  accepted_count: number;
  rejected_count: number;
  duplicate_count: number;
  ignored_count: number;
  warnings: string[];
}

export interface TrainingConfigRequest {
  dataset_id: string;
  dataset_path: string;
  output_dir: string;
  base_model_path: string;
  lora_name: string;
  resolution: number;
  repeats: number;
  batch_size: number;
  max_train_steps: number;
  learning_rate: number;
  network_dim: number;
  network_alpha: number;
}

export interface TrainingJobConfig extends TrainingConfigRequest {
  job_id: string;
  config_path: string | null;
  accepted_image_count: number;
  completed_lora_path: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface TrainerStatus {
  trainer_entrypoint: string;
  config_path: string;
  trainer_entrypoint_exists: boolean;
  config_path_exists: boolean;
  warnings: string[];
}
