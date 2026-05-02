export type AdultAgeCategory = "adult_18_plus" | "adult_21_plus" | "adult_25_plus" | "adult_30_plus";
export type GenerationMode = "portrait" | "full_body" | "lifestyle_post" | "studio" | "reference_match";
export type QualityPreset = "fast" | "balanced" | "high" | "ultra";
export type DatasetType = "body_part" | "body_shape" | "pose" | "style" | "fictional_face_identity";
export type FacePolicy = "reject_faces" | "redact_faces" | "body_part_crops_only";
export type SourceRights = "synthetic" | "owned" | "licensed" | "consented";
export type VisibilityLevel = "clear" | "partial" | "covered" | "not_visible" | "unclear";
export type DatasetPrepTarget =
  | "face_identity"
  | "chest_detail"
  | "butt_hips"
  | "genital_detail"
  | "legs_feet"
  | "upper_torso"
  | "full_body_context";
export type DatasetPrepScanMode = "local" | "claude" | "off";
export type DatasetPrepJobState = "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
export type TrainingRunState = "queued" | "running" | "cancelled" | "completed" | "failed";

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

export interface SelectedFolderResponse {
  folder_path: string | null;
}

export interface ImageCountResponse {
  path: string;
  image_count: number;
}

export interface GenerationRequest {
  character_id: string;
  mode: GenerationMode;
  quality: QualityPreset;
  scene_prompt: string;
  extra_negative: string;
  seed: number | null;
  global_lora_files?: string[] | null;
  lora_strength?: number;
}

export interface PromptEnhanceRequest {
  character_id: string;
  brief: string;
  mode: GenerationMode;
}

export interface PromptEnhanceResponse {
  scene_prompt: string;
  body_detail_prompt: string;
  extra_negative: string;
  summary: string;
}

export interface GenerationSettings {
  checkpoint_name: string;
}

export interface GenerationPreflightItem {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface GenerationPreflightResponse {
  ready: boolean;
  settings: GenerationSettings;
  items: GenerationPreflightItem[];
  warnings: string[];
}

export interface PromptRecipe {
  positive: string;
  negative: string;
  seed: number;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  lora_files: string[];
  lora_strength?: number;
}

export interface GenerationJobResponse {
  prompt_id: string;
  recipe: PromptRecipe;
}

export type GenerationJobState = "queued" | "running" | "completed" | "failed" | "unknown";

export interface GenerationOutputImage {
  filename: string;
  subfolder: string;
  type: string;
}

export interface GenerationJobStatus {
  prompt_id: string;
  status: GenerationJobState;
  queue_position: number | null;
  images: GenerationOutputImage[];
  error: string | null;
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

export interface DatasetPrepRequest {
  source_folder: string;
  output_folder: string | null;
  target: DatasetPrepTarget;
  recursive: boolean;
  scan_mode: DatasetPrepScanMode;
  use_ai: boolean;
  ai_max_images: number;
  ai_model: string;
}

export interface DatasetPrepImage {
  source_path: string;
  output_path: string | null;
  width: number;
  height: number;
  face_count: number;
  accepted: boolean;
  reason: string;
  method: string;
  crop_box: number[] | null;
}

export interface DatasetPrepResponse {
  output_folder: string;
  processed_count: number;
  cropped_count: number;
  skipped_count: number;
  ai_attempted_count: number;
  ai_guided_count: number;
  ai_failed_count: number;
  face_guided_count: number;
  fallback_count: number;
  warnings: string[];
  images: DatasetPrepImage[];
}

export interface DatasetPrepJobStatus {
  job_id: string;
  status: DatasetPrepJobState;
  total_count: number;
  processed_count: number;
  cropped_count: number;
  skipped_count: number;
  ai_attempted_count: number;
  ai_guided_count: number;
  ai_failed_count: number;
  face_guided_count: number;
  fallback_count: number;
  active_file: string | null;
  output_folder: string | null;
  scan_mode: DatasetPrepScanMode;
  use_ai: boolean;
  ai_max_images: number;
  cancel_requested: boolean;
  error: string | null;
  result: DatasetPrepResponse | null;
  created_at?: string;
  updated_at?: string;
}

export interface GlobalLearningRequest {
  pack_name: string;
  source_folder: string;
  dataset_type: DatasetType;
  target: DatasetPrepTarget;
  recursive: boolean;
  scan_mode: DatasetPrepScanMode;
  preset: "fast" | "balanced" | "high_quality";
  base_model_path: string;
  output_dir: string;
}

export interface GlobalLearningResponse {
  prep: DatasetPrepResponse;
  training_job: TrainingJobConfig;
  version: number;
  warnings: string[];
}

export interface TrainingConfigRequest {
  dataset_id: string;
  dataset_path: string;
  output_dir: string;
  base_model_path: string;
  lora_name: string;
  dataset_type: DatasetType | null;
  global_pack: boolean;
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

export interface ClearTrainingJobsResponse {
  removed_count: number;
  remaining_jobs: TrainingJobConfig[];
}

export interface TrainingRunStatus {
  run_id: string;
  job_id: string;
  status: TrainingRunState;
  trainer_entrypoint: string;
  config_path: string;
  output_lora_path: string | null;
  process_id: number | null;
  exit_code: number | null;
  log_path: string | null;
  progress_current: number | null;
  progress_total: number | null;
  progress_percent: number | null;
  tail: string[];
  error: string | null;
  started_at?: string;
  updated_at?: string;
}

export interface TrainerStatus {
  trainer_entrypoint: string;
  config_path: string;
  trainer_entrypoint_exists: boolean;
  config_path_exists: boolean;
  warnings: string[];
}

export interface ApiKeyStatus {
  provider: string;
  saved: boolean;
  model: string;
}

export interface ApiKeyTestResponse {
  provider: string;
  ok: boolean;
  status_code: number | null;
  model: string;
  message: string;
}
