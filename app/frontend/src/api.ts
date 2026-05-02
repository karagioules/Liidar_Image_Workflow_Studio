import type {
  ApiKeyStatus,
  ApiKeyTestResponse,
  CharacterProfile,
  ClearTrainingJobsResponse,
  DatasetPrepJobStatus,
  DatasetPrepRequest,
  DatasetPrepResponse,
  DatasetScanReport,
  DatasetScanRequest,
  GenerationJobResponse,
  GenerationJobStatus,
  GenerationOutputImage,
  GenerationRequest,
  GlobalLearningRequest,
  GlobalLearningResponse,
  ImageCountResponse,
  PathBrowserResponse,
  PromptRecipe,
  ReferenceAnalysisResponse,
  RuntimeStatus,
  SelectedFolderResponse,
  SelectedReferenceImagesResponse,
  SystemLiveMetrics,
  TrainerStatus,
  TrainingConfigRequest,
  TrainingJobConfig,
  TrainingRunStatus
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    headers: isFormData
      ? init?.headers
      : {
          "Content-Type": "application/json",
          ...init?.headers
        },
    ...init
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // The backend usually returns JSON, but keep fetch errors readable if it does not.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const api = {
  runtime: () => request<RuntimeStatus>("/api/runtime"),
  systemLive: () => request<SystemLiveMetrics>("/api/system/live"),
  anthropicKeyStatus: () => request<ApiKeyStatus>("/api/settings/anthropic-key"),
  saveAnthropicKey: (apiKey: string, model = "claude-haiku-4-5-20251001") =>
    request<ApiKeyStatus>("/api/settings/anthropic-key", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey, model })
    }),
  deleteAnthropicKey: () =>
    request<ApiKeyStatus>("/api/settings/anthropic-key", {
      method: "DELETE"
    }),
  testAnthropicKey: () =>
    request<ApiKeyTestResponse>("/api/settings/anthropic-key/test", {
      method: "POST"
    }),
  browseFilesystem: (path: string) => {
    const params = new URLSearchParams();
    if (path.trim()) {
      params.set("path", path.trim());
    }
    return request<PathBrowserResponse>(`/api/filesystem/browse?${params.toString()}`);
  },
  selectReferenceImages: (mode: "files" | "folder") =>
    request<SelectedReferenceImagesResponse>(`/api/filesystem/select-references?${new URLSearchParams({ mode }).toString()}`, {
      method: "POST"
    }),
  selectFolder: () =>
    request<SelectedFolderResponse>("/api/filesystem/select-folder", {
      method: "POST"
    }),
  characters: () => request<CharacterProfile[]>("/api/characters"),
  analyzeReferences: (profile: CharacterProfile) =>
    request<ReferenceAnalysisResponse>("/api/characters/analyze-references?vision=true", {
      method: "POST",
      body: JSON.stringify(profile)
    }),
  thumbnailUrl: (path: string) => `${API_BASE}/api/filesystem/thumbnail?${new URLSearchParams({ path }).toString()}`,
  imageCount: (path: string) => request<ImageCountResponse>(`/api/filesystem/image-count?${new URLSearchParams({ path }).toString()}`),
  saveCharacter: (profile: CharacterProfile) =>
    request<CharacterProfile>("/api/characters", {
      method: "POST",
      body: JSON.stringify(profile)
    }),
  previewGeneration: (payload: GenerationRequest) =>
    request<PromptRecipe>("/api/generate/preview", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  queueGeneration: (payload: GenerationRequest) =>
    request<GenerationJobResponse>("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  generationJob: (promptId: string) => request<GenerationJobStatus>(`/api/generate/jobs/${promptId}`),
  generatedImageUrl: (image: GenerationOutputImage) =>
    `${API_BASE}/api/generate/image?${new URLSearchParams({
      filename: image.filename,
      subfolder: image.subfolder,
      type: image.type
    }).toString()}`,
  scanTraining: (payload: DatasetScanRequest) =>
    request<DatasetScanReport>("/api/training/scan", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  trainingJobs: () => request<TrainingJobConfig[]>("/api/training/jobs"),
  clearPendingTrainingJobs: () =>
    request<ClearTrainingJobsResponse>("/api/training/jobs/pending", {
      method: "DELETE"
    }),
  trainingRuns: () => request<TrainingRunStatus[]>("/api/training/runs"),
  createGlobalLearningJob: (payload: GlobalLearningRequest) =>
    request<GlobalLearningResponse>("/api/learning/global-job", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  prepDatasetCrops: (payload: DatasetPrepRequest) =>
    request<DatasetPrepResponse>("/api/dataset-prep/crop", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  startDatasetPrepJob: (payload: DatasetPrepRequest) =>
    request<DatasetPrepJobStatus>("/api/dataset-prep/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  datasetPrepJob: (jobId: string) => request<DatasetPrepJobStatus>(`/api/dataset-prep/jobs/${jobId}`),
  cancelDatasetPrepJob: (jobId: string) =>
    request<DatasetPrepJobStatus>(`/api/dataset-prep/jobs/${jobId}/cancel`, {
      method: "POST"
    }),
  createTrainingConfig: (payload: { request: TrainingConfigRequest; accepted_image_count: number }) =>
    request<TrainingJobConfig>("/api/training/config", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  startTrainingRun: (jobId: string, trainerEntrypoint = "tools/train_global_lora.ps1") =>
    request<TrainingRunStatus>(`/api/training/${jobId}/runs`, {
      method: "POST",
      body: JSON.stringify({ trainer_entrypoint: trainerEntrypoint })
    }),
  trainingRun: (runId: string) => request<TrainingRunStatus>(`/api/training/runs/${runId}`),
  cancelTrainingRun: (runId: string) =>
    request<TrainingRunStatus>(`/api/training/runs/${runId}/cancel`, {
      method: "POST"
    }),
  trainingStatus: (trainerEntryPoint: string, configPath: string) => {
    const params = new URLSearchParams({ trainer_entrypoint: trainerEntryPoint });
    if (configPath.trim()) {
      params.set("config_path", configPath);
    }
    return request<TrainerStatus>(`/api/training/status?${params.toString()}`);
  },
  registerLora: (jobId: string, loraPath: string) =>
    request<TrainingJobConfig>(`/api/training/${jobId}/register-lora`, {
      method: "POST",
      body: JSON.stringify({ lora_path: loraPath })
    })
};
