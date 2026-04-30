import type {
  CharacterProfile,
  DatasetScanReport,
  DatasetScanRequest,
  GenerationJobResponse,
  GenerationRequest,
  PathBrowserResponse,
  PromptRecipe,
  ReferenceAnalysisResponse,
  RuntimeStatus,
  SelectedReferenceImagesResponse,
  TrainerStatus,
  TrainingConfigRequest,
  TrainingJobConfig
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
  characters: () => request<CharacterProfile[]>("/api/characters"),
  analyzeReferences: (profile: CharacterProfile) =>
    request<ReferenceAnalysisResponse>("/api/characters/analyze-references?vision=true", {
      method: "POST",
      body: JSON.stringify(profile)
    }),
  thumbnailUrl: (path: string) => `${API_BASE}/api/filesystem/thumbnail?${new URLSearchParams({ path }).toString()}`,
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
  scanTraining: (payload: DatasetScanRequest) =>
    request<DatasetScanReport>("/api/training/scan", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  createTrainingConfig: (payload: { request: TrainingConfigRequest; accepted_image_count: number }) =>
    request<TrainingJobConfig>("/api/training/config", {
      method: "POST",
      body: JSON.stringify(payload)
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
