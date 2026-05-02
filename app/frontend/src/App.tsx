import { Activity, AlertTriangle, Camera, CheckCircle2, ChevronDown, ChevronRight, ClipboardList, Copy, FolderOpen, Play, Plus, RefreshCcw, Save, Scissors, Search, UserRound, X } from "lucide-react";
import React from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type {
  AdultContentSignals,
  AttributeDetail,
  CharacterProfile,
  DatasetPrepJobStatus,
  DatasetPrepResponse,
  DatasetPrepTarget,
  DatasetScanReport,
  DatasetType,
  FacePolicy,
  GenerationJobStatus,
  GenerationMode,
  GenerationPreflightResponse,
  GenerationSettings,
  GlobalLearningResponse,
  PromptRecipe,
  QualityPreset,
  ReferenceAnalysisResponse,
  RuntimeStatus,
  SystemLiveMetrics,
  TrainerStatus,
  TrainingJobConfig,
  TrainingRunStatus,
  VisionTag
} from "./types";

type TabId = "characters" | "generate" | "training";
type TrainingPreset = "fast" | "balanced" | "high_quality";
type PackInfluence = "natural" | "balanced" | "strong";
type TrackedGenerationJob = GenerationJobStatus & { recipe: PromptRecipe };
type AppLogLevel = "info" | "success" | "warning" | "error";
type AppLogEntry = { id: string; at: string; level: AppLogLevel; message: string };

const blankCharacter: CharacterProfile = {
  id: "new-character",
  display_name: "",
  age_category: "adult_25_plus",
  face_summary: "",
  hair: "",
  eyes: "",
  skin_tone: "",
  body_shape: "",
  chest: "",
  grooming: "",
  style_notes: "",
  negative_notes: "",
  reference_images: [],
  lora_files: [],
  seed_strategy: "vary",
  locked_seed: null
};

function newCharacterProfile(): CharacterProfile {
  return {
    ...blankCharacter,
    id: `character-${Date.now()}`
  };
}

const tabs: Array<{ id: TabId; label: string; icon: typeof Camera }> = [
  { id: "characters", label: "Characters", icon: UserRound },
  { id: "generate", label: "Generate", icon: Camera },
  { id: "training", label: "Training", icon: ClipboardList }
];

const prepTargets: DatasetPrepTarget[] = ["face_identity", "chest_detail", "butt_hips", "genital_detail", "legs_feet", "full_body_context"];
const packInfluenceOptions: Array<{ id: PackInfluence; label: string; strength: number; description: string }> = [
  { id: "natural", label: "Natural", strength: 0.55, description: "Light influence, keeps the prompt in charge." },
  { id: "balanced", label: "Balanced", strength: 0.75, description: "Default blend between prompt and trained pack." },
  { id: "strong", label: "Strong", strength: 0.95, description: "Pushes the trained pack harder." }
];

function LiidarLogo() {
  return (
    <svg className="brand-logo" viewBox="0 0 40 40" aria-hidden="true">
      <path className="brand-logo-body" d="M20 5.5c4.5 4.1 6.8 9.1 6.8 15.2 0 5.2-2.2 9.4-6.8 12.9-4.6-3.5-6.8-7.7-6.8-12.9 0-6.1 2.3-11.1 6.8-15.2Z" />
      <path className="brand-logo-cut" d="M20 8.4v24.1M14.2 17.3c3.7 1.9 7.9 1.9 11.6 0M13.8 24.1c4.2 2.1 8.2 2.1 12.4 0" />
      <path className="brand-logo-orbit" d="M8 28.5c5.8 5.5 18.2 5.5 24 0" />
    </svg>
  );
}

function tabSubtitle(tab: TabId) {
  const subtitles: Record<TabId, string> = {
    characters: "Build identity blueprints from adult references.",
    generate: "Compose finished stills from one clear brief.",
    training: "Prepare high-quality sculpture packs for body detail."
  };
  return subtitles[tab];
}

function referencePathsText(profile: CharacterProfile): string {
  return profile.reference_images.join("\n");
}

function parseReferencePaths(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((path) => path.trim())
    .filter(Boolean);
}

function uniquePaths(paths: string[]): string[] {
  return Array.from(new Set(paths.map((path) => path.trim()).filter(Boolean)));
}

function labelize(value: string) {
  return value.replace(/_/g, " ");
}

function prepTargetLabel(value: DatasetPrepTarget) {
  const labels: Record<DatasetPrepTarget, string> = {
    face_identity: "Face / identity",
    chest_detail: "Breasts / chest",
    butt_hips: "Butt / hips",
    genital_detail: "Genital detail",
    legs_feet: "Legs / feet",
    upper_torso: "Upper body",
    full_body_context: "Full body"
  };
  return labels[value];
}

function prepTargetHint(value: DatasetPrepTarget) {
  const hints: Record<DatasetPrepTarget, string> = {
    face_identity: "Use clear, consistent adult face references when the pack should preserve identity.",
    chest_detail: "Use clear close examples when the pack should improve breast or chest anatomy.",
    butt_hips: "Use images where the hips and butt are the main visible shape.",
    genital_detail: "Use only clear adult references for precise anatomy detail.",
    legs_feet: "Use full or lower-body images where legs and feet are visible.",
    upper_torso: "Use this for shoulders, torso, waist, and upper-body consistency.",
    full_body_context: "Use this for overall body shape, proportions, pose, and silhouette."
  };
  return hints[value];
}

function datasetTypeForPrepTarget(value: DatasetPrepTarget): DatasetType {
  if (value === "face_identity") {
    return "fictional_face_identity";
  }
  if (value === "full_body_context") {
    return "body_shape";
  }
  return "body_part";
}

function trainingPresetSettings(preset: TrainingPreset) {
  if (preset === "fast") {
    return { resolution: 640, max_train_steps: 450, learning_rate: 0.0001, network_dim: 16, network_alpha: 8, repeats: 4 };
  }
  if (preset === "high_quality") {
    return { resolution: 896, max_train_steps: 1200, learning_rate: 0.00002, network_dim: 32, network_alpha: 16, repeats: 8 };
  }
  return { resolution: 768, max_train_steps: 900, learning_rate: 0.00003, network_dim: 32, network_alpha: 16, repeats: 6 };
}

function packNameFromDataset(name: string, type: DatasetType) {
  const base = name.trim() || `global-${type}`;
  return base.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "global_improvement_pack";
}

function logTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function inferGenerationModeFromBrief(brief: string): GenerationMode {
  const value = brief.toLowerCase();
  if (/\b(full body|whole body|head to toe|standing|front view|side view)\b/.test(value)) {
    return "full_body";
  }
  if (/\b(studio|backdrop|professional portrait|headshot)\b/.test(value)) {
    return "studio";
  }
  if (/\b(reference|match|same pose|same composition)\b/.test(value)) {
    return "reference_match";
  }
  if (/\b(selfie|mirror|candid|lifestyle|casual|social media)\b/.test(value)) {
    return "lifestyle_post";
  }
  return "portrait";
}

function upsertTrainingRun(runs: TrainingRunStatus[], run: TrainingRunStatus) {
  return [...runs.filter((item) => item.run_id !== run.run_id), run];
}

function keepLastTrainingProgress(next: TrainingRunStatus, previous: TrainingRunStatus | null): TrainingRunStatus {
  if (
    previous?.run_id !== next.run_id ||
    next.progress_current !== null ||
    next.progress_total !== null ||
    next.progress_percent !== null
  ) {
    return next;
  }
  return {
    ...next,
    progress_current: previous.progress_current,
    progress_total: previous.progress_total,
    progress_percent: previous.progress_percent
  };
}

function upsertGenerationJob(jobs: TrackedGenerationJob[], job: TrackedGenerationJob) {
  return [job, ...jobs.filter((item) => item.prompt_id !== job.prompt_id)].slice(0, 12);
}

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("generate");
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<SystemLiveMetrics | null>(null);
  const [appLogs, setAppLogs] = useState<AppLogEntry[]>([
    { id: "startup", at: logTime(), level: "info", message: "Studio opened. Checking backend." }
  ]);
  const lastLiveLogKey = useRef("");
  const lastRuntimeLogKey = useRef("");
  const [characters, setCharacters] = useState<CharacterProfile[]>([]);
  const [selectedCharacterId, setSelectedCharacterId] = useState("");
  const [editingCharacter, setEditingCharacter] = useState<CharacterProfile>(blankCharacter);
  const [recipe, setRecipe] = useState<PromptRecipe | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [mode, setMode] = useState<GenerationMode>("portrait");
  const [quality, setQuality] = useState<QualityPreset>("ultra");
  const [photoBrief, setPhotoBrief] = useState("soft natural light, realistic adult studio photo");
  const [scenePrompt, setScenePrompt] = useState("soft natural light, realistic adult studio photo");
  const [bodyDetailPrompt, setBodyDetailPrompt] = useState("");
  const [selectedGlobalLora, setSelectedGlobalLora] = useState("all");
  const [packInfluence, setPackInfluence] = useState<PackInfluence>("balanced");
  const [extraNegative, setExtraNegative] = useState("");
  const [generationJobs, setGenerationJobs] = useState<TrackedGenerationJob[]>([]);
  const [generationSettings, setGenerationSettings] = useState<GenerationSettings>({ checkpoint_name: "sdxl_base_1.0.safetensors" });
  const [generationPreflight, setGenerationPreflight] = useState<GenerationPreflightResponse | null>(null);
  const [isEnhancingPrompt, setIsEnhancingPrompt] = useState(false);
  const [enhancedPromptSummary, setEnhancedPromptSummary] = useState("");

  const [datasetName, setDatasetName] = useState("global-body-pack");
  const [sourceFolder, setSourceFolder] = useState("");
  const [datasetType, setDatasetType] = useState<DatasetType>("body_shape");
  const [facePolicy, setFacePolicy] = useState<FacePolicy>("reject_faces");
  const [scanReport, setScanReport] = useState<DatasetScanReport | null>(null);
  const [datasetPath, setDatasetPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("models/sdxl_base_1.0.safetensors");
  const [loraName, setLoraName] = useState("global_body_pack");
  const [outputDir, setOutputDir] = useState("outputs/global_lora");
  const [acceptedImageCount, setAcceptedImageCount] = useState(0);
  const [trainingPreset, setTrainingPreset] = useState<TrainingPreset>("high_quality");
  const [trainingJob, setTrainingJob] = useState<TrainingJobConfig | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJobConfig[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRunStatus[]>([]);
  const [trainingRun, setTrainingRun] = useState<TrainingRunStatus | null>(null);
  const [isTrainingRunning, setIsTrainingRunning] = useState(false);
  const [learningJob, setLearningJob] = useState<GlobalLearningResponse | null>(null);
  const [isLearning, setIsLearning] = useState(false);
  const [trainerEntrypoint, setTrainerEntrypoint] = useState("tools/train_global_lora.ps1");
  const [trainerConfigPath, setTrainerConfigPath] = useState("");
  const [trainerStatus, setTrainerStatus] = useState<TrainerStatus | null>(null);
  const [prepSourceFolder, setPrepSourceFolder] = useState("");
  const [prepOutputFolder, setPrepOutputFolder] = useState("");
  const [prepTarget, setPrepTarget] = useState<DatasetPrepTarget>("chest_detail");
  const [prepRecursive, setPrepRecursive] = useState(true);
  const [prepReport, setPrepReport] = useState<DatasetPrepResponse | null>(null);
  const [prepJob, setPrepJob] = useState<DatasetPrepJobStatus | null>(null);
  const [isPreparingDataset, setIsPreparingDataset] = useState(false);

  useEffect(() => {
    void loadInitialData();
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        void loadInitialData({ preserveCharacter: true, silent: true });
      }
    }, 5000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (loraName === "global_body_pack" && datasetName.trim() && datasetName !== "global-body-pack") {
      setLoraName(packNameFromDataset(datasetName, datasetType));
    }
  }, [datasetName, datasetType, loraName]);

  useEffect(() => {
    let cancelled = false;

    const loadLiveMetrics = async () => {
      try {
        const metrics = await api.systemLive();
        if (!cancelled) {
          setLiveMetrics(metrics);
        }
      } catch {
        if (!cancelled) {
          setLiveMetrics(null);
        }
      }
    };

    void loadLiveMetrics();
    const intervalId = window.setInterval(() => void loadLiveMetrics(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    const selected = characters.find((character) => character.id === selectedCharacterId);
    if (selected) {
      setEditingCharacter(selected);
    }
  }, [characters, selectedCharacterId]);

  useEffect(() => {
    if (!prepJob || !["queued", "running", "cancelling"].includes(prepJob.status)) {
      return;
    }
    let cancelled = false;
    const intervalId = window.setInterval(async () => {
      try {
        const status = await api.datasetPrepJob(prepJob.job_id);
        if (cancelled) {
          return;
        }
        setPrepJob(status);
        if (status.result) {
          setPrepReport(status.result);
          setPrepOutputFolder(status.result.output_folder);
          setSourceFolder(status.result.output_folder);
          setDatasetPath(status.result.output_folder);
          setAcceptedImageCount(status.result.cropped_count);
          setLoraName(packNameFromDataset(datasetName, datasetType));
        }
        if (["completed", "cancelled", "failed"].includes(status.status)) {
          setIsPreparingDataset(false);
          if (status.status === "completed" && status.result) {
            setMessage(`Dataset prep created ${status.result.cropped_count} crop${status.result.cropped_count === 1 ? "" : "s"}.`);
          }
          if (status.status === "cancelled") {
            setMessage("Dataset prep cancelled.");
          }
          if (status.status === "failed" && status.error) {
            setError(status.error);
          }
        }
      } catch (caught) {
        if (!cancelled) {
          setIsPreparingDataset(false);
          setError(caught instanceof Error ? caught.message : "Unable to read dataset prep progress.");
        }
      }
    }, 900);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [prepJob]);

  useEffect(() => {
    if (!trainingRun || !["queued", "running"].includes(trainingRun.status)) {
      return;
    }
    let cancelled = false;
    const intervalId = window.setInterval(async () => {
      try {
        const status = await api.trainingRun(trainingRun.run_id);
        if (cancelled) {
          return;
        }
        const stableStatus = keepLastTrainingProgress(status, trainingRun);
        setTrainingRun(stableStatus);
        setTrainingRuns((current) => upsertTrainingRun(current, stableStatus));
        if (!["queued", "running"].includes(stableStatus.status)) {
          setIsTrainingRunning(false);
          if (stableStatus.status === "completed") {
            setMessage("Global training completed and the new pack is active for future generations.");
            setTrainingJobs(await api.trainingJobs());
          }
          if (stableStatus.status === "cancelled") {
            setMessage("Training run cancelled.");
          }
          if (stableStatus.status === "failed") {
            setError(stableStatus.error ?? "Training run failed. Check the log tail below.");
          }
        }
      } catch (caught) {
        if (!cancelled) {
          setIsTrainingRunning(false);
          setError(caught instanceof Error ? caught.message : "Unable to read training progress.");
        }
      }
    }, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [trainingRun]);

  useEffect(() => {
    const activeJobs = generationJobs.filter((job) => ["queued", "running", "unknown"].includes(job.status));
    if (!activeJobs.length) {
      return;
    }
    let cancelled = false;
    const pollJobs = async () => {
      const updates = await Promise.allSettled(activeJobs.map((job) => api.generationJob(job.prompt_id)));
      if (cancelled) {
        return;
      }
      setGenerationJobs((current) => {
        let next = current;
        updates.forEach((result, index) => {
          const originalJob = activeJobs[index];
          if (result.status === "fulfilled") {
            next = upsertGenerationJob(next, { ...result.value, recipe: originalJob.recipe });
          } else {
            next = upsertGenerationJob(next, {
              ...originalJob,
              status: "unknown",
              error: result.reason instanceof Error ? result.reason.message : "Unable to read generation status."
            });
          }
        });
        return next;
      });
    };
    void pollJobs();
    const intervalId = window.setInterval(() => void pollJobs(), 2500);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [generationJobs]);

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterId) ?? characters[0],
    [characters, selectedCharacterId]
  );

  const appendAppLog = useCallback((level: AppLogLevel, nextMessage: string) => {
    const trimmed = nextMessage.trim();
    if (!trimmed) {
      return;
    }
    setAppLogs((current) =>
      [
        { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, at: logTime(), level, message: trimmed },
        ...current
      ].slice(0, 80)
    );
  }, []);

  useEffect(() => {
    if (message) {
      appendAppLog("success", message);
    }
  }, [appendAppLog, message]);

  useEffect(() => {
    if (error) {
      appendAppLog("error", error);
    }
  }, [appendAppLog, error]);

  useEffect(() => {
    if (!runtime) {
      return;
    }
    const runtimeKey = [
      runtime.comfyui_path_exists ? "comfy-ok" : "comfy-missing",
      runtime.python_version,
      runtime.amd_driver_version ?? "no-amd-driver",
      runtime.warnings.join("|")
    ].join(":");
    if (runtimeKey === lastRuntimeLogKey.current) {
      return;
    }
    lastRuntimeLogKey.current = runtimeKey;
    appendAppLog(
      runtime.comfyui_path_exists ? "success" : "warning",
      `Runtime checked: Python ${runtime.python_version}, ${runtime.gpu_names.length ? runtime.gpu_names.join(", ") : "no GPU reported"}, ComfyUI ${runtime.comfyui_path_exists ? "found" : "missing"}.`
    );
    runtime.warnings.forEach((warning) => appendAppLog("warning", warning));
  }, [appendAppLog, runtime]);

  useEffect(() => {
    const liveKey = liveMetrics ? `online:${liveMetrics.warnings.join("|")}` : "offline";
    if (liveKey === lastLiveLogKey.current) {
      return;
    }
    lastLiveLogKey.current = liveKey;
    if (!liveMetrics) {
      appendAppLog("warning", "System metrics unavailable.");
      return;
    }
    appendAppLog("info", "System metrics connected.");
    liveMetrics.warnings.forEach((warning) => appendAppLog("warning", warning));
  }, [appendAppLog, liveMetrics]);

  async function loadInitialData(options: { preserveCharacter?: boolean; silent?: boolean } = {}) {
    try {
      if (!options.silent) {
        setError("");
      }
      const [runtimeStatus, profiles, jobs, runs, settings] = await Promise.all([
        api.runtime(),
        api.characters(),
        api.trainingJobs(),
        api.trainingRuns(),
        api.generationSettings()
      ]);
      setRuntime(runtimeStatus);
      setCharacters(profiles);
      setTrainingJobs(jobs);
      setTrainingRuns(runs);
      setGenerationSettings(settings);
      const activeRun = [...runs].reverse().find((run) => ["queued", "running"].includes(run.status)) ?? runs[runs.length - 1] ?? null;
      setTrainingRun(activeRun);
      setIsTrainingRunning(Boolean(activeRun && ["queued", "running"].includes(activeRun.status)));
      const latestPendingJob = [...jobs].reverse().find((job) => job.global_pack && !job.completed_lora_path) ?? null;
      if (!trainingJob && latestPendingJob) {
        setTrainingJob(latestPendingJob);
        setTrainerConfigPath(latestPendingJob.config_path ?? "");
        setDatasetPath(latestPendingJob.dataset_path);
        setOutputDir(latestPendingJob.output_dir);
        setLoraName(latestPendingJob.lora_name);
        setAcceptedImageCount(latestPendingJob.accepted_image_count);
      }
      if (profiles[0] && !options.preserveCharacter) {
        setSelectedCharacterId(profiles[0].id);
        setEditingCharacter(profiles[0]);
      }
    } catch (caught) {
      const loadError = caught instanceof Error ? caught.message : "Unable to load studio data.";
      if (options.silent) {
        appendAppLog("warning", `Auto-refresh failed: ${loadError}`);
      } else {
        setError(loadError);
      }
    }
  }

  async function saveCharacter() {
    try {
      setError("");
      const saved = await api.saveCharacter(editingCharacter);
      setCharacters((current) => {
        const withoutSaved = current.filter((character) => character.id !== saved.id);
        return [...withoutSaved, saved];
      });
      setSelectedCharacterId(saved.id);
      setMessage("Blueprint saved.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save blueprint.");
    }
  }

  function generationPayload() {
    const requestedGlobalLoras =
      selectedGlobalLora === "all"
        ? null
        : selectedGlobalLora === "none"
          ? []
          : [selectedGlobalLora];
    const selectedInfluence = packInfluenceOptions.find((item) => item.id === packInfluence) ?? packInfluenceOptions[1];
    return {
      character_id: selectedCharacter?.id ?? "",
      mode: inferGenerationModeFromBrief(photoBrief),
      quality: "ultra" as QualityPreset,
      scene_prompt: [scenePrompt, bodyDetailPrompt].map((part) => part.trim()).filter(Boolean).join(", "),
      extra_negative: extraNegative,
      seed: null,
      global_lora_files: requestedGlobalLoras,
      lora_strength: selectedInfluence.strength
    };
  }

  async function previewRecipe() {
    try {
      setError("");
      setRecipe(await api.previewGeneration(generationPayload()));
      setGenerationPreflight(await api.generationPreflight(generationPayload()));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to preview recipe.");
    }
  }

  async function checkGenerationReadiness() {
    try {
      setError("");
      setGenerationPreflight(await api.generationPreflight(generationPayload()));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to check generation readiness.");
    }
  }

  async function enhancePhotoBrief() {
    if (!photoBrief.trim()) {
      setError("Write a brief first.");
      return;
    }
    try {
      setIsEnhancingPrompt(true);
      setError("");
      const enhanced = await api.enhancePrompt({
        character_id: selectedCharacter?.id ?? "",
        brief: photoBrief,
        mode: inferGenerationModeFromBrief(photoBrief)
      });
      setMode(inferGenerationModeFromBrief(enhanced.scene_prompt));
      setQuality("ultra");
      setPhotoBrief(enhanced.scene_prompt);
      setScenePrompt(enhanced.scene_prompt);
      setBodyDetailPrompt(enhanced.body_detail_prompt);
      setExtraNegative(enhanced.extra_negative);
      setEnhancedPromptSummary(enhanced.summary);
      setRecipe(null);
      setMessage("Brief enhanced locally.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to enhance brief.");
    } finally {
      setIsEnhancingPrompt(false);
    }
  }

  async function saveGenerationSettings() {
    try {
      setError("");
      const saved = await api.saveGenerationSettings(generationSettings);
      setGenerationSettings(saved);
      setGenerationPreflight(await api.generationPreflight(generationPayload()));
      setMessage("Generation settings saved.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save generation settings.");
    }
  }

  async function queueGeneration() {
    try {
      setError("");
      const preflight = await api.generationPreflight(generationPayload());
      setGenerationPreflight(preflight);
      if (!preflight.ready) {
        setError("Generation needs attention. Check the readiness list.");
        return;
      }
      const result = await api.queueGeneration(generationPayload());
      setRecipe(result.recipe);
      setGenerationJobs((current) =>
        upsertGenerationJob(current, {
          prompt_id: result.prompt_id,
          status: "queued",
          queue_position: null,
          images: [],
          error: null,
          recipe: result.recipe
        })
      );
      setMessage(`Generation queued: ${result.prompt_id}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start generation.");
    }
  }

  async function scanDataset() {
    try {
      setError("");
      const report = await api.scanTraining({
        name: datasetName,
        source_folder: sourceFolder,
        dataset_type: datasetType,
        face_policy: facePolicy,
        character_id: null,
        source_rights: "owned",
        tags: []
      });
      setScanReport(report);
      setDatasetPath(sourceFolder);
      setAcceptedImageCount(Math.max(1, report.accepted_count));
      setLoraName(packNameFromDataset(datasetName, datasetType));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to scan dataset.");
    }
  }

  async function selectTrainingFolder() {
    try {
      setError("");
      const result = await api.selectFolder();
      if (result.folder_path) {
        setSourceFolder(result.folder_path);
        setDatasetPath(result.folder_path);
        const count = await api.imageCount(result.folder_path);
        setAcceptedImageCount(count.image_count);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to select training folder.");
    }
  }

  async function selectPrepFolder(kind: "source" | "output") {
    try {
      setError("");
      const result = await api.selectFolder();
      if (result.folder_path) {
        if (kind === "source") {
          setPrepSourceFolder(result.folder_path);
        } else {
          setPrepOutputFolder(result.folder_path);
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to select folder.");
    }
  }

  async function runDatasetPrep() {
    try {
      setError("");
      setMessage("");
      setIsPreparingDataset(true);
      setPrepReport(null);
      const job = await api.startDatasetPrepJob({
        source_folder: prepSourceFolder,
        output_folder: null,
        target: prepTarget,
        recursive: true,
        scan_mode: "local",
        use_ai: false,
        ai_max_images: 0,
        ai_model: ""
      });
      setPrepJob(job);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to prep dataset crops.");
      setIsPreparingDataset(false);
    }
  }

  function changePrepTarget(nextTarget: DatasetPrepTarget) {
    const nextDatasetType = datasetTypeForPrepTarget(nextTarget);
    setPrepTarget(nextTarget);
    setDatasetType(nextDatasetType);
    setLoraName(packNameFromDataset(datasetName, nextDatasetType));
  }

  async function cancelDatasetPrep() {
    if (!prepJob) {
      return;
    }
    try {
      setError("");
      const status = await api.cancelDatasetPrepJob(prepJob.job_id);
      setPrepJob(status);
    } finally {
      setMessage("Cancelling dataset prep...");
    }
  }

  async function createTrainingConfig() {
    try {
      setError("");
      const trainingFolder = (datasetPath || sourceFolder).trim();
      if (!trainingFolder) {
        throw new Error("Select a prepared crop folder before creating a training job.");
      }
      const counted = await api.imageCount(trainingFolder);
      if (counted.image_count <= 0) {
        throw new Error("The selected training folder does not contain any images.");
      }
      setDatasetPath(trainingFolder);
      setAcceptedImageCount(counted.image_count);
      const preset = trainingPresetSettings("high_quality");
      const packName = (loraName || packNameFromDataset(datasetName, datasetType)).trim();
      const job = await api.createTrainingConfig({
        request: {
          dataset_id: scanReport?.dataset_id || datasetName,
          dataset_path: trainingFolder,
          output_dir: outputDir,
          base_model_path: baseModelPath,
          lora_name: packName,
          dataset_type: datasetType,
          global_pack: true,
          resolution: preset.resolution,
          repeats: preset.repeats,
          batch_size: 1,
          max_train_steps: preset.max_train_steps,
          learning_rate: preset.learning_rate,
          network_dim: preset.network_dim,
          network_alpha: preset.network_alpha
        },
        accepted_image_count: counted.image_count
      });
      setLoraName(job.lora_name);
      setTrainingJob(job);
      setTrainingJobs((current) => [...current.filter((item) => item.job_id !== job.job_id), job]);
      setTrainerConfigPath(job.config_path ?? "");
      setMessage(`Global training job created for ${job.lora_name}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create training config.");
    }
  }

  async function createGlobalLearningJob() {
    try {
      setError("");
      setMessage("");
      setIsLearning(true);
      const result = await api.createGlobalLearningJob({
        pack_name: datasetName,
        source_folder: sourceFolder,
        dataset_type: datasetType,
        target: prepTarget,
        recursive: true,
        scan_mode: "local",
        preset: "high_quality",
        base_model_path: baseModelPath,
        output_dir: outputDir
      });
      setLearningJob(result);
      setTrainingJob(result.training_job);
      setTrainerConfigPath(result.training_job.config_path ?? "");
      setDatasetPath(result.prep.output_folder);
      setAcceptedImageCount(result.prep.cropped_count);
      setLoraName(result.training_job.lora_name);
      setTrainingJobs((current) => [...current.filter((item) => item.job_id !== result.training_job.job_id), result.training_job]);
      setMessage(`Global learning job ${result.training_job.lora_name} created with ${result.prep.cropped_count} clean crop${result.prep.cropped_count === 1 ? "" : "s"}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create global learning job.");
    } finally {
      setIsLearning(false);
    }
  }

  async function checkTrainerStatus() {
    try {
      setError("");
      setTrainerStatus(await api.trainingStatus(trainerEntrypoint, trainerConfigPath));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to check trainer status.");
    }
  }

  async function startTrainingRun() {
    if (!trainingJob) {
      return;
    }
    try {
      setError("");
      setMessage("");
      setIsTrainingRunning(true);
      const status = await api.startTrainingRun(trainingJob.job_id, trainerEntrypoint);
      setTrainingRun(status);
      setTrainingRuns((current) => upsertTrainingRun(current, status));
      setMessage(`Training started for ${trainingJob.lora_name}.`);
    } catch (caught) {
      setIsTrainingRunning(false);
      setError(caught instanceof Error ? caught.message : "Unable to start training.");
    }
  }

  async function cancelTrainingRun() {
    if (!trainingRun) {
      return;
    }
    try {
      setError("");
      const status = await api.cancelTrainingRun(trainingRun.run_id);
      setTrainingRun(status);
      setTrainingRuns((current) => upsertTrainingRun(current, status));
      setIsTrainingRunning(false);
      setMessage("Cancelling training run...");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to cancel training.");
    }
  }

  async function clearPendingTrainingJobs() {
    const activeRun = trainingRun && ["queued", "running"].includes(trainingRun.status);
    if (activeRun) {
      setError("Cancel or finish the current training run before clearing pending jobs.");
      return;
    }
    try {
      setError("");
      setMessage("");
      const result = await api.clearPendingTrainingJobs();
      const remainingJobs = result.remaining_jobs;
      const latestPendingJob = [...remainingJobs].reverse().find((job) => job.global_pack && !job.completed_lora_path) ?? null;
      const selectedStillExists = trainingJob ? remainingJobs.some((job) => job.job_id === trainingJob.job_id) : false;
      setTrainingJobs(remainingJobs);
      if (!selectedStillExists) {
        setTrainingJob(latestPendingJob);
        setTrainerConfigPath(latestPendingJob?.config_path ?? "");
      }
      if (trainingRun && !remainingJobs.some((job) => job.job_id === trainingRun.job_id)) {
        setTrainingRun(null);
      }
      setMessage(`Cleared ${result.removed_count} pending training job${result.removed_count === 1 ? "" : "s"}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to clear pending training jobs.");
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <LiidarLogo />
          <div>
            <strong>Liidar</strong>
            <span>Body sculpture studio</span>
          </div>
          <RuntimeBadge runtime={runtime} />
        </div>
        <nav className="tab-list" aria-label="Studio sections">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon aria-hidden="true" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
        <SystemMonitor metrics={liveMetrics} />
        <AppLogPanel logs={appLogs} onClear={() => setAppLogs([])} />
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{tabs.find((tab) => tab.id === activeTab)?.label}</h1>
            <p>{tabSubtitle(activeTab)}</p>
          </div>
        </header>

        {error ? <div className="notice error">{error}</div> : null}
        {message ? <div className="notice success">{message}</div> : null}

        {activeTab === "characters" ? (
          <CharactersPanel
            characters={characters}
            selectedCharacterId={selectedCharacterId}
            editingCharacter={editingCharacter}
            onSelect={setSelectedCharacterId}
            onNew={() => {
              setSelectedCharacterId("");
              setEditingCharacter(newCharacterProfile());
            }}
            onChange={setEditingCharacter}
            onSave={saveCharacter}
          />
        ) : null}
        {activeTab === "generate" ? (
          <GeneratePanel
            characters={characters}
            selectedCharacterId={selectedCharacter?.id ?? ""}
            mode={mode}
            quality={quality}
            photoBrief={photoBrief}
            scenePrompt={scenePrompt}
            bodyDetailPrompt={bodyDetailPrompt}
            selectedGlobalLora={selectedGlobalLora}
            packInfluence={packInfluence}
            extraNegative={extraNegative}
            recipe={recipe}
            generationJobs={generationJobs}
            trainingJobs={trainingJobs}
            settings={generationSettings}
            preflight={generationPreflight}
            isEnhancingPrompt={isEnhancingPrompt}
            enhancedPromptSummary={enhancedPromptSummary}
            onCharacterChange={setSelectedCharacterId}
            onModeChange={setMode}
            onQualityChange={setQuality}
            onPhotoBriefChange={setPhotoBrief}
            onScenePromptChange={setScenePrompt}
            onBodyDetailPromptChange={setBodyDetailPrompt}
            onSelectedGlobalLoraChange={setSelectedGlobalLora}
            onPackInfluenceChange={setPackInfluence}
            onExtraNegativeChange={setExtraNegative}
            onSettingsChange={setGenerationSettings}
            onSaveSettings={() => void saveGenerationSettings()}
            onCheckReadiness={() => void checkGenerationReadiness()}
            onEnhanceBrief={() => void enhancePhotoBrief()}
            onPreview={previewRecipe}
            onQueue={queueGeneration}
          />
        ) : null}
        {activeTab === "training" ? (
          <TrainingPanel
            datasetName={datasetName}
            sourceFolder={sourceFolder}
            datasetType={datasetType}
            facePolicy={facePolicy}
            trainingPreset={trainingPreset}
            scanReport={scanReport}
            datasetPath={datasetPath}
            baseModelPath={baseModelPath}
            loraName={loraName}
            outputDir={outputDir}
            acceptedImageCount={acceptedImageCount}
            trainerEntrypoint={trainerEntrypoint}
            trainerConfigPath={trainerConfigPath}
            trainerStatus={trainerStatus}
            trainingJob={trainingJob}
            trainingJobs={trainingJobs}
            trainingRun={trainingRun}
            trainingRuns={trainingRuns}
            isTrainingRunning={isTrainingRunning}
            learningJob={learningJob}
            isLearning={isLearning}
            prepSourceFolder={prepSourceFolder}
            prepTarget={prepTarget}
            prepReport={prepReport}
            prepJob={prepJob}
            isPreparing={isPreparingDataset}
            onDatasetNameChange={setDatasetName}
            onSourceFolderChange={setSourceFolder}
            onFacePolicyChange={setFacePolicy}
            onTrainingPresetChange={setTrainingPreset}
            onDatasetPathChange={setDatasetPath}
            onBaseModelPathChange={setBaseModelPath}
            onLoraNameChange={setLoraName}
            onOutputDirChange={setOutputDir}
            onAcceptedImageCountChange={setAcceptedImageCount}
            onTrainerEntrypointChange={setTrainerEntrypoint}
            onTrainerConfigPathChange={setTrainerConfigPath}
            onSelectSourceFolder={selectTrainingFolder}
            onCreateGlobalLearningJob={() => void createGlobalLearningJob()}
            onScan={scanDataset}
            onCreateConfig={createTrainingConfig}
            onStartTraining={() => void startTrainingRun()}
            onCancelTraining={() => void cancelTrainingRun()}
            onClearPendingJobs={() => void clearPendingTrainingJobs()}
            onCheckStatus={checkTrainerStatus}
            onPrepSourceFolderChange={setPrepSourceFolder}
            onPrepTargetChange={changePrepTarget}
            onSelectPrepSource={() => void selectPrepFolder("source")}
            onRunPrep={() => void runDatasetPrep()}
            onCancelPrep={() => void cancelDatasetPrep()}
          />
        ) : null}
      </main>
    </div>
  );
}

function RuntimeBadge({ runtime }: { runtime: RuntimeStatus | null }) {
  const ready = Boolean(runtime?.comfyui_path_exists);
  const statusText = runtime ? (ready ? "Ready" : "Needs setup") : "Checking";
  const detailText = runtime?.amd_driver_version ? "AMD detected" : runtime?.gpu_names[0] ?? "Runtime";
  return (
    <div className={ready ? "runtime-badge ready" : "runtime-badge warning"} title={runtime ? `Python ${runtime.python_version} - ${runtime.gpu_names.join(", ") || "No GPU reported"}` : "Checking runtime"}>
      {ready ? <CheckCircle2 aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
      <div>
        <span>Runtime</span>
        <strong>{statusText}</strong>
        <small>{detailText}</small>
      </div>
    </div>
  );
}

function AppLogPanel({ logs, onClear }: { logs: AppLogEntry[]; onClear: () => void }) {
  const [copied, setCopied] = useState(false);
  const logText = logs.map((entry) => `[${entry.at}] ${entry.level.toUpperCase()}: ${entry.message}`).join("\n");

  async function copyLogs() {
    if (!logText) {
      return;
    }
    await navigator.clipboard.writeText(logText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <section className="sidebar-log-panel" aria-label="Live app logs">
      <div className="monitor-heading">
        <span>Activity</span>
        <strong>{logs.length ? `${logs.length} events` : "Clear"}</strong>
      </div>
      <div className="log-actions">
        <button type="button" onClick={() => void copyLogs()} disabled={!logs.length}>
          <Copy aria-hidden="true" />
          {copied ? "Copied" : "Copy all"}
        </button>
        <button type="button" onClick={onClear} disabled={!logs.length}>
          <X aria-hidden="true" />
          Clear
        </button>
      </div>
      <div className="live-log-list">
        {logs.length ? logs.slice(0, 18).map((entry) => (
          <article key={entry.id} className={`live-log-entry ${entry.level}`}>
            <div>
              <span>{entry.at}</span>
              <strong>{entry.level}</strong>
            </div>
            <p>{entry.message}</p>
          </article>
        )) : <p className="empty-log">No events yet.</p>}
      </div>
    </section>
  );
}

function SystemMonitor({ metrics }: { metrics: SystemLiveMetrics | null }) {
  const gpuText = metrics?.gpu_percent == null ? "Unavailable" : `${Math.round(metrics.gpu_percent)}%`;
  const vramText = metrics?.gpu_memory_used_gb == null ? "VRAM unavailable" : `${metrics.gpu_memory_used_gb.toFixed(2)} GB VRAM`;
  return (
    <section className="sidebar-monitor" aria-label="Live system monitor">
      <div className="monitor-heading">
        <span>System</span>
        <strong>{metrics ? "Live" : "Checking"}</strong>
      </div>
      <MonitorBar label="CPU" value={metrics?.cpu_percent ?? 0} text={metrics ? `${Math.round(metrics.cpu_percent)}%` : "Checking"} />
      <MonitorBar
        label="RAM"
        value={metrics?.ram_percent ?? 0}
        text={metrics ? `${metrics.ram_used_gb.toFixed(1)} / ${metrics.ram_total_gb.toFixed(1)} GB` : "Checking"}
      />
      <MonitorBar label="GPU" value={metrics?.gpu_percent ?? 0} text={gpuText} muted={metrics?.gpu_percent == null} />
      <div className="monitor-meta">
        <span>{vramText}</span>
        <span>App memory {metrics ? `${Math.round(metrics.process_memory_mb)} MB` : "Checking"}</span>
      </div>
      {metrics?.warnings[0] ? <p className="monitor-warning">{metrics.warnings[0]}</p> : null}
    </section>
  );
}

function MonitorBar({ label, value, text, muted = false }: { label: string; value: number; text: string; muted?: boolean }) {
  const safeValue = Math.min(Math.max(value, 0), 100);
  return (
    <div className={muted ? "monitor-row muted" : "monitor-row"}>
      <div>
        <span>{label}</span>
        <strong>{text}</strong>
      </div>
      <div className="monitor-track" aria-hidden="true">
        <span style={{ width: `${safeValue}%` }} />
      </div>
    </div>
  );
}

function CharactersPanel(props: {
  characters: CharacterProfile[];
  selectedCharacterId: string;
  editingCharacter: CharacterProfile;
  onSelect: (id: string) => void;
  onNew: () => void;
  onChange: (profile: CharacterProfile) => void;
  onSave: () => void;
}) {
  const { characters, selectedCharacterId, editingCharacter, onSelect, onNew, onChange, onSave } = props;
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [analysisByCharacterId, setAnalysisByCharacterId] = useState<Record<string, ReferenceAnalysisResponse>>({});
  const currentAnalysis = analysisByCharacterId[editingCharacter.id] ?? null;
  useEffect(() => {
    setAnalysisError("");
  }, [editingCharacter.id]);
  const update = (field: keyof CharacterProfile, value: string) => onChange({ ...editingCharacter, [field]: value });
  const updateReferences = (paths: string[]) => {
    setAnalysisByCharacterId((current) => {
      const next = { ...current };
      delete next[editingCharacter.id];
      return next;
    });
    onChange({ ...editingCharacter, reference_images: uniquePaths(paths) });
  };
  const analyzeReferences = async () => {
    if (!editingCharacter.reference_images.length) {
      return;
    }
    try {
      setIsAnalyzing(true);
      setAnalysisError("");
      const analyzed = await api.analyzeReferences(editingCharacter);
      setAnalysisByCharacterId((current) => ({ ...current, [editingCharacter.id]: analyzed }));
      onChange({ ...editingCharacter, ...analyzed, reference_images: analyzed.reference_images ?? editingCharacter.reference_images });
    } catch (caught) {
      setAnalysisError(caught instanceof Error ? caught.message : "Unable to analyze reference images.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section className="panel split-panel">
      <div className="list-rail">
        <h2>Character blueprints</h2>
        <button type="button" className="secondary-button rail-action" onClick={onNew}>
          <Plus aria-hidden="true" />
          New blueprint
        </button>
        {characters.map((character) => (
          <button
            type="button"
            key={character.id}
            className={selectedCharacterId === character.id ? "row-button active" : "row-button"}
            onClick={() => onSelect(character.id)}
          >
            <strong>{character.display_name}</strong>
          </button>
        ))}
      </div>
      <form className="form-grid" onSubmit={(event) => { event.preventDefault(); onSave(); }}>
        <label>
          Blueprint name
          <input value={editingCharacter.display_name} onChange={(event) => update("display_name", event.target.value)} />
        </label>
        <ReferencePathModule paths={editingCharacter.reference_images} onChange={updateReferences} />
        {analysisError ? <div className="inline-error">{analysisError}</div> : null}
        {isAnalyzing ? <AnalysisProgress imageCount={editingCharacter.reference_images.length} /> : null}
        {currentAnalysis ? <ReferenceAnalysisPanel analysis={currentAnalysis} /> : <CharacterReadiness profile={editingCharacter} />}
        <div className="actions compact-actions">
          <button type="button" className="primary-button" onClick={() => void analyzeReferences()} disabled={!editingCharacter.reference_images.length || isAnalyzing}>
            <Search aria-hidden="true" />
            {isAnalyzing ? "Reading references" : "Build blueprint"}
          </button>
        </div>
        <details className="advanced-fields">
          <summary>Manual blueprint fields</summary>
          <div className="form-grid inner-grid">
            <Textarea label="Face summary" value={editingCharacter.face_summary} onChange={(value) => update("face_summary", value)} />
            <Textarea label="Hair" value={editingCharacter.hair} onChange={(value) => update("hair", value)} />
            <Textarea label="Eyes" value={editingCharacter.eyes} onChange={(value) => update("eyes", value)} />
            <Textarea label="Skin tone" value={editingCharacter.skin_tone} onChange={(value) => update("skin_tone", value)} />
            <Textarea label="Body shape" value={editingCharacter.body_shape} onChange={(value) => update("body_shape", value)} />
            <Textarea label="Chest" value={editingCharacter.chest} onChange={(value) => update("chest", value)} />
            <Textarea label="Grooming" value={editingCharacter.grooming} onChange={(value) => update("grooming", value)} />
            <Textarea label="Style notes" value={editingCharacter.style_notes} onChange={(value) => update("style_notes", value)} />
            <Textarea label="Negative notes" value={editingCharacter.negative_notes} onChange={(value) => update("negative_notes", value)} />
          </div>
        </details>
        <button type="submit" className="primary-button">
          <Save aria-hidden="true" />
          Save blueprint
        </button>
      </form>
    </section>
  );
}

function GeneratePanel(props: {
  characters: CharacterProfile[];
  selectedCharacterId: string;
  mode: GenerationMode;
  quality: QualityPreset;
  photoBrief: string;
  scenePrompt: string;
  bodyDetailPrompt: string;
  selectedGlobalLora: string;
  packInfluence: PackInfluence;
  extraNegative: string;
  recipe: PromptRecipe | null;
  generationJobs: TrackedGenerationJob[];
  trainingJobs: TrainingJobConfig[];
  settings: GenerationSettings;
  preflight: GenerationPreflightResponse | null;
  isEnhancingPrompt: boolean;
  enhancedPromptSummary: string;
  onCharacterChange: (id: string) => void;
  onModeChange: (mode: GenerationMode) => void;
  onQualityChange: (quality: QualityPreset) => void;
  onPhotoBriefChange: (value: string) => void;
  onScenePromptChange: (value: string) => void;
  onBodyDetailPromptChange: (value: string) => void;
  onSelectedGlobalLoraChange: (value: string) => void;
  onPackInfluenceChange: (value: PackInfluence) => void;
  onExtraNegativeChange: (value: string) => void;
  onSettingsChange: (settings: GenerationSettings) => void;
  onSaveSettings: () => void;
  onCheckReadiness: () => void;
  onEnhanceBrief: () => void;
  onPreview: () => void;
  onQueue: () => void;
}) {
  const activePacks = props.trainingJobs.filter((job) => job.global_pack && job.completed_lora_path);
  const selectedInfluence = packInfluenceOptions.find((option) => option.id === props.packInfluence) ?? packInfluenceOptions[1];

  return (
    <section className="panel generate-workflow-panel">
      <div className="generate-control-grid">
        <section className="generate-card">
          <div className="generate-card-heading">
            <strong>Identity</strong>
            <span>Blueprint base</span>
          </div>
          <label>
            Character blueprint
            <select value={props.selectedCharacterId} onChange={(event) => props.onCharacterChange(event.target.value)}>
              {props.characters.map((character) => <option key={character.id} value={character.id}>{character.display_name}</option>)}
            </select>
          </label>
          <div className="auto-generation-note">
            <strong>Auto sculpt</strong>
            <span>Highest quality. Framing follows the brief.</span>
          </div>
        </section>
      </div>

      <section className="generate-scene-card">
        <div className="generate-card-heading">
          <strong>Local model</strong>
          <span>ComfyUI target</span>
        </div>
        <div className="checkpoint-row">
          <label>
            Checkpoint
            <input
              value={props.settings.checkpoint_name}
              onChange={(event) => props.onSettingsChange({ ...props.settings, checkpoint_name: event.target.value })}
              placeholder="sdxl_base_1.0.safetensors"
            />
          </label>
          <button type="button" className="secondary-button" onClick={props.onSaveSettings}>
            <Save aria-hidden="true" />
            Save
          </button>
          <button type="button" className="secondary-button" onClick={props.onCheckReadiness}>
            <RefreshCcw aria-hidden="true" />
            Check
          </button>
        </div>
        <GenerationPreflightPanel preflight={props.preflight} />
      </section>

      <section className="generate-scene-card">
        <div className="generate-card-heading">
          <strong>Compose image</strong>
          <span>One brief, locally expanded</span>
        </div>
        <div className="brief-composer">
          <Textarea
            label="Brief"
            value={props.photoBrief}
            onChange={props.onPhotoBriefChange}
          />
          <div className="actions compact-actions">
            <button type="button" className="secondary-button" onClick={props.onEnhanceBrief} disabled={props.isEnhancingPrompt || !props.photoBrief.trim()}>
              <RefreshCcw aria-hidden="true" />
              {props.isEnhancingPrompt ? "Composing" : "Enhance brief"}
            </button>
            <button type="button" className="secondary-button" onClick={props.onPreview}>
              <Search aria-hidden="true" />
              Preview
            </button>
          </div>
          {props.enhancedPromptSummary ? <p className="prompt-status-line">{props.enhancedPromptSummary}</p> : null}
        </div>
        <details className="generate-advanced">
          <summary>Generated prompt details</summary>
          <section className="advanced-pack-routing">
            <div className="generate-card-heading">
              <strong>Pack routing</strong>
              <span>{activePacks.length ? `${activePacks.length} active` : "None active"}</span>
            </div>
            <label>
              Routing mode
              <select value={props.selectedGlobalLora} onChange={(event) => props.onSelectedGlobalLoraChange(event.target.value)}>
                <option value="all">Auto-select relevant packs</option>
                <option value="none">No trained pack</option>
                {activePacks.map((job) => (
                  <option key={job.job_id} value={job.completed_lora_path ?? ""}>
                    {job.lora_name}
                  </option>
                ))}
              </select>
            </label>
            <div className="pack-strength-row" role="radiogroup" aria-label="Pack influence">
              {packInfluenceOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  role="radio"
                  aria-checked={props.packInfluence === option.id}
                  className={props.packInfluence === option.id ? "strength-button active" : "strength-button"}
                  onClick={() => props.onPackInfluenceChange(option.id)}
                  title={option.description}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <small>Strength {selectedInfluence.strength.toFixed(2)}</small>
          </section>
          <div className="form-grid two-column">
            <Textarea label="Scene prompt" value={props.scenePrompt} onChange={props.onScenePromptChange} />
            <Textarea label="Body detail prompt" value={props.bodyDetailPrompt} onChange={props.onBodyDetailPromptChange} />
          </div>
          <Textarea label="Extra negative" value={props.extraNegative} onChange={props.onExtraNegativeChange} />
        </details>
      </section>

      <div className="actions">
        <button
          type="button"
          className="primary-button"
          onClick={props.onQueue}
          disabled={props.preflight ? !props.preflight.ready : false}
          title={props.preflight && !props.preflight.ready ? "Check readiness before generating." : "Send to ComfyUI"}
        >
          <Play aria-hidden="true" />
          Generate
        </button>
      </div>
      {props.recipe ? (
        <div className="recipe-box">
          <div className="recipe-summary-header">
            <h3>Generation recipe</h3>
            <span>{props.recipe.width} x {props.recipe.height} · {props.recipe.steps} steps · CFG {props.recipe.cfg}</span>
          </div>
          <dl className="recipe-summary-grid">
            <div><dt>Character</dt><dd>{props.characters.find((character) => character.id === props.selectedCharacterId)?.display_name ?? "Selected character"}</dd></div>
            <div><dt>Pack influence</dt><dd>{props.recipe.lora_files.length ? `${props.recipe.lora_files.length} pack${props.recipe.lora_files.length === 1 ? "" : "s"} · ${props.recipe.lora_strength ?? selectedInfluence.strength}` : "No trained pack"}</dd></div>
            <div><dt>Seed</dt><dd>{props.recipe.seed}</dd></div>
          </dl>
          {props.recipe.lora_files.length ? (
            <div className="active-pack-list">
              {props.recipe.lora_files.map((file) => <span key={file}>{file.split(/[\\/]/).pop() ?? file}</span>)}
            </div>
          ) : null}
          <details className="recipe-technical">
            <summary>Full prompt recipe</summary>
            <p>{props.recipe.positive}</p>
            <dl>
              <div><dt>Negative</dt><dd>{props.recipe.negative}</dd></div>
            </dl>
          </details>
        </div>
      ) : null}
      <GenerationJobsPanel jobs={props.generationJobs} />
    </section>
  );
}

function GenerationPreflightPanel({ preflight }: { preflight: GenerationPreflightResponse | null }) {
  if (!preflight) {
    return <p className="empty-analysis-note">Preview or check once before generating.</p>;
  }
  return (
    <div className="preflight-panel">
      <div className={`inline-status ${preflight.ready ? "ok" : "warning"}`}>
        {preflight.ready ? <CheckCircle2 aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
        <span>{preflight.ready ? "Ready to generate" : "Needs attention"}</span>
      </div>
      <div className="preflight-list">
        {preflight.items.map((item) => (
          <div className={item.ok ? "preflight-item ok" : "preflight-item warning"} key={item.id}>
            {item.ok ? <CheckCircle2 aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
            <div>
              <strong>{item.label}</strong>
              <span>{item.detail}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GenerationJobsPanel({ jobs }: { jobs: TrackedGenerationJob[] }) {
  const completedImages = jobs.flatMap((job) => job.images.map((image) => ({ job, image })));

  return (
    <section className="generation-jobs-panel" aria-label="Generation jobs">
      <div className="panel-heading compact-heading">
        <h3>Generation jobs</h3>
        <p>Queued ComfyUI jobs and completed local outputs.</p>
      </div>
      {jobs.length ? (
        <div className="generation-job-list">
          {jobs.map((job) => {
            const isActive = ["queued", "running", "unknown"].includes(job.status);
            return (
              <article className={`generation-job-card ${job.status}`} key={job.prompt_id}>
                <div className="generation-job-main">
                  <div>
                    <strong>{labelize(job.status)}</strong>
                    <span>{job.prompt_id}</span>
                  </div>
                  <div>
                    <span>{job.queue_position ? `Queue position ${job.queue_position}` : `${job.recipe.width} x ${job.recipe.height}`}</span>
                    <span>{job.recipe.steps} steps · CFG {job.recipe.cfg}</span>
                  </div>
                </div>
                {isActive ? (
                  <div className="progress-track" role="progressbar" aria-label={`Generation ${job.prompt_id}`}>
                    <span />
                  </div>
                ) : null}
                {job.error ? <div className="inline-error">{job.error}</div> : null}
                {job.images.length ? (
                  <div className="generation-job-images">
                    {job.images.map((image) => (
                      <a href={api.generatedImageUrl(image)} target="_blank" rel="noreferrer" key={`${job.prompt_id}-${image.filename}`}>
                        <img src={api.generatedImageUrl(image)} alt={image.filename} />
                        <span>{image.filename}</span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      ) : (
        <p className="empty-analysis-note">No queued jobs yet.</p>
      )}
      {completedImages.length ? (
        <div className="generation-gallery">
          <h3>Completed outputs</h3>
          <div>
            {completedImages.slice(0, 12).map(({ job, image }) => (
              <a href={api.generatedImageUrl(image)} target="_blank" rel="noreferrer" key={`${job.prompt_id}-${image.filename}-gallery`}>
                <img src={api.generatedImageUrl(image)} alt={image.filename} />
              </a>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function TrainingPanel(props: {
  datasetName: string;
  sourceFolder: string;
  datasetType: DatasetType;
  facePolicy: FacePolicy;
  trainingPreset: TrainingPreset;
  scanReport: DatasetScanReport | null;
  datasetPath: string;
  baseModelPath: string;
  loraName: string;
  outputDir: string;
  acceptedImageCount: number;
  trainerEntrypoint: string;
  trainerConfigPath: string;
  trainerStatus: TrainerStatus | null;
  trainingJob: TrainingJobConfig | null;
  trainingJobs: TrainingJobConfig[];
  trainingRun: TrainingRunStatus | null;
  trainingRuns: TrainingRunStatus[];
  isTrainingRunning: boolean;
  learningJob: GlobalLearningResponse | null;
  isLearning: boolean;
  prepSourceFolder: string;
  prepTarget: DatasetPrepTarget;
  prepReport: DatasetPrepResponse | null;
  prepJob: DatasetPrepJobStatus | null;
  isPreparing: boolean;
  onDatasetNameChange: (value: string) => void;
  onSourceFolderChange: (value: string) => void;
  onFacePolicyChange: (value: FacePolicy) => void;
  onTrainingPresetChange: (value: TrainingPreset) => void;
  onDatasetPathChange: (value: string) => void;
  onBaseModelPathChange: (value: string) => void;
  onLoraNameChange: (value: string) => void;
  onOutputDirChange: (value: string) => void;
  onAcceptedImageCountChange: (value: number) => void;
  onTrainerEntrypointChange: (value: string) => void;
  onTrainerConfigPathChange: (value: string) => void;
  onSelectSourceFolder: () => void;
  onCreateGlobalLearningJob: () => void;
  onScan: () => void;
  onCreateConfig: () => void;
  onStartTraining: () => void;
  onCancelTraining: () => void;
  onClearPendingJobs: () => void;
  onCheckStatus: () => void;
  onPrepSourceFolderChange: (value: string) => void;
  onPrepTargetChange: (value: DatasetPrepTarget) => void;
  onSelectPrepSource: () => void;
  onRunPrep: () => void;
  onCancelPrep: () => void;
}) {
  const trainingFolder = props.datasetPath || props.sourceFolder;
  const hasTrainingFolder = Boolean(trainingFolder.trim());
  const globalPackName = props.loraName || packNameFromDataset(props.datasetName, props.datasetType);
  const completedGlobalPacks = props.trainingJobs.filter((job) => job.global_pack && job.completed_lora_path);
  const pendingGlobalJobs = props.trainingJobs.filter((job) => job.global_pack && !job.completed_lora_path);
  const latestCompletedPack = completedGlobalPacks.length ? completedGlobalPacks[completedGlobalPacks.length - 1] : null;
  const activeRun = props.trainingRun;
  const runIsActive = Boolean(activeRun && ["queued", "running"].includes(activeRun.status));
  const setupSaved = Boolean(props.trainingJob && props.trainingJob.lora_name === globalPackName);
  const primaryActionLabel = runIsActive ? "Training running" : setupSaved ? "Start training" : "Save training setup";
  const primaryAction = setupSaved ? props.onStartTraining : props.onCreateConfig;
  const prepProgressTotal = props.prepJob?.total_count ?? 0;
  const prepProgressProcessed = props.prepJob?.processed_count ?? 0;
  const prepProgressPercent = prepProgressTotal > 0 ? Math.round((prepProgressProcessed / prepProgressTotal) * 100) : props.isPreparing ? 4 : 0;
  const prepIsCancelable = Boolean(props.prepJob && ["queued", "running", "cancelling"].includes(props.prepJob.status));
  const canRunPrep = Boolean(props.prepSourceFolder.trim()) && !props.isPreparing;
  const setupStatus = runIsActive
    ? "Training is running"
    : setupSaved
      ? `${globalPackName} is ready to train`
      : hasTrainingFolder
        ? "Ready to save"
        : "Prepare references first";

  return (
    <section className="training-layout global-training">
      <div className="panel training-simple-panel training-step-panel">
        <div className="panel-heading">
          <h2>Train sculpture pack</h2>
          <p>Prepare references, name the pack, then train at the best local preset.</p>
        </div>

        <section className="training-step-card">
          <div className="training-step-header">
            <span className="step-number">1</span>
            <div>
              <h3>Prepare references</h3>
              <p>Choose the raw folder and the body focus this pack should learn.</p>
            </div>
          </div>
          <div className="folder-picker-row">
            <input value={props.prepSourceFolder} onChange={(event) => props.onPrepSourceFolderChange(event.target.value)} placeholder="Select raw image folder" />
            <button type="button" className="secondary-button inline-button" onClick={props.onSelectPrepSource}>
              <FolderOpen aria-hidden="true" />
              Select folder
            </button>
          </div>
          <label>
            Body focus
            <select value={props.prepTarget} onChange={(event) => props.onPrepTargetChange(event.target.value as DatasetPrepTarget)}>
              {prepTargets.map((target) => <option key={target} value={target}>{prepTargetLabel(target)}</option>)}
            </select>
            <small className="field-hint">{prepTargetHint(props.prepTarget)}</small>
          </label>
          <div className="actions compact-actions">
            <button type="button" className="primary-button" onClick={props.onRunPrep} disabled={!canRunPrep}>
              <Scissors aria-hidden="true" />
              {props.isPreparing ? "Preparing" : "Prepare references"}
            </button>
            {prepIsCancelable ? (
              <button type="button" className="secondary-button danger-button" onClick={props.onCancelPrep} disabled={props.prepJob?.status === "cancelling"}>
                <X aria-hidden="true" />
                {props.prepJob?.status === "cancelling" ? "Cancelling" : "Cancel"}
              </button>
            ) : null}
          </div>
          {props.prepJob ? (
            <section className="prep-progress-panel">
              <div className="prep-progress-header">
                <strong>{labelize(props.prepJob.status)}</strong>
                <span>{prepProgressProcessed} / {prepProgressTotal || "?"} images</span>
              </div>
              <div className="progress-track" aria-label="Dataset prep progress" role="progressbar" aria-valuenow={prepProgressPercent} aria-valuemin={0} aria-valuemax={100}>
                <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, prepProgressPercent))}%` }} />
              </div>
              <div className="prep-progress-meta">
                <span>{props.prepJob.cropped_count} crops saved, {props.prepJob.skipped_count} skipped</span>
                {props.prepJob.error ? <span className="error-text">{props.prepJob.error}</span> : null}
              </div>
            </section>
          ) : null}
          {props.prepReport ? (
            <div className="training-library-strip">
              <span>{props.prepReport.cropped_count} crops ready</span>
              <span>{props.prepReport.output_folder}</span>
            </div>
          ) : null}
        </section>

        <section className="training-step-card">
          <div className="training-step-header">
            <span className="step-number">2</span>
            <div>
              <h3>Name the sculpture</h3>
              <p>Generation will auto-select this pack when the brief needs it.</p>
            </div>
          </div>

          <div className="pack-setup-card">
            <label>
              Pack name
              <input
                value={props.datasetName}
                disabled={runIsActive}
                onChange={(event) => {
                  const nextName = event.target.value;
                  props.onDatasetNameChange(nextName);
                  props.onLoraNameChange(packNameFromDataset(nextName, props.datasetType));
                }}
              />
            </label>
            <div className="field-block">
              <label htmlFor="training-folder">Export cropped images folder</label>
              <div className="folder-picker-row">
                <input
                  id="training-folder"
                  value={trainingFolder}
                  disabled={runIsActive}
                  onChange={(event) => {
                    props.onSourceFolderChange(event.target.value);
                    props.onDatasetPathChange(event.target.value);
                  }}
                  placeholder="Cropped images folder appears after step 1"
                />
                <button type="button" className="secondary-button inline-button" onClick={props.onSelectSourceFolder} disabled={runIsActive}>
                  <FolderOpen aria-hidden="true" />
                  Select folder
                </button>
              </div>
              <small className="field-hint">{props.acceptedImageCount ? `${props.acceptedImageCount} exported crops ready` : "Auto-filled after references are prepared."}</small>
            </div>
          </div>
        </section>

        <section className="training-step-card training-final-step">
          <div className="training-step-header">
            <span className="step-number">3</span>
            <div>
              <h3>Train locally</h3>
              <p>Uses the highest quality preset automatically.</p>
            </div>
          </div>

        <div className="training-action-card">
          <div>
            <span>Status</span>
            <strong>{setupStatus}</strong>
            <small>{setupSaved ? "Ready to start local training." : "Save once the cropped images folder is ready."}</small>
          </div>
          <div className="actions compact-actions">
            <button
              type="button"
              className="primary-button"
              onClick={primaryAction}
              disabled={!hasTrainingFolder || props.isTrainingRunning || runIsActive}
            >
              {setupSaved ? <Play aria-hidden="true" /> : <Save aria-hidden="true" />}
              {primaryActionLabel}
            </button>
            {runIsActive ? (
              <button type="button" className="secondary-button danger-button" onClick={props.onCancelTraining}>
                <X aria-hidden="true" />
                Cancel
              </button>
            ) : null}
          </div>
        </div>
        </section>

        {activeRun ? <TrainingRunPanel run={activeRun} /> : null}

        {completedGlobalPacks.length || latestCompletedPack ? (
          <div className="training-library-strip">
            <span>{completedGlobalPacks.length} active pack{completedGlobalPacks.length === 1 ? "" : "s"}</span>
            <span>{latestCompletedPack ? `Latest: ${latestCompletedPack.lora_name}` : "No finished pack yet"}</span>
          </div>
        ) : null}

        <details className="training-diagnostics hidden">
          <summary>Queue tools and technical details</summary>
          <div className="count-grid">
            <Metric label="Ready packs" value={completedGlobalPacks.length} />
            <Metric label="Pending jobs" value={pendingGlobalJobs.length} />
            <Metric label="Latest ready" value={latestCompletedPack?.lora_name ?? "None yet"} />
            <Metric label="Runs this session" value={props.trainingRuns.length} />
          </div>
          <div className="actions compact-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={props.onClearPendingJobs}
              disabled={!pendingGlobalJobs.length || runIsActive}
            >
              <X aria-hidden="true" />
              Clear pending jobs
            </button>
          </div>
          {pendingGlobalJobs.length ? (
            <div className="job-list">
              {pendingGlobalJobs.slice(-5).map((job) => (
                <article key={job.job_id} className="job-row">
                  <strong>{job.lora_name}</strong>
                  <span>{job.accepted_image_count} images · {job.config_path ?? "config saved"}</span>
                </article>
              ))}
            </div>
          ) : null}
        </details>
      </div>

    </section>
  );
}

function trainingOverallProgress(run: TrainingRunStatus) {
  if (run.status === "completed") {
    return { percent: 100, label: "Complete", detail: "Pack is ready" };
  }
  if (run.status === "failed") {
    return { percent: run.progress_percent ?? 0, label: "Failed", detail: "Training stopped before completion" };
  }
  if (run.status === "cancelled") {
    return { percent: run.progress_percent ?? 0, label: "Cancelled", detail: "Training was stopped" };
  }

  const current = run.progress_current;
  const total = run.progress_total;
  const phasePercent = run.progress_percent ?? 0;
  if (current === null || total === null) {
    return { percent: 0, label: "Starting", detail: "Waiting for trainer progress" };
  }

  const isTrainingPhase = total >= 1000;
  if (isTrainingPhase) {
    return {
      percent: Math.min(99, 25 + phasePercent * 0.75),
      label: "Training model",
      detail: `${current} / ${total} steps`
    };
  }

  return {
    percent: Math.min(25, phasePercent * 0.25),
    label: "Preparing images",
    detail: `${current} / ${total} images`
  };
}

function TrainingRunPanel({ run }: { run: TrainingRunStatus }) {
  const isActive = ["queued", "running"].includes(run.status);
  const statusLabel = labelize(run.status);
  const overallProgress = trainingOverallProgress(run);

  return (
    <section className={`training-run-panel ${run.status}`} aria-live="polite">
      <div className="training-run-header">
        <div className="inline-status">
          {run.status === "failed" ? <AlertTriangle aria-hidden="true" /> : <Activity aria-hidden="true" />}
          <span>{statusLabel}</span>
        </div>
        <span>{run.process_id ? `PID ${run.process_id}` : "No process id yet"}</span>
      </div>
      {isActive ? (
        <div className="training-progress-block">
          <div
            className="progress-track determinate"
            role="progressbar"
            aria-label="Training progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(overallProgress.percent)}
          >
            <span className="progress-fill" style={{ width: `${overallProgress.percent}%` }} />
          </div>
          <div className="training-progress-meta">
            <strong>{`${overallProgress.percent.toFixed(1)}% · ${overallProgress.label}`}</strong>
            <span>{overallProgress.detail}</span>
          </div>
        </div>
      ) : null}
      {run.error ? <div className="inline-error">{run.error}</div> : null}
      <details className="run-path-details hidden">
        <summary>Technical run details</summary>
        <div className="training-run-grid">
          <Metric label="Exit code" value={run.exit_code ?? "Running"} />
          <Metric label="Config" value={run.config_path || "Missing"} />
          <Metric label="Output" value={run.output_lora_path ?? "Pending"} />
          <Metric label="Log" value={run.log_path ?? "Pending"} />
        </div>
        {run.tail.length ? (
          <div className="log-tail nested-log-tail">
            <strong>Training log tail</strong>
            <pre>{run.tail.join("\n")}</pre>
          </div>
        ) : null}
      </details>
    </section>
  );
}

function ReferencePathModule({ paths, onChange }: { paths: string[]; onChange: (paths: string[]) => void }) {
  const [browserError, setBrowserError] = useState("");
  const [isSelecting, setIsSelecting] = useState(false);
  const [importMessage, setImportMessage] = useState("");
  const [isListOpen, setIsListOpen] = useState(false);

  useEffect(() => {
    if (!paths.length) {
      setIsListOpen(false);
    }
  }, [paths.length]);

  const removePath = (path: string) => onChange(paths.filter((candidate) => candidate !== path));
  const selectReferences = async (mode: "files" | "folder") => {
    try {
      setIsSelecting(true);
      setBrowserError("");
      setImportMessage("");
      const result = await api.selectReferenceImages(mode);
      if (!result.selected_paths.length) {
        setImportMessage("No images selected.");
        return;
      }
      onChange(uniquePaths([...paths, ...result.selected_paths]));
      setIsListOpen(false);
      const selectedCount = result.selected_paths.length;
      setImportMessage(`${selectedCount} image${selectedCount === 1 ? "" : "s"} selected.`);
    } catch (caught) {
      setBrowserError(caught instanceof Error ? caught.message : "Unable to select reference images.");
    } finally {
      setIsSelecting(false);
    }
  };

  return (
    <section className="path-module" aria-label="Reference image path module">
      <div className="path-module-heading">
        <div>
          <strong>Reference images</strong>
          <span>{paths.length} selected</span>
        </div>
      </div>
      <div className="picker-actions">
        <button type="button" className="secondary-button picker-button" onClick={() => void selectReferences("files")} disabled={isSelecting}>
          <Plus aria-hidden="true" />
          Select images
        </button>
        <button type="button" className="secondary-button picker-button" onClick={() => void selectReferences("folder")} disabled={isSelecting}>
          <FolderOpen aria-hidden="true" />
          Select folder
        </button>
      </div>
      {importMessage ? <div className="inline-success">{importMessage}</div> : null}
      {browserError ? <div className="inline-error">{browserError}</div> : null}
      <div className="selected-paths">
        {paths.length ? (
          <button
            type="button"
            className="selected-paths-bar"
            aria-expanded={isListOpen}
            aria-controls="selected-reference-list"
            aria-label={`${isListOpen ? "Hide" : "Show"} selected image list`}
            onClick={() => setIsListOpen((current) => !current)}
          >
            <span>
              {isListOpen ? <ChevronDown aria-hidden="true" /> : <ChevronRight aria-hidden="true" />}
              <strong>{paths.length} selected</strong>
            </span>
            <small>{isListOpen ? "Hide list" : "Show list"}</small>
          </button>
        ) : null}
        {paths.length ? (
          isListOpen ? (
          <div id="selected-reference-list" className="selected-path-list">
            {paths.map((path) => (
              <div className="selected-path" key={path}>
                <img className="path-thumb" src={api.thumbnailUrl(path)} alt="" />
                <span>{path}</span>
                <button type="button" className="icon-mini-button" onClick={() => removePath(path)} aria-label={`Remove ${path}`}>
                  <X aria-hidden="true" />
                </button>
              </div>
            ))}
          </div>
          ) : null
        ) : (
          <p>No reference images selected.</p>
        )}
      </div>
      <details className="paste-paths">
        <summary>Paste paths</summary>
        <Textarea label="Reference image paths" value={paths.join("\n")} onChange={(value) => onChange(uniquePaths(parseReferencePaths(value)))} />
      </details>
    </section>
  );
}

function CharacterReadiness({ profile }: { profile: CharacterProfile }) {
  const checks: Array<[string, boolean]> = [
    ["References", profile.reference_images.length >= 1],
    ["Name", Boolean(profile.display_name.trim())],
    ["Profile", Boolean(profile.face_summary.trim() || profile.style_notes.trim())]
  ];
  const passed = checks.filter(([, ok]) => ok).length;
  return (
    <section className="analysis-panel compact-panel">
      <div className="analysis-score">
        <strong>{passed}/{checks.length}</strong>
        <span>Setup readiness</span>
      </div>
      <div className="readiness-list">
        {checks.map(([label, ok]) => (
          <span className={ok ? "ready" : "pending"} key={label}>{label}</span>
        ))}
      </div>
    </section>
  );
}

function AnalysisProgress({ imageCount }: { imageCount: number }) {
  return (
    <section className="analysis-progress" aria-live="polite">
      <div>
        <strong>Analyzing references</strong>
        <span>Reading selected images and local vision captions for {imageCount} image{imageCount === 1 ? "" : "s"}.</span>
      </div>
      <div className="progress-track" role="progressbar" aria-label="Analyzing references">
        <span />
      </div>
    </section>
  );
}

function ReferenceAnalysisPanel({ analysis }: { analysis: ReferenceAnalysisResponse }) {
  const intelligence = analysis.aggregate_intelligence;
  const topTags = intelligence.strong_tags.slice(0, 14);
  return (
    <section className="analysis-dashboard">
      <div className="analysis-dashboard-header">
        <div className="score-orb">
          <strong>{analysis.consistency_score}</strong>
          <span>Consistency</span>
        </div>
        <div className="analysis-title-block">
          <span>Character analysis</span>
          <h3>Reference blueprint</h3>
          <p>{intelligence.prompt_summary}</p>
        </div>
        <div className="analysis-kpis">
          <Metric label="References" value={analysis.analysis_images.length} />
          <Metric label="Adult read" value={intelligence.adult_content.nudity_level} />
          <Metric label="Signal confidence" value={`${intelligence.adult_content.confidence}%`} />
        </div>
      </div>

      <div className="analysis-section-grid">
        <section className="analysis-section">
          <SectionHeading eyebrow="Identity base" title="What should stay consistent" />
          <div className="intelligence-grid">
            <AttributeCard label="Face lock" detail={intelligence.face} />
            <AttributeCard label="Hair" detail={intelligence.hair} />
            <AttributeCard label="Body shape" detail={intelligence.body_shape} />
            <AttributeCard label="Waist and hips" detail={intelligence.waist_hips} />
          </div>
        </section>

        <section className="analysis-section">
          <SectionHeading eyebrow="Body visibility" title="Adult-content signals" />
          <AdultSignalCard signals={intelligence.adult_content} />
        </section>
      </div>

      <section className="analysis-section">
        <SectionHeading eyebrow="Local tagger" title="Strongest detected tags" />
        {topTags.length ? <VisionTagCloud tags={topTags} /> : <p className="empty-analysis-note">No strong local tags reported.</p>}
      </section>

      {intelligence.uncertainty_notes.length ? (
        <section className="analysis-section analysis-warnings">
          <SectionHeading eyebrow="Reference coverage" title="Signals to improve" />
          <WarningList warnings={intelligence.uncertainty_notes} />
        </section>
      ) : (
        <section className="analysis-section analysis-ready">
          <SectionHeading eyebrow="Reference coverage" title="Pack looks usable" />
          <p>The selected references provide enough signal for this character blueprint. Add targeted references only when you want to lock a very specific detail.</p>
        </section>
      )}

      <details className="image-review-section" open>
        <summary>
          <span>Image-by-image review</span>
          <strong>{analysis.analysis_images.length} references</strong>
        </summary>
        <div className="analysis-image-grid">
          {analysis.analysis_images.map((image) => (
            <article className="analysis-image-card" key={image.path}>
              <div className="analysis-image-header">
                <img src={api.thumbnailUrl(image.path)} alt="" />
                <div>
                  <strong>{image.file_name}</strong>
                  <span>{image.width} x {image.height} · {image.orientation}</span>
                </div>
              </div>
              {image.caption ? <p className="image-caption">{image.caption}</p> : null}
              <div className="image-signal-list">
                <SignalPill label="Coverage" value={image.body_attributes.coverage} />
                <SignalPill label="Chest" value={image.body_attributes.chest_visibility} />
                <SignalPill label="Genitals" value={image.adult_content.genital_visibility} />
                <SignalPill label="Framing" value={image.body_attributes.pose_framing} />
                <SignalPill label="Confidence" value={`${image.body_attributes.confidence}%`} />
              </div>
              {image.vision_tags.length ? <VisionTagCloud tags={image.vision_tags.slice(0, 8)} compact /> : null}
              <p className="cue-evidence">{image.body_attributes.evidence}</p>
            </article>
          ))}
        </div>
      </details>

      <div className="analysis-body">
        {analysis.analysis_warnings.length ? <WarningList warnings={analysis.analysis_warnings} /> : null}
      </div>
    </section>
  );
}

function SectionHeading({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div className="section-heading">
      <span>{eyebrow}</span>
      <h4>{title}</h4>
    </div>
  );
}

function SignalPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="signal-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function VisionTagCloud({ tags, compact = false }: { tags: VisionTag[]; compact?: boolean }) {
  return (
    <div className={compact ? "vision-tag-cloud compact" : "vision-tag-cloud"}>
      {tags.slice(0, compact ? 8 : 14).map((tag) => (
        <span key={`${tag.name}-${tag.confidence}`} title={tag.source}>
          {tag.name} <strong>{tag.confidence}%</strong>
        </span>
      ))}
    </div>
  );
}

function AttributeCard({ label, detail }: { label: string; detail: AttributeDetail }) {
  return (
    <article className="attribute-card">
      <div>
        <strong>{label}</strong>
        <span>{detail.visibility} · {detail.confidence}%</span>
      </div>
      <p>{detail.summary}</p>
      <small>{detail.evidence}</small>
    </article>
  );
}

function AdultSignalCard({ signals }: { signals: AdultContentSignals }) {
  return (
    <article className="adult-signal-card">
      <div>
        <h4>Adult-content visibility</h4>
        <span>{signals.confidence}% confidence</span>
      </div>
      <dl>
        <div><dt>Nudity</dt><dd>{signals.nudity_level}</dd></div>
        <div><dt>Breasts</dt><dd>{signals.breast_visibility}</dd></div>
        <div><dt>Nipple/areola</dt><dd>{signals.nipple_areola_visibility}</dd></div>
        <div><dt>Genitals</dt><dd>{signals.genital_visibility}</dd></div>
        <div><dt>Buttocks</dt><dd>{signals.buttocks_visibility}</dd></div>
        <div><dt>Sexual activity</dt><dd>{signals.sexual_activity}</dd></div>
      </dl>
      <p>{signals.evidence}</p>
    </article>
  );
}

function Textarea({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label>
      {label}
      <textarea value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function WarningList({ warnings }: { warnings: string[] }) {
  return (
    <ul className="warning-list">
      {warnings.map((warning) => <li key={warning}>{warning}</li>)}
    </ul>
  );
}

export default App;
