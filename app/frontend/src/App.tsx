import { Activity, AlertTriangle, Camera, CheckCircle2, ChevronDown, ChevronRight, ClipboardList, Cpu, FolderOpen, Play, Plus, RefreshCcw, Save, Scissors, Search, UserRound, X } from "lucide-react";
import React from "react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type {
  AdultAgeCategory,
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

type TabId = "runtime" | "characters" | "generate" | "training" | "prep";
type TrainingPreset = "fast" | "balanced" | "high_quality";
type TrackedGenerationJob = GenerationJobStatus & { recipe: PromptRecipe };

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

const tabs: Array<{ id: TabId; label: string; icon: typeof Activity }> = [
  { id: "runtime", label: "Runtime status", icon: Cpu },
  { id: "characters", label: "Characters", icon: UserRound },
  { id: "generate", label: "Generate", icon: Camera },
  { id: "training", label: "Training", icon: ClipboardList },
  { id: "prep", label: "Dataset prep", icon: Scissors }
];

const ageCategories: AdultAgeCategory[] = ["adult_18_plus", "adult_21_plus", "adult_25_plus", "adult_30_plus"];
const generationModes: GenerationMode[] = ["portrait", "full_body", "lifestyle_post", "studio", "reference_match"];
const qualityPresets: QualityPreset[] = ["fast", "balanced", "high", "ultra"];
const datasetTypes: DatasetType[] = ["body_part", "body_shape", "pose", "style", "fictional_face_identity"];
const facePolicies: FacePolicy[] = ["reject_faces", "redact_faces", "body_part_crops_only"];
const prepTargets: DatasetPrepTarget[] = ["chest_detail", "upper_torso", "full_body_context"];
const trainingPresets: Array<{ id: TrainingPreset; label: string; description: string }> = [
  { id: "balanced", label: "Balanced", description: "Best default for reusable global packs." },
  { id: "fast", label: "Fast", description: "Quick test pass with fewer steps." },
  { id: "high_quality", label: "High quality", description: "Slower, stronger adapter training." }
];

function LiidarLogo() {
  return (
    <svg className="brand-logo" viewBox="0 0 32 32" aria-hidden="true">
      <path className="brand-logo-beam" d="M10 6v20M10 26h11" />
      <path className="brand-logo-scan" d="M15 9h10M15 16h7M15 23h10" />
      <circle cx="7" cy="10" r="1.8" />
      <circle cx="24" cy="16" r="1.8" />
      <circle cx="12" cy="22" r="1.8" />
    </svg>
  );
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

function datasetTypeLabel(value: DatasetType) {
  const labels: Record<DatasetType, string> = {
    body_part: "Body detail knowledge",
    body_shape: "Body proportions",
    pose: "Pose library",
    style: "Photo realism / style",
    fictional_face_identity: "Character identity adapter"
  };
  return labels[value];
}

function prepTargetLabel(value: DatasetPrepTarget) {
  const labels: Record<DatasetPrepTarget, string> = {
    chest_detail: "Chest detail",
    upper_torso: "Upper torso",
    full_body_context: "Full-body context"
  };
  return labels[value];
}

function trainingPresetSettings(preset: TrainingPreset) {
  if (preset === "fast") {
    return { resolution: 640, max_train_steps: 450, learning_rate: 0.0001, network_dim: 16, network_alpha: 8, repeats: 4 };
  }
  if (preset === "high_quality") {
    return { resolution: 896, max_train_steps: 1200, learning_rate: 0.00008, network_dim: 32, network_alpha: 16, repeats: 8 };
  }
  return { resolution: 768, max_train_steps: 900, learning_rate: 0.0001, network_dim: 32, network_alpha: 16, repeats: 6 };
}

function packNameFromDataset(name: string, type: DatasetType) {
  const base = name.trim() || `global-${type}`;
  return base.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "global_improvement_pack";
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
  const [activeTab, setActiveTab] = useState<TabId>("runtime");
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<SystemLiveMetrics | null>(null);
  const [characters, setCharacters] = useState<CharacterProfile[]>([]);
  const [selectedCharacterId, setSelectedCharacterId] = useState("");
  const [editingCharacter, setEditingCharacter] = useState<CharacterProfile>(blankCharacter);
  const [recipe, setRecipe] = useState<PromptRecipe | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [mode, setMode] = useState<GenerationMode>("portrait");
  const [quality, setQuality] = useState<QualityPreset>("balanced");
  const [scenePrompt, setScenePrompt] = useState("soft natural light, believable still photo");
  const [extraNegative, setExtraNegative] = useState("");
  const [generationJobs, setGenerationJobs] = useState<TrackedGenerationJob[]>([]);

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
  const [trainingPreset, setTrainingPreset] = useState<TrainingPreset>("balanced");
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

  async function loadInitialData() {
    try {
      setError("");
      const [runtimeStatus, profiles, jobs, runs] = await Promise.all([
        api.runtime(),
        api.characters(),
        api.trainingJobs(),
        api.trainingRuns()
      ]);
      setRuntime(runtimeStatus);
      setCharacters(profiles);
      setTrainingJobs(jobs);
      setTrainingRuns(runs);
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
      if (profiles[0]) {
        setSelectedCharacterId(profiles[0].id);
        setEditingCharacter(profiles[0]);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load studio data.");
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
      setMessage("Character saved.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save character.");
    }
  }

  function generationPayload() {
    return {
      character_id: selectedCharacter?.id ?? "",
      mode,
      quality,
      scene_prompt: scenePrompt,
      extra_negative: extraNegative,
      seed: null
    };
  }

  async function previewRecipe() {
    try {
      setError("");
      setRecipe(await api.previewGeneration(generationPayload()));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to preview recipe.");
    }
  }

  async function queueGeneration() {
    try {
      setError("");
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
      setMessage(`Queued still photo job ${result.prompt_id}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to queue generation.");
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
        output_folder: prepOutputFolder.trim() ? prepOutputFolder : null,
        target: prepTarget,
        recursive: prepRecursive,
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
      const preset = trainingPresetSettings(trainingPreset);
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
        target: "chest_detail",
        recursive: true,
        scan_mode: "local",
        preset: trainingPreset,
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
          </div>
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
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p>Local generation</p>
            <h1>{tabs.find((tab) => tab.id === activeTab)?.label}</h1>
          </div>
          <button type="button" className="icon-button" onClick={loadInitialData} aria-label="Refresh studio data">
            <RefreshCcw aria-hidden="true" />
          </button>
        </header>

        {error ? <div className="notice error">{error}</div> : null}
        {message ? <div className="notice success">{message}</div> : null}

        {activeTab === "runtime" ? <RuntimePanel runtime={runtime} /> : null}
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
            scenePrompt={scenePrompt}
            extraNegative={extraNegative}
            recipe={recipe}
            generationJobs={generationJobs}
            onCharacterChange={setSelectedCharacterId}
            onModeChange={setMode}
            onQualityChange={setQuality}
            onScenePromptChange={setScenePrompt}
            onExtraNegativeChange={setExtraNegative}
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
            onDatasetNameChange={setDatasetName}
            onSourceFolderChange={setSourceFolder}
            onDatasetTypeChange={setDatasetType}
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
          />
        ) : null}
        {activeTab === "prep" ? (
          <DatasetPrepPanel
            sourceFolder={prepSourceFolder}
            outputFolder={prepOutputFolder}
            target={prepTarget}
            recursive={prepRecursive}
            report={prepReport}
            job={prepJob}
            isPreparing={isPreparingDataset}
            onSourceFolderChange={setPrepSourceFolder}
            onOutputFolderChange={setPrepOutputFolder}
            onTargetChange={setPrepTarget}
            onRecursiveChange={setPrepRecursive}
            onSelectSource={() => void selectPrepFolder("source")}
            onSelectOutput={() => void selectPrepFolder("output")}
            onRun={() => void runDatasetPrep()}
            onCancel={() => void cancelDatasetPrep()}
          />
        ) : null}
      </main>
    </div>
  );
}

function RuntimePanel({ runtime }: { runtime: RuntimeStatus | null }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Runtime status</h2>
        <p>Local hardware and ComfyUI readiness.</p>
      </div>
      <div className="status-grid">
        <Metric label="OS" value={runtime?.os_name ?? "Loading"} />
        <Metric label="Python" value={runtime?.python_version ?? "Loading"} />
        <Metric label="CPU" value={runtime?.cpu_name ?? "Loading"} />
        <Metric label="RAM" value={runtime ? `${runtime.total_ram_gb} GB` : "Loading"} />
        <Metric label="GPU" value={runtime?.gpu_names.join(", ") || "No GPU reported"} />
        <Metric label="AMD driver" value={runtime?.amd_driver_version ?? "Not detected"} />
      </div>
      <div className={`inline-status ${runtime?.comfyui_path_exists ? "ok" : "warning"}`}>
        {runtime?.comfyui_path_exists ? <CheckCircle2 aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
        <span>ComfyUI path {runtime?.comfyui_path_exists ? "found" : "not found"}</span>
      </div>
      {runtime?.warnings.length ? <WarningList warnings={runtime.warnings} /> : null}
    </section>
  );
}

function SystemMonitor({ metrics }: { metrics: SystemLiveMetrics | null }) {
  const gpuText = metrics?.gpu_percent == null ? "No counter" : `${Math.round(metrics.gpu_percent)}%`;
  const vramText = metrics?.gpu_memory_used_gb == null ? "Dedicated VRAM unavailable" : `${metrics.gpu_memory_used_gb.toFixed(2)} GB dedicated`;
  return (
    <section className="sidebar-monitor" aria-label="Live system monitor">
      <div className="monitor-heading">
        <span>Live system</span>
        <strong>{metrics ? "Online" : "Loading"}</strong>
      </div>
      <MonitorBar label="CPU" value={metrics?.cpu_percent ?? 0} text={metrics ? `${Math.round(metrics.cpu_percent)}%` : "Loading"} />
      <MonitorBar
        label="RAM"
        value={metrics?.ram_percent ?? 0}
        text={metrics ? `${metrics.ram_used_gb.toFixed(1)} / ${metrics.ram_total_gb.toFixed(1)} GB` : "Loading"}
      />
      <MonitorBar label="GPU" value={metrics?.gpu_percent ?? 0} text={gpuText} muted={metrics?.gpu_percent == null} />
      <div className="monitor-meta">
        <span>{vramText}</span>
        <span>Backend {metrics ? `${Math.round(metrics.process_memory_mb)} MB` : "..."}</span>
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
        <h2>Characters</h2>
        <button type="button" className="secondary-button rail-action" onClick={onNew}>
          <Plus aria-hidden="true" />
          New character
        </button>
        {characters.map((character) => (
          <button
            type="button"
            key={character.id}
            className={selectedCharacterId === character.id ? "row-button active" : "row-button"}
            onClick={() => onSelect(character.id)}
          >
            <strong>{character.display_name}</strong>
            <span>{labelize(character.age_category)}</span>
          </button>
        ))}
      </div>
      <form className="form-grid" onSubmit={(event) => { event.preventDefault(); onSave(); }}>
        <label>
          Display name
          <input value={editingCharacter.display_name} onChange={(event) => update("display_name", event.target.value)} />
        </label>
        <label>
          Age category
          <select value={editingCharacter.age_category} onChange={(event) => update("age_category", event.target.value as AdultAgeCategory)}>
            {ageCategories.map((age) => <option key={age} value={age}>{labelize(age)}</option>)}
          </select>
        </label>
        <ReferencePathModule paths={editingCharacter.reference_images} onChange={updateReferences} />
        {analysisError ? <div className="inline-error">{analysisError}</div> : null}
        {isAnalyzing ? <AnalysisProgress imageCount={editingCharacter.reference_images.length} /> : null}
        {currentAnalysis ? <ReferenceAnalysisPanel analysis={currentAnalysis} /> : <CharacterReadiness profile={editingCharacter} />}
        <div className="actions compact-actions">
          <button type="button" className="primary-button" onClick={() => void analyzeReferences()} disabled={!editingCharacter.reference_images.length || isAnalyzing}>
            <Search aria-hidden="true" />
            {isAnalyzing ? "Analyzing" : "Analyze references"}
          </button>
        </div>
        <details className="advanced-fields">
          <summary>Advanced profile controls</summary>
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
          Save character
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
  scenePrompt: string;
  extraNegative: string;
  recipe: PromptRecipe | null;
  generationJobs: TrackedGenerationJob[];
  onCharacterChange: (id: string) => void;
  onModeChange: (mode: GenerationMode) => void;
  onQualityChange: (quality: QualityPreset) => void;
  onScenePromptChange: (value: string) => void;
  onExtraNegativeChange: (value: string) => void;
  onPreview: () => void;
  onQueue: () => void;
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Generate believable still photos</h2>
        <p>Build an SDXL prompt recipe and queue a local image job.</p>
      </div>
      <div className="form-grid two-column">
        <label>
          Character
          <select value={props.selectedCharacterId} onChange={(event) => props.onCharacterChange(event.target.value)}>
            {props.characters.map((character) => <option key={character.id} value={character.id}>{character.display_name}</option>)}
          </select>
        </label>
        <label>
          Mode
          <select value={props.mode} onChange={(event) => props.onModeChange(event.target.value as GenerationMode)}>
            {generationModes.map((item) => <option key={item} value={item}>{labelize(item)}</option>)}
          </select>
        </label>
        <label>
          Quality
          <select value={props.quality} onChange={(event) => props.onQualityChange(event.target.value as QualityPreset)}>
            {qualityPresets.map((item) => <option key={item} value={item}>{labelize(item)}</option>)}
          </select>
        </label>
        <Textarea label="Scene prompt" value={props.scenePrompt} onChange={props.onScenePromptChange} />
        <Textarea label="Extra negative" value={props.extraNegative} onChange={props.onExtraNegativeChange} />
      </div>
      <div className="actions">
        <button type="button" className="secondary-button" onClick={props.onPreview}>
          <Search aria-hidden="true" />
          Preview recipe
        </button>
        <button type="button" className="primary-button" onClick={props.onQueue}>
          <Play aria-hidden="true" />
          Queue generation
        </button>
      </div>
      {props.recipe ? (
        <div className="recipe-box">
          <h3>Preview recipe</h3>
          <p>{props.recipe.positive}</p>
          <dl>
            <div><dt>Negative</dt><dd>{props.recipe.negative}</dd></div>
            <div><dt>Seed</dt><dd>{props.recipe.seed}</dd></div>
            <div><dt>Size</dt><dd>{props.recipe.width} x {props.recipe.height}</dd></div>
            <div><dt>Steps</dt><dd>{props.recipe.steps}</dd></div>
            <div><dt>CFG</dt><dd>{props.recipe.cfg}</dd></div>
          </dl>
        </div>
      ) : null}
      <GenerationJobsPanel jobs={props.generationJobs} />
    </section>
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

function DatasetPrepPanel(props: {
  sourceFolder: string;
  outputFolder: string;
  target: DatasetPrepTarget;
  recursive: boolean;
  report: DatasetPrepResponse | null;
  job: DatasetPrepJobStatus | null;
  isPreparing: boolean;
  onSourceFolderChange: (value: string) => void;
  onOutputFolderChange: (value: string) => void;
  onTargetChange: (value: DatasetPrepTarget) => void;
  onRecursiveChange: (value: boolean) => void;
  onSelectSource: () => void;
  onSelectOutput: () => void;
  onRun: () => void;
  onCancel: () => void;
}) {
  const canRun = Boolean(props.sourceFolder.trim()) && !props.isPreparing;
  const progressTotal = props.job?.total_count ?? 0;
  const progressProcessed = props.job?.processed_count ?? 0;
  const progressPercent = progressTotal > 0 ? Math.round((progressProcessed / progressTotal) * 100) : props.isPreparing ? 4 : 0;
  const isCancelable = Boolean(props.job && ["queued", "running", "cancelling"].includes(props.job.status));
  const aiModeText = "Local AI scan is active. It uses the local detector and writes only crops with a usable detected target.";
  return (
    <section className="training-layout dataset-prep-layout">
      <div className="panel training-overview">
        <div className="panel-heading">
          <h2>Dataset prep</h2>
          <p>Crop raw folders into cleaner training inputs before creating a global pack.</p>
        </div>
        <div className="prep-note">
          <strong>Detection mode</strong>
          <span>{aiModeText}</span>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <h2>Crop setup</h2>
          <p>Output folder is optional. If blank, crops are written to Desktop/Liidar_Dataset_Crops.</p>
        </div>
        <div className="form-grid two-column">
          <div className="field-block">
            <label htmlFor="prep-source-folder">Source folder</label>
            <div className="folder-picker-row">
              <input id="prep-source-folder" value={props.sourceFolder} onChange={(event) => props.onSourceFolderChange(event.target.value)} placeholder="Select raw image folder" />
              <button type="button" className="secondary-button inline-button" onClick={props.onSelectSource}>
                <FolderOpen aria-hidden="true" />
                Select folder
              </button>
            </div>
          </div>
          <div className="field-block">
            <label htmlFor="prep-output-folder">Output folder</label>
            <div className="folder-picker-row">
              <input id="prep-output-folder" value={props.outputFolder} onChange={(event) => props.onOutputFolderChange(event.target.value)} placeholder="Desktop default" />
              <button type="button" className="secondary-button inline-button" onClick={props.onSelectOutput}>
                <FolderOpen aria-hidden="true" />
                Select folder
              </button>
            </div>
          </div>
          <label>
            Crop target
            <select value={props.target} onChange={(event) => props.onTargetChange(event.target.value as DatasetPrepTarget)}>
              {prepTargets.map((target) => <option key={target} value={target}>{prepTargetLabel(target)}</option>)}
            </select>
          </label>
          <label className="check-row">
            <input type="checkbox" checked={props.recursive} onChange={(event) => props.onRecursiveChange(event.target.checked)} />
            Include subfolders
          </label>
        </div>
        <section className="api-key-panel">
          <div className="panel-heading">
            <h3>Detection mode</h3>
            <p>Dataset prep always uses the local scanner on this PC.</p>
          </div>
          <div className="inline-status ok">
            <CheckCircle2 aria-hidden="true" />
            <span>Local AI scanning runs on this PC.</span>
          </div>
        </section>
        <div className="actions compact-actions">
          <button type="button" className="primary-button" onClick={props.onRun} disabled={!canRun}>
            <Scissors aria-hidden="true" />
            {props.isPreparing ? "Preparing crops" : "Create crop folder"}
          </button>
          {isCancelable ? (
            <button type="button" className="secondary-button danger-button" onClick={props.onCancel} disabled={props.job?.status === "cancelling"}>
              <X aria-hidden="true" />
              {props.job?.status === "cancelling" ? "Cancelling" : "Cancel"}
            </button>
          ) : null}
        </div>
        {props.job ? (
          <section className="prep-progress-panel">
            <div className="prep-progress-header">
              <strong>{labelize(props.job.status)}</strong>
              <span>{progressProcessed} / {progressTotal || "?"} images</span>
            </div>
            <div className="progress-track" aria-label="Dataset prep progress" role="progressbar" aria-valuenow={progressPercent} aria-valuemin={0} aria-valuemax={100}>
              <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }} />
            </div>
            <div className="prep-progress-meta">
              <span>
                Local AI scan: {props.job.cropped_count} crops saved, {props.job.skipped_count} skipped
              </span>
              <span>{props.job.fallback_count} fallback crop{props.job.fallback_count === 1 ? "" : "s"}</span>
              {props.job.active_file ? <span>Current: {props.job.active_file}</span> : null}
              {props.job.error ? <span className="error-text">{props.job.error}</span> : null}
            </div>
          </section>
        ) : null}
      </div>

      {props.report ? (
        <div className="panel">
          <div className="panel-heading">
            <h2>Prep result</h2>
            <p>{props.report.output_folder}</p>
          </div>
          <div className="count-grid">
            <Metric label="Processed" value={props.report.processed_count} />
            <Metric label="Cropped" value={props.report.cropped_count} />
            <Metric label="Skipped" value={props.report.skipped_count} />
            <Metric label="Output folder" value={props.report.output_folder} />
          </div>
          {props.report.warnings.length ? <WarningList warnings={props.report.warnings} /> : null}
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
  onDatasetNameChange: (value: string) => void;
  onSourceFolderChange: (value: string) => void;
  onDatasetTypeChange: (value: DatasetType) => void;
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
  const setupStatus = runIsActive
    ? "Training is running"
    : setupSaved
      ? `${globalPackName} is ready to train`
      : "Save the setup before training";

  return (
    <section className="training-layout global-training">
      <div className="panel training-simple-panel">
        <div className="panel-heading">
          <h2>Train a global pack</h2>
          <p>Select prepared crops, name what this pack teaches, then let the app handle the trainer.</p>
        </div>

        <div className="training-step-strip compact">
          <div className={hasTrainingFolder ? "active" : ""}><strong>1</strong><span>Setup</span></div>
          <div className={setupSaved || runIsActive ? "active" : ""}><strong>2</strong><span>Train</span></div>
          <div className={latestCompletedPack ? "active" : ""}><strong>3</strong><span>Active pack</span></div>
        </div>

        <div className="training-simple-grid">
          <div className="training-field-stack">
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
            <label>
              What this teaches
              <select
                value={props.datasetType}
                disabled={runIsActive}
                onChange={(event) => props.onDatasetTypeChange(event.target.value as DatasetType)}
              >
                {datasetTypes.map((item) => <option key={item} value={item}>{datasetTypeLabel(item)}</option>)}
              </select>
            </label>
          </div>

          <div className="training-folder-card">
            <span>Prepared crop folder</span>
            <div className="folder-picker-row">
              <input
                id="training-folder"
                value={trainingFolder}
                disabled={runIsActive}
                onChange={(event) => {
                  props.onSourceFolderChange(event.target.value);
                  props.onDatasetPathChange(event.target.value);
                }}
                placeholder="Select a crop folder created in Dataset prep"
              />
              <button type="button" className="secondary-button inline-button" onClick={props.onSelectSourceFolder} disabled={runIsActive}>
                <FolderOpen aria-hidden="true" />
                Select folder
              </button>
            </div>
            <small>{props.acceptedImageCount ? `${props.acceptedImageCount} images ready` : "Image count is checked when setup is saved."}</small>
          </div>
        </div>

        <div className="preset-grid compact-presets" role="radiogroup" aria-label="Training quality">
          {trainingPresets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              role="radio"
              aria-checked={props.trainingPreset === preset.id}
              className={props.trainingPreset === preset.id ? "preset-card active" : "preset-card"}
              disabled={runIsActive}
              onClick={() => props.onTrainingPresetChange(preset.id)}
            >
              <strong>{preset.label}</strong>
              <span>{preset.description}</span>
            </button>
          ))}
        </div>

        <div className="training-action-card">
          <div>
            <span>Status</span>
            <strong>{setupStatus}</strong>
            <small>Output pack: {globalPackName}.safetensors</small>
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

        {activeRun ? <TrainingRunPanel run={activeRun} /> : null}

        <div className="training-library-strip">
          <span>{completedGlobalPacks.length} active pack{completedGlobalPacks.length === 1 ? "" : "s"}</span>
          <span>{pendingGlobalJobs.length} saved setup{pendingGlobalJobs.length === 1 ? "" : "s"}</span>
          <span>{latestCompletedPack ? `Latest: ${latestCompletedPack.lora_name}` : "No finished pack yet"}</span>
        </div>

        <details className="training-diagnostics">
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
      <details className="run-path-details">
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
