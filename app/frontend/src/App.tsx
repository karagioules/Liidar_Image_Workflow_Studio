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
  GenerationMode,
  PromptRecipe,
  QualityPreset,
  ReferenceAnalysisResponse,
  RuntimeStatus,
  SystemLiveMetrics,
  TrainerStatus,
  TrainingJobConfig,
  VisionTag
} from "./types";

type TabId = "runtime" | "characters" | "generate" | "training" | "prep";
type TrainingPreset = "fast" | "balanced" | "high_quality";

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
    return { max_train_steps: 600, learning_rate: 0.0001, network_dim: 16, network_alpha: 8, repeats: 6 };
  }
  if (preset === "high_quality") {
    return { max_train_steps: 1800, learning_rate: 0.00008, network_dim: 64, network_alpha: 32, repeats: 12 };
  }
  return { max_train_steps: 1200, learning_rate: 0.0001, network_dim: 32, network_alpha: 16, repeats: 10 };
}

function packNameFromDataset(name: string, type: DatasetType) {
  const base = name.trim() || `global-${type}`;
  return base.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "global_improvement_pack";
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

  const [datasetName, setDatasetName] = useState("global-body-pack");
  const [sourceFolder, setSourceFolder] = useState("");
  const [datasetType, setDatasetType] = useState<DatasetType>("body_shape");
  const [facePolicy, setFacePolicy] = useState<FacePolicy>("reject_faces");
  const [scanReport, setScanReport] = useState<DatasetScanReport | null>(null);
  const [datasetPath, setDatasetPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("models/sdxl_base_1.0.safetensors");
  const [loraName, setLoraName] = useState("global_body_pack");
  const [outputDir, setOutputDir] = useState("outputs/global_lora");
  const [acceptedImageCount, setAcceptedImageCount] = useState(1);
  const [trainingPreset, setTrainingPreset] = useState<TrainingPreset>("balanced");
  const [trainingJob, setTrainingJob] = useState<TrainingJobConfig | null>(null);
  const [trainerEntrypoint, setTrainerEntrypoint] = useState("train_network.py");
  const [trainerConfigPath, setTrainerConfigPath] = useState("");
  const [trainerStatus, setTrainerStatus] = useState<TrainerStatus | null>(null);
  const [prepSourceFolder, setPrepSourceFolder] = useState("");
  const [prepOutputFolder, setPrepOutputFolder] = useState("");
  const [prepTarget, setPrepTarget] = useState<DatasetPrepTarget>("chest_detail");
  const [prepRecursive, setPrepRecursive] = useState(true);
  const [prepUseAi, setPrepUseAi] = useState(false);
  const [prepAiMaxImages, setPrepAiMaxImages] = useState(25);
  const [anthropicKeySaved, setAnthropicKeySaved] = useState(false);
  const [anthropicKeyInput, setAnthropicKeyInput] = useState("");
  const [anthropicModel, setAnthropicModel] = useState("claude-3-haiku-20240307");
  const [prepReport, setPrepReport] = useState<DatasetPrepResponse | null>(null);
  const [prepJob, setPrepJob] = useState<DatasetPrepJobStatus | null>(null);
  const [isPreparingDataset, setIsPreparingDataset] = useState(false);

  useEffect(() => {
    void loadInitialData();
  }, []);

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

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterId) ?? characters[0],
    [characters, selectedCharacterId]
  );

  async function loadInitialData() {
    try {
      setError("");
      const [runtimeStatus, profiles, keyStatus] = await Promise.all([api.runtime(), api.characters(), api.anthropicKeyStatus()]);
      setRuntime(runtimeStatus);
      setCharacters(profiles);
      setAnthropicKeySaved(keyStatus.saved);
      setAnthropicModel(keyStatus.model);
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
        use_ai: prepUseAi,
        ai_max_images: prepAiMaxImages,
        ai_model: anthropicModel
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

  async function saveAnthropicKey() {
    try {
      setError("");
      const status = await api.saveAnthropicKey(anthropicKeyInput, anthropicModel);
      setAnthropicKeySaved(status.saved);
      setAnthropicModel(status.model);
      setAnthropicKeyInput("");
      setMessage("Claude API key saved locally.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save Claude API key.");
    }
  }

  async function removeAnthropicKey() {
    try {
      setError("");
      const status = await api.deleteAnthropicKey();
      setAnthropicKeySaved(status.saved);
      setMessage("Claude API key removed.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to remove Claude API key.");
    }
  }

  async function createTrainingConfig() {
    try {
      setError("");
      const preset = trainingPresetSettings(trainingPreset);
      const job = await api.createTrainingConfig({
        request: {
          dataset_id: scanReport?.dataset_id || datasetName,
          dataset_path: datasetPath || sourceFolder,
          output_dir: outputDir,
          base_model_path: baseModelPath,
          lora_name: loraName || packNameFromDataset(datasetName, datasetType),
          dataset_type: datasetType,
          global_pack: true,
          resolution: 1024,
          repeats: preset.repeats,
          batch_size: 1,
          max_train_steps: preset.max_train_steps,
          learning_rate: preset.learning_rate,
          network_dim: preset.network_dim,
          network_alpha: preset.network_alpha
        },
        accepted_image_count: acceptedImageCount
      });
      setTrainingJob(job);
      setTrainerConfigPath(job.config_path ?? "");
      setMessage(`Global training job created for ${job.lora_name}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create training config.");
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
            onScan={scanDataset}
            onCreateConfig={createTrainingConfig}
            onCheckStatus={checkTrainerStatus}
          />
        ) : null}
        {activeTab === "prep" ? (
          <DatasetPrepPanel
            sourceFolder={prepSourceFolder}
            outputFolder={prepOutputFolder}
            target={prepTarget}
            recursive={prepRecursive}
            useAi={prepUseAi}
            aiMaxImages={prepAiMaxImages}
            anthropicKeySaved={anthropicKeySaved}
            anthropicKeyInput={anthropicKeyInput}
            anthropicModel={anthropicModel}
            report={prepReport}
            job={prepJob}
            isPreparing={isPreparingDataset}
            onSourceFolderChange={setPrepSourceFolder}
            onOutputFolderChange={setPrepOutputFolder}
            onTargetChange={setPrepTarget}
            onRecursiveChange={setPrepRecursive}
            onUseAiChange={setPrepUseAi}
            onAiMaxImagesChange={setPrepAiMaxImages}
            onAnthropicKeyInputChange={setAnthropicKeyInput}
            onSaveAnthropicKey={() => void saveAnthropicKey()}
            onRemoveAnthropicKey={() => void removeAnthropicKey()}
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
    </section>
  );
}

function DatasetPrepPanel(props: {
  sourceFolder: string;
  outputFolder: string;
  target: DatasetPrepTarget;
  recursive: boolean;
  useAi: boolean;
  aiMaxImages: number;
  anthropicKeySaved: boolean;
  anthropicKeyInput: string;
  anthropicModel: string;
  report: DatasetPrepResponse | null;
  job: DatasetPrepJobStatus | null;
  isPreparing: boolean;
  onSourceFolderChange: (value: string) => void;
  onOutputFolderChange: (value: string) => void;
  onTargetChange: (value: DatasetPrepTarget) => void;
  onRecursiveChange: (value: boolean) => void;
  onUseAiChange: (value: boolean) => void;
  onAiMaxImagesChange: (value: number) => void;
  onAnthropicKeyInputChange: (value: string) => void;
  onSaveAnthropicKey: () => void;
  onRemoveAnthropicKey: () => void;
  onSelectSource: () => void;
  onSelectOutput: () => void;
  onRun: () => void;
  onCancel: () => void;
}) {
  const canRun = Boolean(props.sourceFolder.trim()) && !props.isPreparing;
  const previewImages = props.report?.images.filter((image) => image.accepted).slice(0, 12) ?? [];
  const progressTotal = props.job?.total_count ?? 0;
  const progressProcessed = props.job?.processed_count ?? 0;
  const progressPercent = progressTotal > 0 ? Math.round((progressProcessed / progressTotal) * 100) : props.isPreparing ? 4 : 0;
  const isCancelable = Boolean(props.job && ["queued", "running", "cancelling"].includes(props.job.status));
  const aiModeText = props.useAi
    ? props.anthropicKeySaved
      ? `Claude AI enabled for up to ${props.aiMaxImages} image${props.aiMaxImages === 1 ? "" : "s"}; the rest use local cropping.`
      : "Claude AI is selected but no saved key is available."
    : "Claude AI is off. This run will use local cropping only.";
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
            <h3>Claude AI scan</h3>
            <p>Optional. Uses Claude 3 Haiku by default for low-cost vision checks.</p>
          </div>
          <div className="form-grid two-column">
            <label>
              API key
              <input
                type="password"
                value={props.anthropicKeyInput}
                onChange={(event) => props.onAnthropicKeyInputChange(event.target.value)}
                placeholder={props.anthropicKeySaved ? "Saved key active" : "Paste Anthropic API key"}
              />
            </label>
            <label>
              Max AI images
              <input
                type="number"
                min="0"
                max="500"
                value={props.aiMaxImages}
                onChange={(event) => props.onAiMaxImagesChange(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="actions compact-actions">
            <button type="button" className="secondary-button" onClick={props.onSaveAnthropicKey} disabled={!props.anthropicKeyInput.trim()}>
              <Save aria-hidden="true" />
              Save key
            </button>
            <button type="button" className="secondary-button" onClick={props.onRemoveAnthropicKey} disabled={!props.anthropicKeySaved}>
              <X aria-hidden="true" />
              Remove key
            </button>
            <label className="check-row inline-check">
              <input type="checkbox" checked={props.useAi} onChange={(event) => props.onUseAiChange(event.target.checked)} disabled={!props.anthropicKeySaved} />
              Use AI scan
            </label>
          </div>
          <div className={props.anthropicKeySaved ? "inline-status ok" : "inline-status warning"}>
            {props.anthropicKeySaved ? <CheckCircle2 aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
            <span>{props.anthropicKeySaved ? `Claude key saved. Model: ${props.anthropicModel}` : "No Claude key saved."}</span>
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
              <span>{props.job.use_ai ? `AI scan on: ${props.job.ai_guided_count} AI-guided crop${props.job.ai_guided_count === 1 ? "" : "s"}` : "AI scan off: local cropping only"}</span>
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
            <Metric label="AI-guided" value={props.report.ai_guided_count} />
            <Metric label="Face-guided" value={props.report.face_guided_count} />
            <Metric label="Fallback" value={props.report.fallback_count} />
          </div>
          {props.report.warnings.length ? <WarningList warnings={props.report.warnings} /> : null}
          {previewImages.length ? (
            <div className="prep-preview-grid">
              {previewImages.map((image) => (
                <article className="prep-preview-card" key={image.output_path ?? image.source_path}>
                  {image.output_path ? <img src={api.thumbnailUrl(image.output_path)} alt="" /> : null}
                  <div>
                    <strong>{image.method === "ai_guided" ? "AI-guided crop" : image.method === "face_guided" ? "Face-guided crop" : "Fallback crop"}</strong>
                    <span>{image.reason}</span>
                  </div>
                </article>
              ))}
            </div>
          ) : null}
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
  onScan: () => void;
  onCreateConfig: () => void;
  onCheckStatus: () => void;
}) {
  const hasAcceptedImages = Boolean(props.scanReport && props.scanReport.accepted_count > 0);
  const globalPackName = props.loraName || packNameFromDataset(props.datasetName, props.datasetType);

  return (
    <section className="training-layout global-training">
      <div className="panel training-overview">
        <div className="panel-heading">
          <h2>Global improvement training</h2>
          <p>Reusable training packs that can improve future generations across the studio.</p>
        </div>
        <div className="training-step-strip">
          <div className="active"><strong>1</strong><span>Choose data</span></div>
          <div className={props.scanReport ? "active" : ""}><strong>2</strong><span>Review scan</span></div>
          <div className={props.trainingJob ? "active" : ""}><strong>3</strong><span>Create global job</span></div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <h2>1. Data pack</h2>
          <p>Pick the folder and the kind of global knowledge this pack should add.</p>
        </div>
        <div className="form-grid two-column">
          <label>
            Pack name
            <input value={props.datasetName} onChange={(event) => props.onDatasetNameChange(event.target.value)} />
          </label>
          <div className="field-block">
            <label htmlFor="training-folder">Training folder</label>
            <div className="folder-picker-row">
              <input
                id="training-folder"
                value={props.sourceFolder}
                onChange={(event) => props.onSourceFolderChange(event.target.value)}
                placeholder="Select a local folder"
              />
              <button type="button" className="secondary-button inline-button" onClick={props.onSelectSourceFolder}>
                <FolderOpen aria-hidden="true" />
                Select folder
              </button>
            </div>
          </div>
          <label>
            Improvement type
            <select value={props.datasetType} onChange={(event) => props.onDatasetTypeChange(event.target.value as DatasetType)}>
              {datasetTypes.map((item) => <option key={item} value={item}>{datasetTypeLabel(item)}</option>)}
            </select>
          </label>
        </div>
        <button type="button" className="primary-button" onClick={props.onScan}>
          <Search aria-hidden="true" />
          Scan data pack
        </button>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <h2>2. Scan review</h2>
          <p>Only accepted images are used for the global pack.</p>
        </div>
        <div className="count-grid">
          <Metric label="Accepted" value={props.scanReport?.accepted_count ?? 0} />
          <Metric label="Rejected" value={props.scanReport?.rejected_count ?? 0} />
          <Metric label="Duplicate" value={props.scanReport?.duplicate_count ?? 0} />
          <Metric label="Ignored" value={props.scanReport?.ignored_count ?? 0} />
        </div>
        {props.scanReport?.warnings.length ? <WarningList warnings={props.scanReport.warnings} /> : null}
      </div>

      <div className="panel">
        <div className="panel-heading">
          <h2>3. Global training job</h2>
          <p>Creates a reusable adapter pack for current and future generations.</p>
        </div>
        <div className="preset-grid" role="radiogroup" aria-label="Training preset">
          {trainingPresets.map((preset) => (
            <button
              key={preset.id}
              type="button"
              role="radio"
              aria-checked={props.trainingPreset === preset.id}
              className={props.trainingPreset === preset.id ? "preset-card active" : "preset-card"}
              onClick={() => props.onTrainingPresetChange(preset.id)}
            >
              <strong>{preset.label}</strong>
              <span>{preset.description}</span>
            </button>
          ))}
        </div>
        <div className="training-job-summary">
          <Metric label="Pack" value={globalPackName} />
          <Metric label="Images ready" value={props.scanReport?.accepted_count ?? 0} />
          <Metric label="Preset" value={trainingPresets.find((preset) => preset.id === props.trainingPreset)?.label ?? "Balanced"} />
        </div>
        <button type="button" className="primary-button" onClick={props.onCreateConfig} disabled={!hasAcceptedImages}>
          <Save aria-hidden="true" />
          Create global training job
        </button>
        {props.trainingJob ? <div className="inline-status"><CheckCircle2 aria-hidden="true" /><span>Global pack job {props.trainingJob.job_id}</span></div> : null}
      </div>

      <details className="panel advanced-training">
        <summary>Advanced trainer details</summary>
        <div className="panel-heading advanced-panel-heading">
          <h2>Trainer configuration</h2>
          <p>Technical paths and safety policy. Defaults are used unless changed here.</p>
        </div>
        <div className="form-grid two-column">
          <label>
            Dataset path
            <input value={props.datasetPath} onChange={(event) => props.onDatasetPathChange(event.target.value)} />
          </label>
          <label>
            Base model path
            <input value={props.baseModelPath} onChange={(event) => props.onBaseModelPathChange(event.target.value)} />
          </label>
          <label>
            LoRA name
            <input value={props.loraName} onChange={(event) => props.onLoraNameChange(event.target.value)} />
          </label>
          <label>
            Output dir
            <input value={props.outputDir} onChange={(event) => props.onOutputDirChange(event.target.value)} />
          </label>
          <label>
            Accepted image count
            <input
              type="number"
              min="1"
              value={props.acceptedImageCount}
              onChange={(event) => props.onAcceptedImageCountChange(Number(event.target.value))}
            />
          </label>
          <label>
            Face policy
            <select value={props.facePolicy} onChange={(event) => props.onFacePolicyChange(event.target.value as FacePolicy)}>
              {facePolicies.map((item) => <option key={item} value={item}>{labelize(item)}</option>)}
            </select>
          </label>
          <label>
            Trainer entrypoint
            <input value={props.trainerEntrypoint} onChange={(event) => props.onTrainerEntrypointChange(event.target.value)} />
          </label>
          <label>
            Config path
            <input value={props.trainerConfigPath} onChange={(event) => props.onTrainerConfigPathChange(event.target.value)} />
          </label>
        </div>
        <button type="button" className="secondary-button" onClick={props.onCheckStatus}>
          <Activity aria-hidden="true" />
          Check status
        </button>
        {props.trainerStatus ? (
          <>
            <div className="status-grid compact">
              <Metric label="Entrypoint" value={props.trainerStatus.trainer_entrypoint_exists ? "Found" : "Missing"} />
              <Metric label="Config" value={props.trainerStatus.config_path_exists ? "Found" : "Missing"} />
            </div>
            {props.trainerStatus.warnings.length ? <WarningList warnings={props.trainerStatus.warnings} /> : null}
          </>
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
