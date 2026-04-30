import { Activity, AlertTriangle, Camera, CheckCircle2, ClipboardList, Cpu, FileImage, Folder, FolderOpen, Play, Plus, RefreshCcw, Save, Search, SlidersHorizontal, UserRound, X } from "lucide-react";
import React from "react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type {
  AdultAgeCategory,
  CharacterProfile,
  DatasetScanReport,
  DatasetType,
  FacePolicy,
  GenerationMode,
  PathBrowserResponse,
  PromptRecipe,
  QualityPreset,
  RuntimeStatus,
  SourceRights,
  TrainerStatus,
  TrainingJobConfig
} from "./types";

type TabId = "runtime" | "characters" | "generate" | "training";

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

const tabs: Array<{ id: TabId; label: string; icon: typeof Activity }> = [
  { id: "runtime", label: "Runtime status", icon: Cpu },
  { id: "characters", label: "Characters", icon: UserRound },
  { id: "generate", label: "Generate", icon: Camera },
  { id: "training", label: "Training", icon: ClipboardList }
];

const ageCategories: AdultAgeCategory[] = ["adult_18_plus", "adult_21_plus", "adult_25_plus", "adult_30_plus"];
const generationModes: GenerationMode[] = ["portrait", "full_body", "lifestyle_post", "studio", "reference_match"];
const qualityPresets: QualityPreset[] = ["fast", "balanced", "high", "ultra"];
const datasetTypes: DatasetType[] = ["body_part", "body_shape", "pose", "style", "fictional_face_identity"];
const facePolicies: FacePolicy[] = ["reject_faces", "redact_faces", "body_part_crops_only"];
const sourceRights: SourceRights[] = ["synthetic", "owned", "licensed", "consented"];

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

function parentPath(path: string): string {
  const normalized = path.trim();
  const separatorIndex = Math.max(normalized.lastIndexOf("\\"), normalized.lastIndexOf("/"));
  return separatorIndex > 0 ? normalized.slice(0, separatorIndex) : normalized;
}

function displayNameFromReferencePaths(paths: string[]): string {
  const firstPath = paths[0] ?? "";
  const parts = firstPath.split(/[\\/]+/).filter(Boolean);
  const fileName = parts[parts.length - 1] ?? "New character";
  const parent = parts[parts.length - 2];
  if (parent && !["models", "ai faces", "free_posts", "may2025"].includes(parent.toLowerCase())) {
    return parent.replace(/[_-]+/g, " ");
  }
  return fileName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
}

function draftProfileFromReferences(profile: CharacterProfile): CharacterProfile {
  const paths = profile.reference_images;
  const referenceCount = paths.length;
  const name = profile.display_name.trim() || displayNameFromReferencePaths(paths);
  const referenceNote = referenceCount > 1 ? `${referenceCount} reference images` : "1 reference image";
  return {
    ...profile,
    display_name: name,
    age_category: profile.age_category || "adult_25_plus",
    face_summary: `fictional adult face identity guided by ${referenceNote}; keep the same face structure, expression style, and natural skin texture`,
    hair: "match the reference images for hair color, length, volume, and styling",
    eyes: "match the reference images for eye shape and color",
    skin_tone: "match the reference images for natural skin tone and texture",
    body_shape: "match the reference images for believable adult body proportions and posture",
    chest: "match the reference images for natural adult body shape",
    grooming: "match the reference images; keep grooming realistic and consistent",
    style_notes: "reference-guided believable still photo style with natural camera lighting, casual realism, and no overpolished AI look",
    negative_notes: "plastic skin, airbrushed, overprocessed, uncanny symmetry, celebrity, real person, underage, childlike"
  };
}

function labelize(value: string) {
  return value.replace(/_/g, " ");
}

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("runtime");
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
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

  const [datasetName, setDatasetName] = useState("fictional-face-pack");
  const [sourceFolder, setSourceFolder] = useState("");
  const [datasetType, setDatasetType] = useState<DatasetType>("fictional_face_identity");
  const [facePolicy, setFacePolicy] = useState<FacePolicy>("reject_faces");
  const [rights, setRights] = useState<SourceRights>("synthetic");
  const [scanReport, setScanReport] = useState<DatasetScanReport | null>(null);
  const [datasetPath, setDatasetPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("models/sdxl_base_1.0.safetensors");
  const [loraName, setLoraName] = useState("local_identity_lora");
  const [outputDir, setOutputDir] = useState("outputs/lora");
  const [acceptedImageCount, setAcceptedImageCount] = useState(1);
  const [trainingJob, setTrainingJob] = useState<TrainingJobConfig | null>(null);
  const [trainerEntrypoint, setTrainerEntrypoint] = useState("train_network.py");
  const [trainerConfigPath, setTrainerConfigPath] = useState("");
  const [trainerStatus, setTrainerStatus] = useState<TrainerStatus | null>(null);

  useEffect(() => {
    void loadInitialData();
  }, []);

  useEffect(() => {
    const selected = characters.find((character) => character.id === selectedCharacterId);
    if (selected) {
      setEditingCharacter(selected);
    }
  }, [characters, selectedCharacterId]);

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterId) ?? characters[0],
    [characters, selectedCharacterId]
  );

  async function loadInitialData() {
    try {
      setError("");
      const [runtimeStatus, profiles] = await Promise.all([api.runtime(), api.characters()]);
      setRuntime(runtimeStatus);
      setCharacters(profiles);
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
        character_id: datasetType === "fictional_face_identity" ? selectedCharacter?.id ?? null : null,
        source_rights: datasetType === "fictional_face_identity" ? rights : null,
        tags: []
      });
      setScanReport(report);
      setDatasetPath(sourceFolder);
      setAcceptedImageCount(Math.max(1, report.accepted_count));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to scan dataset.");
    }
  }

  async function createTrainingConfig() {
    try {
      setError("");
      const job = await api.createTrainingConfig({
        request: {
          dataset_id: scanReport?.dataset_id || datasetName,
          dataset_path: datasetPath || sourceFolder,
          output_dir: outputDir,
          base_model_path: baseModelPath,
          lora_name: loraName,
          resolution: 1024,
          repeats: 10,
          batch_size: 1,
          max_train_steps: 1200,
          learning_rate: 0.0001,
          network_dim: 32,
          network_alpha: 16
        },
        accepted_image_count: acceptedImageCount
      });
      setTrainingJob(job);
      setTrainerConfigPath(job.config_path ?? "");
      setMessage(`Training config created for ${job.lora_name}.`);
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
          <SlidersHorizontal aria-hidden="true" />
          <div>
            <strong>Local Model Studio</strong>
            <span>AMD-first still photo workspace</span>
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
            characters={characters}
            selectedCharacterId={selectedCharacter?.id ?? ""}
            datasetName={datasetName}
            sourceFolder={sourceFolder}
            datasetType={datasetType}
            facePolicy={facePolicy}
            rights={rights}
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
            onCharacterChange={setSelectedCharacterId}
            onDatasetNameChange={setDatasetName}
            onSourceFolderChange={setSourceFolder}
            onDatasetTypeChange={setDatasetType}
            onFacePolicyChange={setFacePolicy}
            onRightsChange={setRights}
            onDatasetPathChange={setDatasetPath}
            onBaseModelPathChange={setBaseModelPath}
            onLoraNameChange={setLoraName}
            onOutputDirChange={setOutputDir}
            onAcceptedImageCountChange={setAcceptedImageCount}
            onTrainerEntrypointChange={setTrainerEntrypoint}
            onTrainerConfigPathChange={setTrainerConfigPath}
            onScan={scanDataset}
            onCreateConfig={createTrainingConfig}
            onCheckStatus={checkTrainerStatus}
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

function CharactersPanel(props: {
  characters: CharacterProfile[];
  selectedCharacterId: string;
  editingCharacter: CharacterProfile;
  onSelect: (id: string) => void;
  onChange: (profile: CharacterProfile) => void;
  onSave: () => void;
}) {
  const { characters, selectedCharacterId, editingCharacter, onSelect, onChange, onSave } = props;
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const update = (field: keyof CharacterProfile, value: string) => onChange({ ...editingCharacter, [field]: value });
  const updateReferences = (paths: string[]) => onChange({ ...editingCharacter, reference_images: uniquePaths(paths) });
  const createDraft = () => {
    if (!editingCharacter.reference_images.length) {
      return;
    }
    onChange(draftProfileFromReferences(editingCharacter));
  };
  const analyzeReferences = async () => {
    if (!editingCharacter.reference_images.length) {
      return;
    }
    try {
      setIsAnalyzing(true);
      setAnalysisError("");
      onChange(await api.analyzeReferences(editingCharacter));
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
        <div className="actions compact-actions">
          <button type="button" className="primary-button" onClick={() => void analyzeReferences()} disabled={!editingCharacter.reference_images.length || isAnalyzing}>
            <Search aria-hidden="true" />
            {isAnalyzing ? "Analyzing" : "Analyze references"}
          </button>
          <button type="button" className="secondary-button" onClick={createDraft} disabled={!editingCharacter.reference_images.length}>
            <Search aria-hidden="true" />
            Create draft from references
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

function TrainingPanel(props: {
  characters: CharacterProfile[];
  selectedCharacterId: string;
  datasetName: string;
  sourceFolder: string;
  datasetType: DatasetType;
  facePolicy: FacePolicy;
  rights: SourceRights;
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
  onCharacterChange: (id: string) => void;
  onDatasetNameChange: (value: string) => void;
  onSourceFolderChange: (value: string) => void;
  onDatasetTypeChange: (value: DatasetType) => void;
  onFacePolicyChange: (value: FacePolicy) => void;
  onRightsChange: (value: SourceRights) => void;
  onDatasetPathChange: (value: string) => void;
  onBaseModelPathChange: (value: string) => void;
  onLoraNameChange: (value: string) => void;
  onOutputDirChange: (value: string) => void;
  onAcceptedImageCountChange: (value: number) => void;
  onTrainerEntrypointChange: (value: string) => void;
  onTrainerConfigPathChange: (value: string) => void;
  onScan: () => void;
  onCreateConfig: () => void;
  onCheckStatus: () => void;
}) {
  const identityPack = props.datasetType === "fictional_face_identity";

  return (
    <section className="training-layout">
      <div className="panel">
        <div className="panel-heading">
          <h2>Dataset scan</h2>
          <p>Use a typed local path; the backend scans the folder.</p>
        </div>
        <div className="form-grid two-column">
          <label>
            Dataset name
            <input value={props.datasetName} onChange={(event) => props.onDatasetNameChange(event.target.value)} />
          </label>
          <label>
            Source folder path
            <input value={props.sourceFolder} onChange={(event) => props.onSourceFolderChange(event.target.value)} placeholder="H:\photos\training-pack" />
          </label>
          <label>
            Dataset type
            <select value={props.datasetType} onChange={(event) => props.onDatasetTypeChange(event.target.value as DatasetType)}>
              {datasetTypes.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>
            Face policy
            <select value={props.facePolicy} onChange={(event) => props.onFacePolicyChange(event.target.value as FacePolicy)}>
              {facePolicies.map((item) => <option key={item} value={item}>{labelize(item)}</option>)}
            </select>
          </label>
          {identityPack ? (
            <>
              <label>
                Character
                <select value={props.selectedCharacterId} onChange={(event) => props.onCharacterChange(event.target.value)}>
                  {props.characters.map((character) => <option key={character.id} value={character.id}>{character.display_name}</option>)}
                </select>
              </label>
              <label>
                Source rights
                <select value={props.rights} onChange={(event) => props.onRightsChange(event.target.value as SourceRights)}>
                  {sourceRights.map((item) => <option key={item} value={item}>{labelize(item)}</option>)}
                </select>
              </label>
            </>
          ) : null}
        </div>
        <button type="button" className="primary-button" onClick={props.onScan}>
          <Search aria-hidden="true" />
          Scan dataset
        </button>
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
          <h2>Training config</h2>
          <p>Prepare a LoRA job for the accepted local images.</p>
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
        </div>
        <button type="button" className="secondary-button" onClick={props.onCreateConfig}>
          <Save aria-hidden="true" />
          Create training config
        </button>
        {props.trainingJob ? <div className="inline-status"><CheckCircle2 aria-hidden="true" /><span>Job {props.trainingJob.job_id}</span></div> : null}
      </div>

      <div className="panel">
        <div className="panel-heading">
          <h2>Trainer status</h2>
          <p>Check the trainer entrypoint and config path before launching.</p>
        </div>
        <div className="form-grid two-column">
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
      </div>
    </section>
  );
}

function ReferencePathModule({ paths, onChange }: { paths: string[]; onChange: (paths: string[]) => void }) {
  const [browsePath, setBrowsePath] = useState(paths[0] ? parentPath(paths[0]) : "");
  const [browser, setBrowser] = useState<PathBrowserResponse | null>(null);
  const [browserError, setBrowserError] = useState("");
  const [isBrowsing, setIsBrowsing] = useState(false);

  const openPath = async (path = browsePath) => {
    try {
      setIsBrowsing(true);
      setBrowserError("");
      const result = await api.browseFilesystem(path);
      setBrowser(result);
      setBrowsePath(result.current_path ?? path);
    } catch (caught) {
      setBrowserError(caught instanceof Error ? caught.message : "Unable to open path.");
    } finally {
      setIsBrowsing(false);
    }
  };

  const addPath = (path: string) => onChange(uniquePaths([...paths, path]));
  const removePath = (path: string) => onChange(paths.filter((candidate) => candidate !== path));

  return (
    <section className="path-module" aria-label="Reference image path module">
      <div className="path-module-heading">
        <div>
          <strong>Reference images</strong>
          <span>{paths.length} selected</span>
        </div>
      </div>
      <div className="path-open-row">
        <label>
          Browse from path
          <input value={browsePath} onChange={(event) => setBrowsePath(event.target.value)} placeholder="H:/DevWork/Win_Apps/Liidar/Models/Marianna" />
        </label>
        <button type="button" className="secondary-button inline-button" onClick={() => void openPath()} disabled={isBrowsing}>
          <FolderOpen aria-hidden="true" />
          Open path
        </button>
      </div>
      {browserError ? <div className="inline-error">{browserError}</div> : null}
      {browser ? (
        <div className="path-browser">
          <div className="path-browser-bar">
            <strong>{browser.current_path ?? "Local roots"}</strong>
            {browser.parent_path ? (
              <button type="button" className="mini-button" onClick={() => void openPath(browser.parent_path ?? "")}>
                Up
              </button>
            ) : null}
          </div>
          <div className="path-entry-list">
            {browser.entries.map((entry) => (
              <div className="path-entry" key={entry.path}>
                <div>
                  {entry.kind === "file" ? <FileImage aria-hidden="true" /> : <Folder aria-hidden="true" />}
                  <span>{entry.name}</span>
                </div>
                {entry.kind === "file" ? (
                  <button type="button" className="mini-button" onClick={() => addPath(entry.path)} aria-label={`Add ${entry.name}`}>
                    <Plus aria-hidden="true" />
                    Add
                  </button>
                ) : (
                  <button type="button" className="mini-button" onClick={() => void openPath(entry.path)} aria-label={`Open ${entry.name}`}>
                    Open
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : null}
      <div className="selected-paths">
        {paths.length ? (
          paths.map((path) => (
            <div className="selected-path" key={path}>
              <span>{path}</span>
              <button type="button" className="icon-mini-button" onClick={() => removePath(path)} aria-label={`Remove ${path}`}>
                <X aria-hidden="true" />
              </button>
            </div>
          ))
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
