import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const character = {
  id: "ari",
  display_name: "Ari",
  age_category: "adult_25_plus",
  face_summary: "soft oval face",
  hair: "dark shoulder-length hair",
  eyes: "hazel eyes",
  skin_tone: "warm medium skin",
  body_shape: "athletic",
  chest: "natural",
  grooming: "polished",
  style_notes: "quiet studio wardrobe",
  negative_notes: "no plastic skin",
  reference_images: [],
  lora_files: [],
  seed_strategy: "vary",
  locked_seed: null,
  created_at: "2026-04-30T00:00:00Z",
  updated_at: "2026-04-30T00:00:00Z"
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = init?.method ?? "GET";

        if (url.endsWith("/api/runtime")) {
          return jsonResponse({
            os_name: "Windows",
            python_version: "3.13",
            cpu_name: "Ryzen",
            total_ram_gb: 64,
            gpu_names: ["AMD Radeon RX"],
            amd_driver_version: "26.4",
            comfyui_path_exists: true,
            warnings: []
          });
        }

        if (url.endsWith("/api/characters") && method === "GET") {
          return jsonResponse([character]);
        }

        if (url.endsWith("/api/characters") && method === "POST") {
          return jsonResponse(JSON.parse(String(init?.body)));
        }

        if (url.includes("/api/filesystem/browse")) {
          return jsonResponse({
            current_path: "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna",
            parent_path: "H:\\DevWork\\Win_Apps\\Liidar\\Models",
            entries: [
              {
                name: "sozee_2026-04-30_11-47-30.png",
                path: "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png",
                kind: "file"
              },
              {
                name: "Backups",
                path: "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\Backups",
                kind: "directory"
              }
            ]
          });
        }

        if (url.includes("/api/filesystem/select-references")) {
          return jsonResponse({
            selected_paths: ["H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\picked.png"]
          });
        }

        if (url.includes("/api/characters/analyze-references")) {
          return jsonResponse({
            ...character,
            display_name: "Marianna",
            face_summary: "offline image analysis from 1 reference image; fictional adult face",
            hair: "reference-inferred dark hair",
            eyes: "reference-inferred eyes",
            skin_tone: "warm natural skin tone",
            body_shape: "reference-inferred adult body proportions",
            chest: "reference-inferred natural adult body shape",
            grooming: "reference-inferred grooming",
            style_notes: "portrait-oriented local reference style",
            reference_images: ["H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png"],
            consistency_score: 76,
            analysis_warnings: ["Add 2-4 more references for stronger identity consistency."],
            analysis_images: [
              {
                path: "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png",
                file_name: "sozee_2026-04-30_11-47-30.png",
                width: 800,
                height: 1200,
                orientation: "portrait",
                brightness: "natural mid-key lighting",
                tone: "warm",
                texture: "moderate natural texture",
                caption: "a woman in a white shirt and red bikini on the beach",
                body_attributes: {
                  coverage: "swimwear",
                  chest_visibility: "covered by swimwear or clothing",
                  pose_framing: "full or upper body visible",
                  confidence: 72,
                  evidence: "caption mentions bikini"
                },
                image_details: {
                  face: { visibility: "partial", confidence: 55, summary: "face present but not identity-locked", evidence: "caption mentions woman" },
                  hair: { visibility: "clear", confidence: 82, summary: "long dark wavy hair", evidence: "caption mentions dark wavy hair" },
                  skin: { visibility: "partial", confidence: 66, summary: "warm natural skin tone", evidence: "warm image tone" },
                  body_shape: { visibility: "partial", confidence: 70, summary: "adult full-body proportions visible", evidence: "caption mentions full body" },
                  chest: { visibility: "partial", confidence: 72, summary: "swimwear-covered chest/body shape; exact attributes require clearer references", evidence: "caption mentions bikini" },
                  waist_hips: { visibility: "partial", confidence: 64, summary: "partial waist and hip cues from body framing", evidence: "full body framing" },
                  pose: { visibility: "clear", confidence: 80, summary: "standing beach pose", evidence: "caption mentions beach" },
                  clothing: { visibility: "clear", confidence: 84, summary: "red bikini", evidence: "caption mentions red bikini" },
                  lighting: { visibility: "clear", confidence: 75, summary: "natural mid-key lighting", evidence: "image brightness" },
                  camera: { visibility: "partial", confidence: 60, summary: "portrait frame", evidence: "image orientation" },
                  background: { visibility: "clear", confidence: 82, summary: "beach setting", evidence: "caption mentions beach" },
                  quality: { visibility: "clear", confidence: 78, summary: "moderate natural texture", evidence: "image texture" }
                },
                adult_content: {
                  nudity_level: "non-explicit revealing clothing",
                  breast_visibility: "covered or partially visible",
                  nipple_areola_visibility: "not clearly described",
                  genital_visibility: "not clearly described",
                  buttocks_visibility: "not clearly described",
                  sexual_activity: "none described",
                  confidence: 72,
                  evidence: "caption mentions bikini"
                },
                vision_tags: [
                  { name: "rating: questionable", confidence: 61, source: "wd tagger test" },
                  { name: "bikini", confidence: 88, source: "wd tagger test" },
                  { name: "long hair", confidence: 82, source: "wd tagger test" }
                ]
              }
            ],
            aggregate_intelligence: {
              face: { visibility: "partial", confidence: 55, summary: "face present but not identity-locked", evidence: "1 image" },
              hair: { visibility: "clear", confidence: 82, summary: "long dark wavy hair", evidence: "consistent caption cue" },
              skin: { visibility: "partial", confidence: 66, summary: "warm natural skin tone", evidence: "image tone" },
              body_shape: { visibility: "partial", confidence: 70, summary: "adult full-body proportions visible", evidence: "full body framing" },
              chest: { visibility: "partial", confidence: 72, summary: "swimwear-covered chest/body shape; exact attributes require clearer references", evidence: "caption mentions bikini" },
              waist_hips: { visibility: "partial", confidence: 64, summary: "partial waist and hip cues from body framing", evidence: "full body framing" },
              pose: { visibility: "clear", confidence: 80, summary: "standing beach pose", evidence: "caption mentions beach" },
              clothing: { visibility: "clear", confidence: 84, summary: "red bikini", evidence: "caption mentions red bikini" },
              lighting: { visibility: "clear", confidence: 75, summary: "natural mid-key lighting", evidence: "image brightness" },
              camera: { visibility: "partial", confidence: 60, summary: "portrait frame", evidence: "image orientation" },
              background: { visibility: "clear", confidence: 82, summary: "beach setting", evidence: "caption mentions beach" },
              quality: { visibility: "clear", confidence: 78, summary: "moderate natural texture", evidence: "image texture" },
              adult_content: {
                nudity_level: "non-explicit revealing clothing",
                breast_visibility: "covered or partially visible",
                nipple_areola_visibility: "not clearly described",
                genital_visibility: "not clearly described",
                buttocks_visibility: "not clearly described",
                sexual_activity: "none described",
                confidence: 72,
                evidence: "caption mentions bikini"
              },
              strong_tags: [
                { name: "bikini", confidence: 88, source: "wd tagger test" },
                { name: "long hair", confidence: 82, source: "wd tagger test" }
              ],
              prompt_summary: "Prompt-ready reference intelligence: long dark wavy hair, warm natural skin tone, swimwear-covered body cues, standing beach pose.",
              uncertainty_notes: ["Exact anatomy requires clearer uncovered or targeted references."]
            }
          });
        }

        if (url.endsWith("/api/generate/preview")) {
          return jsonResponse({
            positive: "believable still photo, golden hour balcony",
            negative: "no plastic skin",
            seed: 42,
            width: 1024,
            height: 1024,
            steps: 30,
            cfg: 6.5
          });
        }

        if (url.endsWith("/api/training/scan")) {
          return jsonResponse({
            dataset_id: "dataset-1",
            name: "identity",
            dataset_type: "fictional_face_identity",
            accepted_count: 12,
            rejected_count: 2,
            duplicate_count: 1,
            ignored_count: 3,
            accepted: [],
            rejected: [],
            warnings: ["Skipped unreadable image."]
          });
        }

        if (url.endsWith("/api/training/config")) {
          return jsonResponse({
            job_id: "job-1",
            dataset_id: "dataset-1",
            dataset_path: "H:\\photos\\training-pack",
            output_dir: "outputs/lora",
            base_model_path: "base.safetensors",
            lora_name: "ari_style",
            resolution: 1024,
            repeats: 10,
            batch_size: 1,
            max_train_steps: 1200,
            learning_rate: 0.0001,
            network_dim: 32,
            network_alpha: 16,
            accepted_image_count: 12,
            completed_lora_path: null,
            config_path: "H:\\studio\\config\\training\\job-1.json"
          });
        }

        if (url.includes("/api/training/status")) {
          return jsonResponse({
            trainer_entrypoint: "train_network.py",
            config_path: "H:\\studio\\config\\training\\job-1.json",
            trainer_entrypoint_exists: false,
            config_path_exists: true,
            warnings: ["Trainer entrypoint is missing: train_network.py"]
          });
        }

        return jsonResponse({});
      })
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders runtime, character, generation, and training panels", async () => {
    render(<App />);

    expect(await screen.findByRole("tab", { name: "Runtime status" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Characters" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Generate" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Training" })).toBeInTheDocument();
  });

  it("shows character editor fields from the backend profile", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await screen.findByDisplayValue("Ari");
    expect(screen.getByLabelText("Age category")).toBeInTheDocument();
    expect(screen.getByLabelText("Reference image paths")).toBeInTheDocument();
    expect(screen.getByLabelText("Face summary")).not.toBeVisible();
    await user.click(screen.getByText("Advanced profile controls"));
    expect(screen.getByLabelText("Face summary")).toBeVisible();
    expect(screen.getByLabelText("Negative notes")).toBeVisible();
  });

  it("keeps reference actions disabled until images are selected", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));

    expect(screen.getByRole("button", { name: /analyze references/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /create draft from references/i })).toBeDisabled();
  });

  it("creates a draft character profile from reference paths", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.clear(await screen.findByLabelText("Display name"));
    await user.type(
      screen.getByLabelText("Reference image paths"),
      "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png\nH:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-57-43.png"
    );
    await user.click(screen.getByRole("button", { name: /create draft from references/i }));

    expect(screen.getByDisplayValue("Marianna")).toBeInTheDocument();
    await user.click(screen.getByText("Advanced profile controls"));
    expect((screen.getByLabelText("Face summary") as HTMLTextAreaElement).value).toContain("fictional adult face");
    expect((screen.getByLabelText("Style notes") as HTMLTextAreaElement).value).toContain("reference-guided");
  });

  it("adds reference images from the local path browser", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.type(
      screen.getByLabelText("Browse from path"),
      "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna"
    );
    await user.click(screen.getByRole("button", { name: "Open path" }));
    await user.click(await screen.findByRole("button", { name: /add sozee_2026-04-30_11-47-30.png/i }));

    expect(screen.getAllByText("1 selected").length).toBeGreaterThan(0);
    expect(screen.getAllByText("H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png").length).toBeGreaterThan(0);
  });

  it("selects reference images from the native Windows picker control", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.click(screen.getByRole("button", { name: /select images/i }));

    expect(await screen.findByText("1 image selected.")).toBeInTheDocument();
    expect(screen.getAllByText("H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\picked.png").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /analyze references/i })).toBeEnabled();
  });

  it("starts a new character and supports bulk reference actions", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.click(screen.getByRole("button", { name: /new character/i }));
    expect(screen.getByLabelText("Display name")).toHaveValue("");

    await user.type(screen.getByLabelText("Browse from path"), "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna");
    await user.click(screen.getByRole("button", { name: "Open path" }));
    await user.click(await screen.findByRole("button", { name: /add visible files/i }));
    expect(screen.getAllByText("1 selected").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /clear selected/i }));
    expect(screen.getByText("0 selected")).toBeInTheDocument();
  });

  it("saves a completed character profile", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.click(screen.getByRole("button", { name: /new character/i }));
    await user.type(screen.getByLabelText("Display name"), "Nadia");
    await user.click(screen.getByText("Advanced profile controls"));
    await user.type(screen.getByLabelText("Face summary"), "fictional adult face with soft features");
    await user.click(screen.getByRole("button", { name: /save character/i }));

    expect(await screen.findByText("Character saved.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /nadia/i })).toBeInTheDocument();
  });

  it("analyzes selected references into an editable character draft", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.clear(await screen.findByLabelText("Display name"));
    await user.type(
      screen.getByLabelText("Reference image paths"),
      "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png"
    );
    await user.click(screen.getByRole("button", { name: /analyze references/i }));

    expect(await screen.findByDisplayValue("Marianna")).toBeInTheDocument();
    expect(screen.getByText("76")).toBeInTheDocument();
    expect(screen.getByText("a woman in a white shirt and red bikini on the beach")).toBeInTheDocument();
    expect(screen.getByText("Reference blueprint")).toBeInTheDocument();
    expect(screen.getByText("What should stay consistent")).toBeInTheDocument();
    expect(screen.getAllByText("long dark wavy hair").length).toBeGreaterThan(0);
    expect(screen.getByText("Needs better references")).toBeInTheDocument();
    expect(screen.getByText("swimwear")).toBeInTheDocument();
    expect(screen.getByText("covered by swimwear or clothing")).toBeInTheDocument();
    expect(screen.getByText("Add 2-4 more references for stronger identity consistency.")).toBeInTheDocument();
    await user.click(screen.getByText("Advanced profile controls"));
    expect((screen.getByLabelText("Face summary") as HTMLTextAreaElement).value).toContain("offline image analysis");
    expect((screen.getByLabelText("Skin tone") as HTMLTextAreaElement).value).toContain("warm");
  });

  it("keeps reference analysis scoped to the active character", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.type(
      screen.getByLabelText("Reference image paths"),
      "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png"
    );
    await user.click(screen.getByRole("button", { name: /analyze references/i }));
    expect(await screen.findByText("Reference blueprint")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /new character/i }));

    expect(screen.queryByText("Reference blueprint")).not.toBeInTheDocument();
    expect(screen.getByText("Setup readiness")).toBeInTheDocument();
  });

  it("shows analysis progress while references are being analyzed", async () => {
    const fetchMock = vi.mocked(fetch);
    const defaultFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/api/characters/analyze-references")) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return defaultFetch?.(input, init) ?? jsonResponse({});
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.type(
      screen.getByLabelText("Reference image paths"),
      "H:\\DevWork\\Win_Apps\\Liidar\\Models\\Marianna\\sozee_2026-04-30_11-47-30.png"
    );
    await user.click(screen.getByRole("button", { name: /analyze references/i }));

    expect(await screen.findByRole("progressbar", { name: /analyzing references/i })).toBeInTheDocument();
    expect(screen.getByText(/Reading selected images/i)).toBeInTheDocument();
  });

  it("previews a believable still photo recipe", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Generate" }));
    await user.click(screen.getByRole("button", { name: /preview recipe/i }));

    expect(await screen.findByText("believable still photo, golden hour balcony")).toBeInTheDocument();
  });

  it("renders training controls and scan counts", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Training" }));
    expect(screen.getByLabelText("Source folder path")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Dataset type" })).toHaveTextContent("fictional_face_identity");
    expect(screen.getByLabelText("Face policy")).toBeInTheDocument();
    expect(screen.getByLabelText("Source rights")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /scan dataset/i }));

    expect(await screen.findByText("12")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("Skipped unreadable image.")).toBeInTheDocument();
  });

  it("uses backend config path and surfaces trainer warnings", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Training" }));
    await user.click(screen.getByRole("button", { name: /create training config/i }));

    expect(await screen.findByDisplayValue("H:\\studio\\config\\training\\job-1.json")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /check status/i }));

    expect(await screen.findByText("Trainer entrypoint is missing: train_network.py")).toBeInTheDocument();
  });
});
