import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
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

const recipe = {
  positive: "believable still photo, golden hour balcony",
  negative: "no plastic skin",
  seed: 42,
  width: 1024,
  height: 1024,
  steps: 30,
  cfg: 6.5,
  lora_files: [],
  lora_strength: 0.75
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

function trainingJob(loraName = "global_body_pack") {
  return {
    job_id: "job-1",
    dataset_id: "global-body-pack",
    dataset_path: "H:\\Desktop\\Liidar_Dataset_Crops\\run-1",
    output_dir: "outputs/global_lora",
    base_model_path: "models/sdxl_base_1.0.safetensors",
    lora_name: loraName,
    dataset_type: "body_part",
    global_pack: true,
    resolution: 896,
    repeats: 8,
    batch_size: 1,
    max_train_steps: 1200,
    learning_rate: 0.00002,
    network_dim: 32,
    network_alpha: 16,
    accepted_image_count: 8,
    completed_lora_path: null,
    config_path: "H:\\studio\\config\\training\\job-1.json"
  };
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

        if (url.endsWith("/api/system/live")) {
          return jsonResponse({
            timestamp: "2026-05-01T00:00:00Z",
            cpu_percent: 21.5,
            ram_used_gb: 15.2,
            ram_total_gb: 31.9,
            ram_percent: 47.6,
            gpu_percent: 63.2,
            gpu_memory_used_gb: 7.25,
            gpu_memory_total_gb: null,
            gpu_provider: "Windows performance counters",
            process_memory_mb: 312.4,
            warnings: []
          });
        }

        if (url.endsWith("/api/characters") && method === "GET") {
          return jsonResponse([character]);
        }

        if (url.endsWith("/api/characters") && method === "POST") {
          return jsonResponse(JSON.parse(String(init?.body)));
        }

        if (url.includes("/api/characters/analyze-references")) {
          return jsonResponse({
            ...character,
            display_name: "Ari",
            reference_images: ["H:\\refs\\ari.png"],
            consistency_score: 76,
            analysis_warnings: ["Add 2-4 more references for stronger identity consistency."],
            analysis_images: [
              {
                path: "H:\\refs\\ari.png",
                file_name: "ari.png",
                width: 800,
                height: 1200,
                orientation: "portrait",
                brightness: "natural mid-key lighting",
                tone: "warm",
                texture: "moderate natural texture",
                caption: "studio reference",
                body_attributes: {
                  coverage: "reference",
                  chest_visibility: "visible",
                  pose_framing: "upper body",
                  confidence: 72,
                  evidence: "local analysis"
                },
                image_details: {},
                adult_content: {
                  nudity_level: "adult reference",
                  breast_visibility: "visible",
                  nipple_areola_visibility: "not clearly described",
                  genital_visibility: "not clearly described",
                  buttocks_visibility: "not clearly described",
                  sexual_activity: "none described",
                  confidence: 72,
                  evidence: "local analysis"
                },
                vision_tags: [{ name: "studio", confidence: 88, source: "local" }]
              }
            ],
            aggregate_intelligence: {
              face: { visibility: "clear", confidence: 82, summary: "soft oval face", evidence: "local analysis" },
              hair: { visibility: "clear", confidence: 82, summary: "dark shoulder-length hair", evidence: "local analysis" },
              skin: { visibility: "partial", confidence: 66, summary: "warm medium skin", evidence: "local analysis" },
              body_shape: { visibility: "partial", confidence: 70, summary: "athletic adult proportions", evidence: "local analysis" },
              chest: { visibility: "partial", confidence: 72, summary: "natural chest", evidence: "local analysis" },
              waist_hips: { visibility: "partial", confidence: 64, summary: "balanced waist and hips", evidence: "local analysis" },
              pose: { visibility: "clear", confidence: 80, summary: "studio pose", evidence: "local analysis" },
              clothing: { visibility: "clear", confidence: 84, summary: "minimal styling", evidence: "local analysis" },
              lighting: { visibility: "clear", confidence: 75, summary: "natural light", evidence: "local analysis" },
              camera: { visibility: "partial", confidence: 60, summary: "portrait frame", evidence: "local analysis" },
              background: { visibility: "clear", confidence: 82, summary: "studio setting", evidence: "local analysis" },
              quality: { visibility: "clear", confidence: 78, summary: "usable reference", evidence: "local analysis" },
              adult_content: {
                nudity_level: "adult reference",
                breast_visibility: "visible",
                nipple_areola_visibility: "not clearly described",
                genital_visibility: "not clearly described",
                buttocks_visibility: "not clearly described",
                sexual_activity: "none described",
                confidence: 72,
                evidence: "local analysis"
              },
              strong_tags: [{ name: "studio", confidence: 88, source: "local" }],
              prompt_summary: "Prompt-ready reference intelligence.",
              uncertainty_notes: ["Add 2-4 more references for stronger identity consistency."]
            }
          });
        }

        if (url.includes("/api/filesystem/select-references")) {
          return jsonResponse({ selected_paths: ["H:\\refs\\ari.png"] });
        }

        if (url.endsWith("/api/filesystem/select-folder")) {
          return jsonResponse({ folder_path: "H:\\raw" });
        }

        if (url.includes("/api/filesystem/image-count")) {
          return jsonResponse({ path: "H:\\Desktop\\Liidar_Dataset_Crops\\run-1", image_count: 8 });
        }

        if (url.endsWith("/api/generate/settings") && method === "GET") {
          return jsonResponse({ checkpoint_name: "sdxl_base_1.0.safetensors" });
        }

        if (url.endsWith("/api/generate/settings") && method === "POST") {
          return jsonResponse(JSON.parse(String(init?.body)));
        }

        if (url.endsWith("/api/generate/enhance-prompt")) {
          return jsonResponse({
            scene_prompt: "golden hour balcony, realistic adult studio photo",
            body_detail_prompt: "natural adult proportions",
            extra_negative: "low quality, plastic skin, watermark",
            summary: "Local prompt recipe ready"
          });
        }

        if (url.endsWith("/api/generate/preview")) {
          return jsonResponse(recipe);
        }

        if (url.endsWith("/api/generate/preflight")) {
          return jsonResponse({
            ready: true,
            settings: { checkpoint_name: "sdxl_base_1.0.safetensors" },
            items: [
              { id: "comfyui", label: "ComfyUI API", ok: true, detail: "ComfyUI is responding." },
              { id: "checkpoint", label: "Checkpoint", ok: true, detail: "Checkpoint found." },
              { id: "loras", label: "LoRA files", ok: true, detail: "No LoRA files selected." }
            ],
            warnings: []
          });
        }

        if (url.endsWith("/api/generate") && method === "POST") {
          return jsonResponse({ prompt_id: "prompt-123", recipe });
        }

        if (url.endsWith("/api/generate/jobs/prompt-123") && method === "GET") {
          return jsonResponse({
            prompt_id: "prompt-123",
            status: "completed",
            queue_position: null,
            images: [{ filename: "local_model_studio_00001_.png", subfolder: "", type: "output" }],
            error: null
          });
        }

        if (url.endsWith("/api/training/jobs") && method === "GET") {
          return jsonResponse([]);
        }

        if (url.endsWith("/api/training/runs") && method === "GET") {
          return jsonResponse([]);
        }

        if (url.endsWith("/api/dataset-prep/jobs") && method === "POST") {
          return jsonResponse({
            job_id: "prep-job-1",
            status: "running",
            total_count: 10,
            processed_count: 0,
            cropped_count: 0,
            skipped_count: 0,
            ai_attempted_count: 0,
            ai_guided_count: 0,
            ai_failed_count: 0,
            face_guided_count: 0,
            fallback_count: 0,
            active_file: "H:\\raw\\one.jpg",
            output_folder: null,
            scan_mode: "local",
            use_ai: false,
            ai_max_images: 0,
            cancel_requested: false,
            error: null,
            result: null
          });
        }

        if (url.endsWith("/api/dataset-prep/jobs/prep-job-1") && method === "GET") {
          return jsonResponse({
            job_id: "prep-job-1",
            status: "completed",
            total_count: 10,
            processed_count: 10,
            cropped_count: 8,
            skipped_count: 2,
            ai_attempted_count: 0,
            ai_guided_count: 0,
            ai_failed_count: 0,
            face_guided_count: 4,
            fallback_count: 1,
            active_file: null,
            output_folder: "H:\\Desktop\\Liidar_Dataset_Crops\\run-1",
            scan_mode: "local",
            use_ai: false,
            ai_max_images: 0,
            cancel_requested: false,
            error: null,
            result: {
              output_folder: "H:\\Desktop\\Liidar_Dataset_Crops\\run-1",
              processed_count: 10,
              cropped_count: 8,
              skipped_count: 2,
              ai_attempted_count: 0,
              ai_guided_count: 0,
              ai_failed_count: 0,
              face_guided_count: 4,
              fallback_count: 1,
              warnings: [],
              images: []
            }
          });
        }

        if (url.endsWith("/api/training/config") && method === "POST") {
          return jsonResponse(trainingJob());
        }

        if (url.endsWith("/api/training/job-1/runs") && method === "POST") {
          return jsonResponse({
            run_id: "run-1",
            job_id: "job-1",
            status: "running",
            trainer_entrypoint: "tools/train_global_lora.ps1",
            config_path: "H:\\studio\\config\\training\\job-1.json",
            output_lora_path: null,
            process_id: 4242,
            exit_code: null,
            log_path: "H:\\studio\\logs\\training-run-1.log",
            progress_current: 10,
            progress_total: 1200,
            progress_percent: 1,
            tail: ["Starting training command:"],
            error: null
          });
        }

        if (url.endsWith("/api/training/runs/run-1") && method === "GET") {
          return jsonResponse({
            run_id: "run-1",
            job_id: "job-1",
            status: "completed",
            trainer_entrypoint: "tools/train_global_lora.ps1",
            config_path: "H:\\studio\\config\\training\\job-1.json",
            output_lora_path: "outputs/global_lora\\global_body_pack.safetensors",
            process_id: 4242,
            exit_code: 0,
            log_path: "H:\\studio\\logs\\training-run-1.log",
            progress_current: 1200,
            progress_total: 1200,
            progress_percent: 100,
            tail: ["wrote global_body_pack.safetensors"],
            error: null
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

  it("renders the current three-module shell with live system status", async () => {
    render(<App />);

    expect(await screen.findByRole("tab", { name: "Characters" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Generate" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Training" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Runtime status" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Dataset prep" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh studio data" })).not.toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Live system monitor" })).toBeInTheDocument();
    expect(screen.getByText("7.25 GB VRAM")).toBeInTheDocument();
  });

  it("keeps character setup focused on blueprint references", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    expect(await screen.findByDisplayValue("Ari")).toBeInTheDocument();
    expect(screen.queryByLabelText("Age category")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /build blueprint/i })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /select images/i }));
    expect(await screen.findByText("1 image selected.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /build blueprint/i }));

    expect(await screen.findByText("Reference blueprint")).toBeInTheDocument();
    expect(screen.getByText("76")).toBeInTheDocument();
    expect(screen.getByText("Prompt-ready reference intelligence.")).toBeInTheDocument();
  });

  it("saves character blueprints through the current copy", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Characters" }));
    await user.click(screen.getByRole("button", { name: /new blueprint/i }));
    await user.type(screen.getByLabelText("Blueprint name"), "Nadia");
    await user.click(screen.getByRole("button", { name: /save blueprint/i }));

    expect((await screen.findAllByText("Blueprint saved.")).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /nadia/i })).toBeInTheDocument();
  });

  it("enhances, previews, and queues generation from one brief", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Generate" }));
    await user.click(screen.getByRole("button", { name: /enhance brief/i }));
    expect((await screen.findAllByText("Brief enhanced locally.")).length).toBeGreaterThan(0);
    expect(screen.getAllByDisplayValue("golden hour balcony, realistic adult studio photo").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "Preview" }));
    expect(await screen.findByText("Generation recipe")).toBeInTheDocument();
    expect(screen.getByText("Ready to generate")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect((await screen.findAllByText("Generation queued: prompt-123.")).length).toBeGreaterThan(0);
    expect(await screen.findByText("local_model_studio_00001_.png")).toBeInTheDocument();
  });

  it("prepares cropped images, saves setup, and starts training", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("tab", { name: "Training" }));
    const trainingPanel = screen.getByRole("heading", { name: "Train sculpture pack" }).closest("section");
    expect(trainingPanel).not.toBeNull();

    await user.click(within(trainingPanel as HTMLElement).getAllByRole("button", { name: /select folder/i })[0]);
    expect(await screen.findByDisplayValue("H:\\raw")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /prepare references/i }));

    expect(await screen.findByText("8 crops ready")).toBeInTheDocument();
    expect(screen.getByDisplayValue("H:\\Desktop\\Liidar_Dataset_Crops\\run-1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /save training setup/i }));
    expect((await screen.findAllByText("Global training job created for global_body_pack.")).length).toBeGreaterThan(0);
    expect(screen.getByText("global_body_pack is ready to train")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));
    expect((await screen.findAllByText("Training started for global_body_pack.")).length).toBeGreaterThan(0);
    expect(screen.getByText("PID 4242")).toBeInTheDocument();
  });
});
