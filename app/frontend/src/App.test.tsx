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
    expect(screen.getByLabelText("Face summary")).toBeInTheDocument();
    expect(screen.getByLabelText("Negative notes")).toBeInTheDocument();
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
