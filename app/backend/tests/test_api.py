from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import local_model_studio.main as main_module
import local_model_studio.runtime_check as runtime_check
from fastapi.testclient import TestClient
from local_model_studio import dataset_prepper
from PIL import Image

from local_model_studio.local_image_tagger import _preferred_providers
from local_model_studio.main import create_app
from local_model_studio.paths import WorkspacePaths
from local_model_studio.schemas import DatasetPrepRequest, DatasetPrepResponse


class FakeComfyClient:
    def __init__(self) -> None:
        self.queued_workflow: dict | None = None

    def queue_prompt(self, workflow: dict) -> str:
        self.queued_workflow = workflow
        return "prompt-123"


class FailingComfyClient:
    def queue_prompt(self, workflow: dict) -> str:
        raise RuntimeError("ComfyUI returned an invalid response.")


@dataclass(frozen=True)
class FakeTag:
    name: str
    confidence: float


@dataclass(frozen=True)
class FakeTagReport:
    tags: list[FakeTag]
    rating: str = "explicit"
    rating_confidence: float = 0.91
    source: str = "wd tagger test"

    def reference_text(self) -> str:
        return f"wd tagger test rating {self.rating}; local visual tags: " + ", ".join(tag.name for tag in self.tags)


def test_profile_crud(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    created = client.post(
        "/api/characters",
        json={
            "id": "ari",
            "display_name": "Ari",
            "face_summary": "soft oval face",
        },
    )
    assert created.status_code == 200
    assert created.json()["id"] == "ari"

    listed = client.get("/api/characters")
    assert listed.status_code == 200
    assert [profile["id"] for profile in listed.json()] == ["ari"]

    fetched = client.get("/api/characters/ari")
    assert fetched.status_code == 200
    assert fetched.json()["display_name"] == "Ari"

    deleted = client.delete("/api/characters/ari")
    assert deleted.status_code == 204

    missing = client.get("/api/characters/ari")
    assert missing.status_code == 404


def test_live_system_metrics_route_returns_usage(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.get("/api/system/live")

    assert response.status_code == 200
    body = response.json()
    assert body["cpu_percent"] >= 0
    assert body["ram_total_gb"] > 0
    assert body["ram_used_gb"] >= 0
    assert body["ram_percent"] >= 0
    assert body["process_memory_mb"] > 0
    assert "timestamp" in body


def test_anthropic_key_routes_save_status_and_delete(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    initial = client.get("/api/settings/anthropic-key")
    saved = client.post("/api/settings/anthropic-key", json={"api_key": "sk-ant-test"})
    removed = client.delete("/api/settings/anthropic-key")

    assert initial.status_code == 200
    assert initial.json()["saved"] is False
    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert saved.json()["model"] == "claude-haiku-4-5-20251001"
    assert removed.status_code == 200
    assert removed.json()["saved"] is False


def test_anthropic_key_status_migrates_retired_haiku_model(tmp_path: Path) -> None:
    secrets = tmp_path / "config" / "secrets"
    secrets.mkdir(parents=True)
    secret_file = secrets / "anthropic.json"
    secret_file.write_text(
        '{"api_key":"sk-ant-test","model":"claude-3-haiku-20240307"}',
        encoding="utf-8",
    )
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.get("/api/settings/anthropic-key")

    assert response.status_code == 200
    assert response.json()["model"] == "claude-haiku-4-5-20251001"
    assert "claude-haiku-4-5-20251001" in secret_file.read_text(encoding="utf-8")


def test_anthropic_key_test_route_surfaces_provider_error(tmp_path: Path, monkeypatch) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    client.post("/api/settings/anthropic-key", json={"api_key": "sk-ant-test"})
    monkeypatch.setattr(
        main_module.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            404,
            json={"error": {"type": "not_found_error", "message": "model not found"}},
        ),
    )

    response = client.post("/api/settings/anthropic-key/test")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["status_code"] == 404
    assert body["message"] == "404 Not Found - not_found_error: model not found"


def test_dataset_prep_request_uses_server_active_model_over_stale_payload(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    client.post("/api/settings/anthropic-key", json={"api_key": "sk-ant-test"})
    request = DatasetPrepRequest(
        source_folder=tmp_path,
        output_folder=tmp_path / "out",
        target="chest_detail",
        use_ai=True,
        ai_model="claude-3-haiku-20240307",
    )

    updated = main_module._server_ai_request(request, main_module.ApiKeyStore(tmp_path))

    assert updated.ai_model == "claude-haiku-4-5-20251001"


def test_gpu_counter_probe_treats_idle_gpu_as_zero(monkeypatch) -> None:
    runtime_check._GPU_COUNTER_CACHE = None
    monkeypatch.setattr(runtime_check.platform, "system", lambda: "Windows")

    class Completed:
        stdout = '{"GpuPercent":0,"DedicatedBytes":123456789,"Provider":"Windows performance counters"}'

    monkeypatch.setattr(runtime_check.subprocess, "run", lambda *args, **kwargs: Completed())

    metrics = runtime_check.build_live_system_metrics()

    assert metrics.gpu_percent == 0
    assert "GPU utilization counter is unavailable." not in metrics.warnings


def test_generate_preview_returns_recipe_containing_scene_prompt(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    client.post("/api/characters", json={"id": "ari", "display_name": "Ari"})

    response = client.post(
        "/api/generate/preview",
        json={
            "character_id": "ari",
            "mode": "portrait",
            "scene_prompt": "golden hour balcony",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    assert "golden hour balcony" in response.json()["positive"]
    assert response.json()["seed"] == 42


def test_generate_route_with_fake_comfy_client_returns_prompt_id(tmp_path: Path) -> None:
    comfy = FakeComfyClient()
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path), comfy_client=comfy))
    client.post("/api/characters", json={"id": "ari", "display_name": "Ari"})

    response = client.post(
        "/api/generate",
        json={"character_id": "ari", "scene_prompt": "studio window light", "seed": 7},
    )

    assert response.status_code == 200
    assert response.json()["prompt_id"] == "prompt-123"
    assert response.json()["recipe"]["seed"] == 7
    assert comfy.queued_workflow is not None
    assert comfy.queued_workflow["4"]["inputs"]["ckpt_name"] == "sdxl_base_1.0.safetensors"


def test_completed_global_pack_is_added_to_future_generation(tmp_path: Path) -> None:
    comfy = FakeComfyClient()
    accepted_dir = tmp_path / "datasets" / "accepted"
    accepted_dir.mkdir(parents=True)
    lora_path = tmp_path / "outputs" / "global_body_pack.safetensors"
    lora_path.parent.mkdir()
    lora_path.write_bytes(b"fake")
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path), comfy_client=comfy))
    client.post("/api/characters", json={"id": "ari", "display_name": "Ari"})
    job = client.post(
        "/api/training/config",
        json={
            "request": {
                "dataset_id": "dataset-1",
                "dataset_path": str(accepted_dir),
                "output_dir": str(lora_path.parent),
                "base_model_path": "base.safetensors",
                "lora_name": "global_body_pack",
                "dataset_type": "body_shape",
                "global_pack": True,
            },
            "accepted_image_count": 8,
        },
    ).json()
    client.post(f"/api/training/{job['job_id']}/register-lora", json={"lora_path": str(lora_path)})

    preview = client.post("/api/generate/preview", json={"character_id": "ari", "seed": 12})
    generated = client.post("/api/generate", json={"character_id": "ari", "seed": 12})

    assert preview.status_code == 200
    assert preview.json()["lora_files"] == [str(lora_path)]
    assert generated.status_code == 200
    assert comfy.queued_workflow is not None
    assert comfy.queued_workflow["21"]["class_type"] == "LoraLoader"
    assert comfy.queued_workflow["21"]["inputs"]["lora_name"] == str(lora_path)
    assert comfy.queued_workflow["3"]["inputs"]["model"] == ["21", 0]


def test_generate_route_returns_503_for_comfy_runtime_error(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path), comfy_client=FailingComfyClient()))
    client.post("/api/characters", json={"id": "ari", "display_name": "Ari"})

    response = client.post(
        "/api/generate",
        json={"character_id": "ari", "scene_prompt": "studio window light"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "ComfyUI returned an invalid response."


def test_missing_character_preview_and_generate_return_404(tmp_path: Path) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path), comfy_client=FakeComfyClient()))

    preview = client.post("/api/generate/preview", json={"character_id": "missing"})
    generate = client.post("/api/generate", json={"character_id": "missing"})

    assert preview.status_code == 404
    assert generate.status_code == 404


def test_file_browser_lists_folders_and_supported_images(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    (source / "Nested").mkdir()
    Image.new("RGB", (8, 8), color="blue").save(source / "sozee.png")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.get("/api/filesystem/browse", params={"path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["current_path"] == str(source)
    assert body["parent_path"] == str(source.parent)
    entries = {(entry["name"], entry["kind"]) for entry in body["entries"]}
    assert ("Nested", "directory") in entries
    assert ("sozee.png", "file") in entries
    assert ("notes.txt", "file") not in entries


def test_file_browser_serves_image_thumbnail(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    reference = source / "sozee.png"
    Image.new("RGB", (800, 1200), color="blue").save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.get("/api/filesystem/thumbnail", params={"path": str(reference)})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 100


def test_dataset_prep_crop_route_writes_crops_to_output_folder(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    source.mkdir()
    Image.new("RGB", (800, 1200), color=(220, 190, 170)).save(source / "body.jpg")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.post(
        "/api/dataset-prep/crop",
        json={
            "source_folder": str(source),
            "output_folder": str(output),
            "target": "chest_detail",
            "recursive": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_folder"] == str(output)
    assert body["processed_count"] == 1
    assert body["cropped_count"] == 1
    assert body["images"][0]["accepted"] is True
    assert body["images"][0]["output_path"].endswith(".jpg")
    assert Path(body["images"][0]["output_path"]).is_file()


def test_dataset_prep_ai_chest_crop_requires_breast_boxes() -> None:
    crop = dataset_prepper._crop_box_from_ai_response(
        {
            "visible_target": True,
            "crop_box": [0.1, 0.05, 0.9, 0.72],
            "target_box": [0.24, 0.34, 0.78, 0.58],
            "face_box": [0.34, 0.04, 0.66, 0.31],
        },
        1000,
        1200,
        "chest_detail",
    )

    assert crop is None


def test_dataset_prep_ai_chest_crop_requires_nipple_boxes() -> None:
    crop = dataset_prepper._crop_box_from_ai_response(
        {
            "visible_target": True,
            "breast_boxes": [[0.18, 0.28, 0.46, 0.62], [0.5, 0.27, 0.78, 0.62]],
            "face_box": [0.3, 0.04, 0.68, 0.25],
        },
        1000,
        1200,
        "chest_detail",
    )

    assert crop is None


def test_dataset_prep_ai_chest_crop_includes_full_breasts_and_nipples() -> None:
    crop = dataset_prepper._crop_box_from_ai_response(
        {
            "visible_target": True,
            "breast_boxes": [[0.18, 0.28, 0.46, 0.62], [0.5, 0.27, 0.78, 0.62]],
            "nipple_boxes": [[0.31, 0.45, 0.36, 0.5], [0.62, 0.44, 0.67, 0.5]],
            "face_box": [0.3, 0.04, 0.68, 0.25],
        },
        1000,
        1200,
        "chest_detail",
    )

    assert crop is not None
    left, top, right, bottom = crop
    assert left <= 180
    assert right >= 780
    assert top <= 336
    assert bottom >= 744
    assert top >= 300


def test_dataset_prep_ai_chest_crop_prefers_breast_boxes_over_broad_target() -> None:
    crop = dataset_prepper._crop_box_from_ai_response(
        {
            "visible_target": True,
            "crop_box": [0.04, 0.02, 0.95, 0.75],
            "target_box": [0.12, 0.12, 0.82, 0.65],
            "breast_boxes": [[0.22, 0.37, 0.46, 0.58], [0.5, 0.36, 0.74, 0.59]],
            "nipple_boxes": [[0.34, 0.47, 0.38, 0.51], [0.61, 0.46, 0.65, 0.51]],
            "face_box": [0.3, 0.04, 0.68, 0.32],
        },
        1000,
        1200,
        "chest_detail",
    )

    assert crop is not None
    left, top, right, bottom = crop
    assert 130 <= left <= 230
    assert 730 <= right <= 830
    assert top >= 320
    assert bottom <= 760


def test_dataset_prep_ai_image_payload_is_cost_capped() -> None:
    encoded = dataset_prepper._encode_image_for_ai(Image.new("RGB", (3000, 2200), color=(220, 190, 170)))
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as resized:
        assert max(resized.size) == 512


def test_dataset_prep_claude_crop_reads_tool_use_response(monkeypatch) -> None:
    captured_payload: dict | None = None

    def fake_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs["json"]
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "name": "return_crop",
                        "input": {
                            "visible_target": True,
                            "breast_boxes": [[0.22, 0.37, 0.46, 0.58], [0.5, 0.36, 0.74, 0.59]],
                            "nipple_boxes": [[0.34, 0.47, 0.38, 0.51], [0.61, 0.46, 0.65, 0.51]],
                            "face_box": [0.3, 0.04, 0.68, 0.32],
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(dataset_prepper.httpx, "post", fake_post)

    crop, warning = dataset_prepper._claude_crop_box(
        Image.new("RGB", (1000, 1200), color=(220, 190, 170)),
        "chest_detail",
        anthropic_api_key="sk-ant-test",
        model="claude-haiku-4-5-20251001",
    )

    assert warning is None
    assert crop is not None
    assert 130 <= crop[0] <= 230
    assert 730 <= crop[2] <= 830
    assert captured_payload is not None
    assert captured_payload["tool_choice"] == {"type": "tool", "name": "return_crop"}
    assert captured_payload["tools"][0]["name"] == "return_crop"
    assert "target_box" not in captured_payload["tools"][0]["input_schema"]["properties"]
    assert "nipple_boxes" in captured_payload["tools"][0]["input_schema"]["required"]


def test_dataset_prep_ai_failure_skips_instead_of_writing_fallback_crop(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    source.mkdir()
    Image.new("RGB", (800, 1200), color=(220, 190, 170)).save(source / "body.jpg")
    monkeypatch.setattr(
        dataset_prepper,
        "_claude_crop_box",
        lambda image, target, anthropic_api_key, model: (None, "Claude returned broad portrait crop."),
    )

    report = dataset_prepper.prepare_dataset_crops(
        DatasetPrepRequest(source_folder=source, output_folder=output, target="chest_detail", use_ai=True),
        anthropic_api_key="sk-ant-test",
    )

    assert report.cropped_count == 0
    assert report.skipped_count == 1
    assert report.ai_attempted_count == 1
    assert report.ai_failed_count == 1
    assert report.fallback_count == 0
    assert report.images[0].method == "ai_failed"


def test_dataset_prep_formats_anthropic_error_body() -> None:
    response = httpx.Response(
        404,
        json={"error": {"type": "not_found_error", "message": "model not found"}},
    )

    assert dataset_prepper._anthropic_error_message(response) == "404 Not Found - not_found_error: model not found"


def test_dataset_prep_job_reports_progress_and_result(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    source.mkdir()
    Image.new("RGB", (800, 1200), color=(220, 190, 170)).save(source / "body.jpg")

    def fake_prepare_dataset_crops(request, *, anthropic_api_key=None, progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback(
                {
                    "total_count": 1,
                    "processed_count": 1,
                    "cropped_count": 1,
                    "skipped_count": 0,
                    "ai_attempted_count": 1,
                    "ai_guided_count": 1,
                    "ai_failed_count": 0,
                    "face_guided_count": 0,
                    "fallback_count": 0,
                    "active_file": str(source / "body.jpg"),
                    "output_folder": str(output),
                }
            )
        return DatasetPrepResponse(
            output_folder=str(output),
            processed_count=1,
            cropped_count=1,
            skipped_count=0,
            ai_attempted_count=1,
            ai_guided_count=1,
            ai_failed_count=0,
            face_guided_count=0,
            fallback_count=0,
            warnings=[],
            images=[],
        )

    monkeypatch.setattr("local_model_studio.dataset_prep_jobs.prepare_dataset_crops", fake_prepare_dataset_crops)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    started = client.post(
        "/api/dataset-prep/jobs",
        json={
            "source_folder": str(source),
            "output_folder": str(output),
            "target": "chest_detail",
            "recursive": True,
            "use_ai": True,
        },
    )

    assert started.status_code == 200
    job_id = started.json()["job_id"]
    status = client.get(f"/api/dataset-prep/jobs/{job_id}")
    for _ in range(20):
        if status.json()["status"] == "completed":
            break
        time.sleep(0.05)
        status = client.get(f"/api/dataset-prep/jobs/{job_id}")

    body = status.json()
    assert body["status"] == "completed"
    assert body["total_count"] == 1
    assert body["processed_count"] == 1
    assert body["cropped_count"] == 1
    assert body["ai_attempted_count"] == 1
    assert body["ai_guided_count"] == 1
    assert body["ai_failed_count"] == 0
    assert body["result"]["output_folder"] == str(output)


def test_select_reference_images_uses_native_picker_result(tmp_path: Path, monkeypatch) -> None:
    reference = tmp_path / "picked.png"
    Image.new("RGB", (32, 32), color="blue").save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    monkeypatch.setattr(main_module, "_select_reference_paths", lambda mode: [str(reference)])

    response = client.post("/api/filesystem/select-references", params={"mode": "files"})

    assert response.status_code == 200
    body = response.json()
    assert body["selected_paths"] == [str(reference)]


def test_select_folder_uses_native_picker_result(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "training-source"
    selected.mkdir()
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    monkeypatch.setattr(main_module, "_select_folder_path", lambda: str(selected))

    response = client.post("/api/filesystem/select-folder")

    assert response.status_code == 200
    assert response.json()["folder_path"] == str(selected)


def test_analyze_references_returns_character_draft_from_local_images(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    reference = source / "portrait.png"
    Image.new("RGB", (800, 1200), color=(235, 225, 210)).save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Marianna"
    assert "offline image analysis" in body["face_summary"]
    assert "portrait-oriented" in body["style_notes"]
    assert "warm" in body["skin_tone"]
    assert body["consistency_score"] >= 60
    assert body["analysis_images"][0]["width"] == 800
    assert body["analysis_images"][0]["orientation"] == "portrait"
    assert body["analysis_warnings"]


def test_analyze_references_includes_local_vision_caption_when_available(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    reference = source / "portrait.png"
    Image.new("RGB", (800, 1200), color=(235, 225, 210)).save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: ["a woman with dark wavy hair in a white shirt"]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "local vision model caption" in body["face_summary"]
    assert "dark wavy hair" in body["hair"]
    assert "white shirt" in body["style_notes"]
    assert body["analysis_images"][0]["caption"] == "a woman with dark wavy hair in a white shirt"


def test_analyze_references_adds_body_attribute_cues_from_caption(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    reference = source / "beach.png"
    Image.new("RGB", (880, 1168), color=(225, 205, 188)).save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: ["a woman in a white shirt and red bikini on the beach"]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    cues = body["analysis_images"][0]["body_attributes"]
    assert cues["coverage"] == "swimwear"
    assert cues["chest_visibility"] == "covered by swimwear or clothing"
    assert cues["pose_framing"] == "full or upper body visible"
    assert "bikini" in cues["evidence"]
    assert "swimwear" in body["chest"]


def test_analyze_references_returns_full_reference_intelligence_profile(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    front = source / "front.png"
    back = source / "back.png"
    Image.new("RGB", (900, 1400), color=(226, 202, 184)).save(front)
    Image.new("RGB", (900, 1400), color=(210, 190, 174)).save(back)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: [
        "adult woman standing on a beach with long dark wavy hair wearing a red bikini, full body view",
        "adult woman from behind wearing denim shorts and a white shirt, long dark wavy hair, full body view",
    ]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(front), str(back)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    first_details = body["analysis_images"][0]["image_details"]
    assert first_details["hair"]["visibility"] == "clear"
    assert "dark wavy hair" in first_details["hair"]["summary"]
    assert first_details["chest"]["visibility"] == "partial"
    assert first_details["clothing"]["summary"] == "red bikini"
    aggregate = body["aggregate_intelligence"]
    assert aggregate["hair"]["visibility"] == "clear"
    assert "long dark wavy hair" in aggregate["hair"]["summary"]
    assert aggregate["chest"]["visibility"] == "partial"
    assert "swimwear-covered" in aggregate["chest"]["summary"]
    assert aggregate["pose"]["visibility"] == "clear"
    assert "full body" in aggregate["pose"]["summary"]
    assert "Prompt-ready" in aggregate["prompt_summary"]
    assert aggregate["uncertainty_notes"]


def test_analyze_references_treats_large_varied_pack_as_usable(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    references = []
    for index in range(8):
        reference = source / f"marianna-{index}.png"
        size = (900, 1400) if index % 2 == 0 else (1400, 900)
        Image.new("RGB", size, color=(226, 202, 184)).save(reference)
        references.append(reference)

    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: [
        "adult woman face with long dark wavy hair standing full body in a red bikini front view",
        "adult woman face with long dark wavy hair standing full body in denim shorts rear back view",
        "adult woman face with long dark wavy hair standing full body side view in a white shirt",
        "adult woman face with long dark wavy hair standing on the beach in bikini",
        "adult woman face with long dark wavy hair full body in casual shorts",
        "adult woman face with long dark wavy hair standing from behind in denim shorts",
        "adult woman face with long dark wavy hair full body front view",
        "adult woman face with long dark wavy hair standing side view",
    ]
    app.state.tagger = lambda paths: [
        FakeTagReport(
            tags=[
                FakeTag("long_hair", 0.88),
                FakeTag("black_hair", 0.82),
                FakeTag("bikini", 0.81),
                FakeTag("breasts", 0.77),
                FakeTag("from_behind", 0.64 if index in {1, 5} else 0.2),
            ],
            rating="questionable",
            rating_confidence=0.72,
        )
        for index, _ in enumerate(paths)
    ]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "Marianna",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference) for reference in references],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    aggregate = body["aggregate_intelligence"]
    assert aggregate["face"]["visibility"] == "clear"
    assert aggregate["body_shape"]["visibility"] == "clear"
    assert aggregate["waist_hips"]["visibility"] == "clear"
    assert aggregate["chest"]["visibility"] == "clear"
    assert aggregate["uncertainty_notes"] == []
    assert not any("mix orientations" in warning for warning in body["analysis_warnings"])


def test_analyze_references_flags_explicit_adult_content_clinically(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "AdultRefs"
    source.mkdir(parents=True)
    reference = source / "explicit.png"
    Image.new("RGB", (900, 1400), color=(226, 202, 184)).save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: [
        "adult nude woman with visible breasts, nipples, vulva, and buttocks in an explicit sexual pose"
    ]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "adult-ref",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    adult = body["analysis_images"][0]["adult_content"]
    assert adult["nudity_level"] == "explicit nudity"
    assert adult["breast_visibility"] == "visible"
    assert adult["nipple_areola_visibility"] == "visible"
    assert adult["genital_visibility"] == "visible"
    assert adult["buttocks_visibility"] == "visible"
    assert adult["sexual_activity"] == "explicit sexual pose or activity cue"
    assert body["aggregate_intelligence"]["adult_content"]["nudity_level"] == "explicit nudity"


def test_analyze_references_uses_local_tagger_for_stronger_body_signals(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "TaggedRefs"
    source.mkdir(parents=True)
    reference = source / "tagged.png"
    Image.new("RGB", (900, 1400), color=(226, 202, 184)).save(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    app.state.captioner = lambda paths: ["adult woman standing in a studio"]
    app.state.tagger = lambda paths: [
        FakeTagReport(
            tags=[
                FakeTag("nude", 0.92),
                FakeTag("breasts", 0.88),
                FakeTag("nipples", 0.81),
                FakeTag("pussy", 0.79),
                FakeTag("ass", 0.73),
                FakeTag("long_hair", 0.69),
            ],
        )
    ]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "tagged-ref",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference)],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    adult = body["analysis_images"][0]["adult_content"]
    assert adult["nudity_level"] == "explicit nudity"
    assert adult["breast_visibility"] == "visible"
    assert adult["nipple_areola_visibility"] == "visible"
    assert adult["genital_visibility"] == "visible"
    assert adult["buttocks_visibility"] == "visible"
    assert body["analysis_images"][0]["vision_tags"][0]["name"] == "rating: explicit"
    assert any(tag["name"] == "nude" for tag in body["aggregate_intelligence"]["strong_tags"])


def test_local_tagger_prefers_directml_provider_when_available() -> None:
    assert _preferred_providers(["DmlExecutionProvider", "CPUExecutionProvider"], "auto") == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _preferred_providers(["DmlExecutionProvider", "CPUExecutionProvider"], "cpu") == ["CPUExecutionProvider"]


def test_analyze_references_keeps_generated_profile_fields_within_schema_limits(tmp_path: Path) -> None:
    source = tmp_path / "Models" / "Marianna"
    source.mkdir(parents=True)
    references = []
    for index in range(4):
        reference = source / f"portrait-{index}.png"
        Image.new("RGB", (800, 1200), color=(235, 225, 210)).save(reference)
        references.append(reference)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    app = client.app
    long_caption = " ".join(["local vision caption with repeating descriptive details"] * 20)
    app.state.captioner = lambda paths: [long_caption for _ in paths]

    response = client.post(
        "/api/characters/analyze-references",
        json={
            "id": "marianna",
            "display_name": "",
            "age_category": "adult_25_plus",
            "reference_images": [str(reference) for reference in references],
            "lora_files": [],
            "seed_strategy": "vary",
            "locked_seed": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["hair"]) <= 240
    assert len(body["eyes"]) <= 160
    assert len(body["grooming"]) <= 240


def test_training_scan_route_accepts_body_part_image(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (8, 8), color="red").save(source / "sample.png")
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.post(
        "/api/training/scan",
        json={
            "name": "hands",
            "source_folder": str(source),
            "dataset_type": "body_part",
            "face_policy": "body_part_crops_only",
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted_count"] == 1
    assert response.json()["accepted"][0]["stored_path"]


def test_training_config_route_persists_job_for_accepted_directory(tmp_path: Path) -> None:
    accepted_dir = tmp_path / "datasets" / "accepted"
    accepted_dir.mkdir(parents=True)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.post(
        "/api/training/config",
        json={
            "request": {
                "dataset_id": "dataset-1",
                "dataset_path": str(accepted_dir),
                "output_dir": str(tmp_path / "out"),
                "base_model_path": "base.safetensors",
                "lora_name": "ari_style",
            },
            "accepted_image_count": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "dataset-1"
    assert body["accepted_image_count"] == 1
    expected_config_path = tmp_path / "config" / "training" / f"{body['job_id']}.json"
    assert body["config_path"] == str(expected_config_path)
    assert expected_config_path.is_file()


def test_register_lora_route_rejects_missing_file_and_accepts_safetensors(
    tmp_path: Path,
) -> None:
    accepted_dir = tmp_path / "datasets" / "accepted"
    accepted_dir.mkdir(parents=True)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    job = client.post(
        "/api/training/config",
        json={
            "request": {
                "dataset_id": "dataset-1",
                "dataset_path": str(accepted_dir),
                "output_dir": str(tmp_path / "out"),
                "base_model_path": "base.safetensors",
                "lora_name": "ari_style",
            },
            "accepted_image_count": 1,
        },
    ).json()

    missing = client.post(
        f"/api/training/{job['job_id']}/register-lora",
        json={"lora_path": str(tmp_path / "missing.safetensors")},
    )
    assert missing.status_code == 400

    lora_path = tmp_path / "ari_style.safetensors"
    lora_path.write_bytes(b"fake")
    accepted = client.post(
        f"/api/training/{job['job_id']}/register-lora",
        json={"lora_path": str(lora_path)},
    )

    assert accepted.status_code == 200
    assert accepted.json()["completed_lora_path"] == str(lora_path)


def test_register_lora_route_returns_404_for_missing_job_before_missing_file(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))

    response = client.post(
        "/api/training/missing-job/register-lora",
        json={"lora_path": str(tmp_path / "missing.safetensors")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Training job not found."


def test_register_lora_route_rejects_invalid_lora_suffix(tmp_path: Path) -> None:
    accepted_dir = tmp_path / "datasets" / "accepted"
    accepted_dir.mkdir(parents=True)
    client = TestClient(create_app(paths=WorkspacePaths(tmp_path)))
    job = client.post(
        "/api/training/config",
        json={
            "request": {
                "dataset_id": "dataset-1",
                "dataset_path": str(accepted_dir),
                "output_dir": str(tmp_path / "out"),
                "base_model_path": "base.safetensors",
                "lora_name": "ari_style",
            },
            "accepted_image_count": 1,
        },
    ).json()
    invalid_lora_path = tmp_path / "ari_style.txt"
    invalid_lora_path.write_text("fake", encoding="utf-8")

    response = client.post(
        f"/api/training/{job['job_id']}/register-lora",
        json={"lora_path": str(invalid_lora_path)},
    )

    assert response.status_code == 400
    assert "LoRA artifact file must end with" in response.json()["detail"]
