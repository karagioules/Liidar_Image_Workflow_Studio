from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Lock, RLock, Thread
from uuid import uuid4

from local_model_studio.dataset_prepper import prepare_dataset_crops
from local_model_studio.schemas import DatasetPrepJobStatus, DatasetPrepRequest, DatasetPrepResponse


class DatasetPrepJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, _DatasetPrepJob] = {}
        self._lock = Lock()

    def start(self, request: DatasetPrepRequest, *, anthropic_api_key: str | None) -> DatasetPrepJobStatus:
        job = _DatasetPrepJob(request, anthropic_api_key=anthropic_api_key)
        with self._lock:
            self._jobs[job.job_id] = job
        status = job.snapshot()
        job.start()
        return status

    def get(self, job_id: str) -> DatasetPrepJobStatus:
        return self._job(job_id).snapshot()

    def cancel(self, job_id: str) -> DatasetPrepJobStatus:
        job = self._job(job_id)
        job.cancel()
        return job.snapshot()

    def _job(self, job_id: str) -> "_DatasetPrepJob":
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError("Dataset prep job not found.") from exc


class _DatasetPrepJob:
    def __init__(self, request: DatasetPrepRequest, *, anthropic_api_key: str | None) -> None:
        self.job_id = uuid4().hex
        self.request = request
        self.anthropic_api_key = anthropic_api_key
        self.cancel_event = Event()
        self.lock = RLock()
        self.status = "queued"
        self.total_count = 0
        self.processed_count = 0
        self.cropped_count = 0
        self.skipped_count = 0
        self.ai_attempted_count = 0
        self.ai_guided_count = 0
        self.ai_failed_count = 0
        self.face_guided_count = 0
        self.fallback_count = 0
        self.active_file: str | None = None
        self.output_folder: str | None = None
        self.error: str | None = None
        self.result: DatasetPrepResponse | None = None
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at

    def start(self) -> None:
        thread = Thread(target=self._run, name=f"dataset-prep-{self.job_id[:8]}", daemon=True)
        thread.start()

    def cancel(self) -> None:
        with self.lock:
            self.cancel_event.set()
            if self.status in {"queued", "running"}:
                self.status = "cancelling"
                self.updated_at = datetime.now(UTC)

    def snapshot(self) -> DatasetPrepJobStatus:
        with self.lock:
            return DatasetPrepJobStatus(
                job_id=self.job_id,
                status=self.status,
                total_count=self.total_count,
                processed_count=self.processed_count,
                cropped_count=self.cropped_count,
                skipped_count=self.skipped_count,
                ai_attempted_count=self.ai_attempted_count,
                ai_guided_count=self.ai_guided_count,
                ai_failed_count=self.ai_failed_count,
                face_guided_count=self.face_guided_count,
                fallback_count=self.fallback_count,
                active_file=self.active_file,
                output_folder=self.output_folder,
                use_ai=self.request.use_ai,
                ai_max_images=self.request.ai_max_images,
                cancel_requested=self.cancel_event.is_set(),
                error=self.error,
                result=self.result,
                created_at=self.created_at,
                updated_at=self.updated_at,
            )

    def _run(self) -> None:
        with self.lock:
            self.status = "running"
            self.updated_at = datetime.now(UTC)
        try:
            result = prepare_dataset_crops(
                self.request,
                anthropic_api_key=self.anthropic_api_key,
                progress_callback=self._update_progress,
                cancel_event=self.cancel_event,
            )
            with self.lock:
                self.result = result
                self.status = "cancelled" if self.cancel_event.is_set() else "completed"
                self.active_file = None
                self.output_folder = result.output_folder
                self.updated_at = datetime.now(UTC)
        except Exception as exc:
            with self.lock:
                self.status = "cancelled" if self.cancel_event.is_set() else "failed"
                self.error = str(exc)
                self.active_file = None
                self.updated_at = datetime.now(UTC)

    def _update_progress(self, progress: dict[str, object]) -> None:
        with self.lock:
            for key in (
                "total_count",
                "processed_count",
                "cropped_count",
                "skipped_count",
                "ai_attempted_count",
                "ai_guided_count",
                "ai_failed_count",
                "face_guided_count",
                "fallback_count",
            ):
                value = progress.get(key)
                if isinstance(value, int):
                    setattr(self, key, value)
            active_file = progress.get("active_file")
            output_folder = progress.get("output_folder")
            self.active_file = active_file if isinstance(active_file, str) else None
            self.output_folder = output_folder if isinstance(output_folder, str) else self.output_folder
            self.updated_at = datetime.now(UTC)
