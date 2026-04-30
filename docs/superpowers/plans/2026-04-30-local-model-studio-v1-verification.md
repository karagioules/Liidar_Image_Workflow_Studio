# Local Model Studio V1 Verification

Date: 2026-04-30

## Commands

- `py -3.12 -m pytest -v`
- `npm test`
- `npm run build`
- `npm audit --audit-level=moderate`
- `powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup\check-system.ps1`
- Backend smoke request: `http://127.0.0.1:8000/api/runtime`
- Frontend smoke request: `http://127.0.0.1:5173`

## Result

- Backend tests: 65 passed.
- Frontend tests: 5 passed.
- Frontend build: passed.
- Frontend audit: 0 vulnerabilities at moderate threshold.
- System check: Windows 11 Pro, Python 3.12.0, AMD Radeon RX 9070 XT driver 32.0.23033.1002, 31.9 GB RAM, ComfyUI found.
- Backend smoke request: HTTP 200.
- Frontend smoke request: HTTP 200.

## Notes

ComfyUI is cloned locally under `ComfyUI/`, which is ignored by git. Model/checkpoint files still need to be placed under ComfyUI's model folders before real image generation can run.
