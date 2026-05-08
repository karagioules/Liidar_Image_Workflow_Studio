# License And Policy Audit

Verdict: PASS FOR LOCAL REPOSITORY, NEEDS EXTERNAL HOSTING CHECK

This audit was run for the local repository at `H:\DevWork\Win_Apps\GK_Liidar` with intended license `MIT`.

## High Findings

None identified in the sanitized working tree.

## Medium Findings

### MEDIUM: GitHub metadata not verified

The local repository is prepared for MIT release, but GitHub-side metadata was not checked from this environment. Before publishing, verify the GitHub repository license detection, description, topics, and visibility settings.

Recommended fix:

- Push the `LICENSE` file.
- Confirm GitHub displays `MIT license`.
- Use a neutral repo description such as: `Local-first AI image workflow studio scaffold`.

### MEDIUM: Third-party dependency compliance must be revisited when code is added

The current repo does not track dependency manifests or bundled binaries. Future image engines, Python packages, Node packages, model files, or installer dependencies can introduce license obligations.

Recommended fix:

- Update `THIRD_PARTY_NOTICES.md` whenever dependencies are added.
- Do not bundle model files unless their licenses explicitly permit redistribution.

## Low Findings

### LOW: App folders are placeholders

The `app/backend` and `app/frontend` folders currently contain no implementation files. This is acceptable for an early scaffold, but the README should not imply that the app is production-ready.

Status:

- README states that the repository is an early planning/scaffold project.

## Checks Performed

- Root license check: `LICENSE` added with standard MIT text.
- README consistency: `README.md` added and aligned with MIT/scaffold status.
- Third-party notices: `THIRD_PARTY_NOTICES.md` added.
- Private artifact sweep: removed the prior `tmp/` experiment folder plus local `.claude/` and `.worktrees/` cache/worktree folders.
- Sensitive media/data sweep: no generated images or datasets are intentionally included after cleanup.
- Model weight sweep: `.gitignore` excludes common model formats and output folders.
- Repo hygiene: public release policy added at `docs/public-release-policy.md`.
- Planning docs: project docs were rewritten as generic local workflow-studio guidance.
- Final local scan: no legacy private/sensitive project terms found outside ignored virtualenv/dependency cache, no generated media/model/archive/database files found outside ignored virtualenv/dependency cache, and only the main Git worktree remains registered.

## Final Local Checks

Run these before publishing or tagging a release:

```powershell
git status --short
rg -n -i "<legacy private project terms>" .
Get-ChildItem -Recurse -File | Where-Object {
  $_.Extension -match '^\.(png|jpg|jpeg|webp|gif|mp4|mov|safetensors|ckpt|pt|pth|onnx|zip|7z|rar|sqlite|db)$'
}
```

Expected result:

- No private/persona-specific files.
- No generated sensitive images or datasets.
- No model weights.
- No credentials or cloud endpoint details.
