# License And Policy Audit

Verdict: LOCAL MEDIA/PRIVATE-DATA CLEANUP PASSED; DEPENDENCY LICENSE REVIEW STILL NEEDED FOR FORMAL RELEASE

This audit was run for the local repository at `H:\DevWork\Win_Apps\GK_Liidar` with intended license `MIT`.

## High Findings

None identified for tracked generated media, private datasets, model weights, credentials, or the canceled private project name after cleanup.

## Medium Findings

### MEDIUM: Dependency license review still needed

The repository tracks Python and Node dependency manifests. Those packages are not vendored, but a formal public release should still review direct and transitive dependency licenses.

Recommended fix:

- Review `app/backend/pyproject.toml`, `app/backend/uv.lock`, `app/frontend/package.json`, and `app/frontend/package-lock.json`.
- Update `THIRD_PARTY_NOTICES.md` with any required notices before tagging a stable release.

### MEDIUM: GitHub metadata should be verified after push

The local repository is prepared for MIT publication, but GitHub-side metadata should be checked after the final branch sync.

Recommended fix:

- Confirm GitHub displays `MIT license`.
- Confirm `main` is the default branch.
- Use a neutral repo description such as: `Local-first AI image workflow studio`.

## Checks Performed

- Root license check: `LICENSE` added with standard MIT text.
- README consistency: `README.md` updated to describe the current app and public-release boundaries.
- Third-party notices: `THIRD_PARTY_NOTICES.md` added with dependency-manifest locations and release-review requirements.
- Private artifact sweep: removed prior local `.claude/` and `.worktrees/` cache/worktree folders.
- Sensitive media/data sweep: no generated images or datasets are intentionally included after cleanup.
- Model weight sweep: `.gitignore` excludes common model formats and output folders.
- Repo hygiene: public release policy added at `docs/public-release-policy.md`.
- Planning docs: project docs were rewritten as generic local workflow-studio guidance.
- Private name sweep: the canceled private project name was removed from tracked source and tests.

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
