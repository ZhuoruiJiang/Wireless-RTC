# Artifact tracking on mininet-wifi

This branch tracks code and experiment artifacts together. `runs/`, `data/`, and `checkpoints/` are deliberately not ignored. Raw runs stay in their existing directories. Small summaries, manifests, selection configurations, and table files remain ordinary Git files. Large per-request/per-tick traces and binary datasets/models use the patterns in `.gitattributes` and require Git LFS.

Environments, dependency caches, temporary build directories, and `.env` credentials remain excluded. This policy concerns research artifacts, not installed dependencies. Before committing a new artifact format, check its size and add an LFS pattern if appropriate. Files above GitHub's regular-file size limit must use LFS. GitHub LFS storage and transfer quotas still apply.

## First server checkout

Install Git LFS using the server's supported package installation method, then, inside the repository:

```bash
git lfs install --local
git fetch origin
git switch --track origin/mininet-wifi
git submodule update --init --recursive
git lfs pull
```

If the branch already exists locally, use `git switch mininet-wifi` followed by `git pull --ff-only`. Preserve any server-side changes before switching or pulling. Do not use a hard reset to resolve divergence.

## Save server-generated artifacts

After an experiment finishes and its outputs are stable:

```bash
git status --short
git add runs/ configs/
# Add data/, checkpoints/, reports/, or other relevant paths when changed.
git diff --cached --stat
git lfs status
git commit -m "Record experiment configuration and results"
git push origin mininet-wifi
```

Review staged paths for secrets and unrelated files. Commit only completed output snapshots, or clearly document an intentional partial/failed run. A `git pull` downloads committed changes; it does not upload server outputs. Git does not automatically track future files simply because their directory is tracked. Include newly produced artifacts in each experiment's commit, together with its configuration and selection metadata.

On another checkout, `git pull --ff-only` and `git lfs pull` retrieve the saved artifacts. Keep `checkpoints/24/policies` if it already exists on the server: those pretrained weights were not present in the Windows checkout and are not supplied by this initial artifact commit.

`reports/midterm/main.tex` is a snapshot of the separately edited report. It does not automatically synchronize with the Windows document editor. `docs/project_proposal.pdf` is a reference copy; the original files were left in place.
