# Codex Interaction Log

This file records the key prompts, plans, implementation decisions, corrections,
and verification results from the Codex-assisted development of this repository.

## 1. Initial Planning

### User Prompt

Read `prompt.md` and provide an implementation plan.

### Codex Response Summary

Codex inspected:

- `prompt.md`
- `CandidateInstructions.md`
- `architecture.jpg`
- `model_architecture.png`
- `weights.csv`
- existing placeholder files under `src/`, `tests/`, and `main_inference.py`

Codex observed that:

- The repository already had a `weights.csv`, but it was ignored by git.
- Implementation files were placeholders.
- The images were RGB PNGs under `images/noisy` and `images/gt`.
- The model architecture and weight names in `weights.csv` matched the translated
  diagram in `model_architecture.png`.

Codex proposed a plan to:

- Implement the PyTorch model in `src/denoiser/model/denoise_model.py`.
- Convert `weights.csv` to `checkpoint.pt`.
- Run inference from a file or directory using `DataLoader`.
- Compute PSNR and SSIM metrics.
- Add a pytest regression test for `images/noisy/00.png`.

The proposed public interfaces were:

- `DenoiseModel`
- `convert_weights_csv_to_checkpoint(...)`
- `load_model(...)`
- `ImageDenoisingDataset`
- `main_inference.py` CLI

## 2. First Implementation

### User Prompt

Implement the proposed plan.

### Codex Actions

Codex implemented:

- PyTorch model and weight-loading utilities in
  `src/denoiser/model/denoise_model.py`.
- Package initializers under `src/`.
- Inference CLI in `main_inference.py`.
- Metric computation in `src/evaluation/compute_metrics.py`.
- Regression test in `tests/test_inference.py`.
- `.gitignore` entries for generated artifacts.

The initial implementation preserved parameter names such as:

```text
model.prior_net.m_head.weight
model.prior_net.m_down1.0.res.0.weight
model.prior_net.m_up3.0.weight
model.prior_net.m_tail.weight
```

This was necessary so the `weights.csv` layer names could be mapped strictly to
the PyTorch `state_dict`.

### Verification

Codex ran inference and metrics. The first pass showed SSIM improvement but a
PSNR drop, indicating a likely architecture interpretation issue:

```text
Average noisy PSNR:    28.2345
Average denoised PSNR: 26.2164
Average noisy SSIM:    0.5397
Average denoised SSIM: 0.6149
```

Codex then tested several in-memory skip-connection variants and found that
using post-downsample encoder tensors improved both metrics.

After correction, metrics improved:

```text
Average noisy PSNR:    28.2345
Average denoised PSNR: 29.7760
Average improvement:   +1.5415 dB
Average noisy SSIM:    0.5397
Average denoised SSIM: 0.7275
Average improvement:   +0.1878
```

`uv run pytest` passed:

```text
1 passed
```

## 3. Skip-Connection Correction

### User Prompt

The model implementation is mostly correct, but the skip connection from
downsampling to upsampling appears to be placed incorrectly. The skip should
start from the downsampling output after the `Conv2d 2x2 stride 2` and connect
to the upsampling path after the two residual blocks, not immediately after
pixel shuffle. Verify carefully.

### Codex Response Summary

Codex confirmed the issue.

The previous code added skips too early, immediately after pixel shuffle:

```python
up3 = self.m_up3[1](self.m_up3[0](body)) + down2
up3 = self.m_up3[2](up3)
up3 = self.m_up3[3](up3)
```

Codex updated the decoder path so each skip is added after the matching decoder
`Res x2` block:

```python
up3 = self.m_up3[1](self.m_up3[0](body))
up3 = self.m_up3[2](up3)
up3 = self.m_up3[3](up3) + down2
```

The same pattern was applied to `up2` and `up1`.

### Verification

Codex regenerated outputs, recomputed metrics, and reran tests.

Final average metrics after this correction:

```text
Average noisy PSNR:    28.2345
Average denoised PSNR: 32.8445
Average improvement:   +4.6099 dB
Average noisy SSIM:    0.5397
Average denoised SSIM: 0.8146
Average improvement:   +0.2749
```

`uv run pytest` passed:

```text
1 passed
```

## 4. Model Simplification

### User Prompt

The implementation is correct but can be simplified. The three downsampling
stages can be simplified by defining one `down_stage` method and calling it with
different channel sizes. The same simplification can be done for upsampling.

### Codex Actions

Codex refactored `PriorNet` by adding:

- `_make_down_stage(in_channels, out_channels)`
- `_make_up_stage(in_channels, out_channels)`
- `_run_down_stage(stage, x)`
- `_run_up_stage(stage, x, skip)`

The refactor preserved the exact `state_dict` names and shapes, so the existing
CSV/checkpoint mapping remained compatible.

### Verification

Codex printed the state dict names and shapes to confirm compatibility with
`weights.csv`.

`uv run pytest` passed:

```text
1 passed
```

## 5. README And GitHub Publication

### User Prompt

Complete the README and push the repository to the user's GitHub repo named
`ImageDenoiser`. Make it public.

### Codex Actions

Codex completed `README.md` with:

- Project summary.
- Repository layout.
- Setup instructions with `uv`.
- Model and weight conversion notes.
- Inference commands.
- Metrics commands.
- Test instructions.
- Final benchmark metrics.

Codex also updated `.gitignore` to keep generated artifacts out of git:

- `weights.csv`
- `checkpoint.pt`
- `outputs/`
- `.matplotlib-cache/`
- `.pytest_cache/`
- `.DS_Store`

Codex created the initial commit:

```text
decd7cc Initial image denoiser implementation
```

Codex then created and pushed the public GitHub repository:

```text
https://github.com/mxz2013/ImageDenoiser
```

GitHub visibility was verified as:

```text
PUBLIC
```

## 6. Current Request

### User Prompt

Save this conversation into `codex_log.md`, commit it, and push it to GitHub.

### Codex Action

This file was created to serve as the LLM interaction log for the project. It
summarizes the prompts, implementation steps, architecture corrections,
verification commands, metric outcomes, and GitHub publication steps.

## Final Verification Commands Used

```bash
uv run python main_inference.py \
  --input images/noisy \
  --output outputs/denoised \
  --checkpoint checkpoint.pt \
  --batch-size 4 \
  --device cpu
```

```bash
uv run python -m src.evaluation.compute_metrics \
  --gt-dir images/gt \
  --noisy-dir images/noisy \
  --denoised-dir outputs/denoised \
  --output-dir outputs/metrics
```

```bash
uv run pytest
```

