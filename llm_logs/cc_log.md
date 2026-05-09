# Claude Code Interaction Log

This file records the key prompts, plans, implementation decisions, and
verification results from the Claude Code-assisted review and improvement of
this repository. The session picked up after the initial Codex implementation
(see `codex_log.md`).

---

## 1. Repository Initialisation

### User Prompt

Analyse this codebase and create a `CLAUDE.md` file covering common commands,
high-level architecture, and any other guidance useful to future Claude Code
sessions.

### Claude Actions

Claude launched two parallel Explore agents to survey:

- Top-level directory structure, `pyproject.toml`, test configuration, and the
  absence of CI, linting, and an existing `CLAUDE.md`.
- Source architecture: `src/denoiser/model/denoise_model.py`,
  `src/evaluation/compute_metrics.py`, `main_inference.py`, and
  `tests/test_inference.py`.

`CLAUDE.md` was written to the repository root. It documents:

- `uv sync` as the only setup step.
- The local-only requirement for `weights.csv` and `checkpoint.pt`.
- The three key commands (inference, metrics, tests).
- The source layout and the role of each file.
- The `PriorNet` U-Net architecture (head → encoder × 3 → bottleneck → decoder
  × 3 → tail with image skip).
- The CSV weight transposition (`(in,out,kH,kW)` → PyTorch `(out,in,kH,kW)`).
- The end-to-end data and inference flow.
- Test dependencies on real image files and the auto-generated checkpoint.

---

## 2. Code Review

### User Prompt

Go through the Codex solution carefully and suggest any improvements if
necessary. Do **not** modify the code — list all improvement plans first.

### Claude Response Summary

Claude read all four source files in full and identified ten issues across
several severity levels.

**Bug risks / correctness**

1. `torch.load` called without `weights_only=True` — triggers a `FutureWarning`
   in PyTorch ≥ 2.x; the default will flip in a future release, breaking
   checkpoint loading.
   — `denoise_model.py:302`

2. CSV index validation only checks the first and last index value; gaps or
   duplicates in the middle would produce silently wrong weights.
   — `denoise_model.py:250`

**Architecture / design**

3. `_run_down_stage` and `_run_up_stage` access `nn.Sequential` children by
   magic integer index (`stage[0]`, `stage[1]`, …). Any layer insertion shifts
   all indices silently.
   — `denoise_model.py:109–120`

4. `PixelShuffleRelu` is a dedicated class that wraps only two standard PyTorch
   modules with no additional logic. It can be inlined as
   `nn.Sequential(nn.PixelShuffle(2), nn.ReLU(inplace=True))`.
   — `denoise_model.py:34–51`

**Robustness**

5. No validation that input image dimensions are divisible by 8, as required by
   the three stride-2 downsampling stages. Non-compliant inputs produce a
   cryptic error inside the model.

6. No per-image progress feedback during inference; the loop is silent for its
   entire duration.

7. `os.environ.setdefault("MPLCONFIGDIR", ...)` runs at module import time,
   mutating the process environment whenever `compute_metrics` is imported, even
   if no plot is ever produced.
   — `compute_metrics.py:14`

**Minor / polish**

8. Double clamping: `run_inference` clamps to `[0, 1]` and `save_image_tensor`
   clamps again. *(User chose to keep this — see Section 3.)*

9. Hardcoded relative `Path("images/noisy/00.png")` in the test assumes pytest
   is always run from the project root.
   — `tests/test_inference.py:17–20`

10. `DataLoader` does not set `pin_memory=True`, which accelerates host-to-device
    transfer when inference runs on CUDA.
    — `main_inference.py:103`

---

## 3. Implementation of Improvements

### User Prompt

I prefer to have double clamping so I want to keep 8. You can improve the rest
now.

### Claude Actions

Claude applied nine improvements across five files.

**`src/denoiser/model/denoise_model.py`**

- Added `import numpy as np`.
- Replaced `PixelShuffleRelu(2)` in `_make_up_stage` with the inline
  `nn.Sequential(nn.PixelShuffle(2), nn.ReLU(inplace=True))` and deleted the
  `PixelShuffleRelu` class entirely. Neither `nn.PixelShuffle` nor `nn.ReLU`
  carry learnable parameters, so no weight names were affected.
- Replaced integer-index access in `_run_down_stage` and `_run_up_stage` with
  tuple unpacking:

  ```python
  # encoder
  res1, res2, downsample = stage
  return downsample(res2(res1(x)))

  # decoder
  conv, upsample, res1, res2 = stage
  return res2(res1(upsample(conv(x)))) + skip
  ```

  Unpacking raises `ValueError` immediately if the stage structure changes.

- Strengthened the CSV index validation from a boundary check to a full
  sequence check:

  ```python
  if not np.array_equal(index_values, np.arange(expected_count)):
      raise ValueError(...)
  ```

- Added `weights_only=True` to `torch.load`:

  ```python
  payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  ```

**`main_inference.py`**

- Added a size guard in `ImageDenoisingDataset.__getitem__` that raises a
  clear `ValueError` when width or height is not divisible by 8.
- Added `logger.info("  %s → %s", name, output_path)` after each saved image.
- Added `pin_memory=selected_device.type == "cuda"` to the `DataLoader`.

**`src/evaluation/compute_metrics.py`**

- Moved `os.environ.setdefault("MPLCONFIGDIR", ...)` and `import matplotlib.pyplot as plt`
  from module level into the body of `plot_metrics()`. Matplotlib is now
  imported lazily, only when a plot is actually requested.

**`tests/conftest.py`** *(new file)*

- Added a session-scoped `project_root` fixture anchored to `__file__`:

  ```python
  @pytest.fixture(scope="session")
  def project_root() -> Path:
      return Path(__file__).resolve().parent.parent
  ```

**`tests/test_inference.py`**

- Updated the test function signature to accept `project_root: Path` and
  replaced all bare `Path("...")` literals with `project_root / "..."`.

### Verification

```text
uv run pytest -v

1 passed in 2.72s
```

---

## 4. Logging

### User Prompt

Change all prints by Logging.

### Claude Actions

Replaced every `print` call in `main_inference.py` and
`src/evaluation/compute_metrics.py` with structured logging.

**Pattern applied to both modules**

```python
import logging
...
logger = logging.getLogger(__name__)
```

`%`-style format strings were used throughout so that argument interpolation is
deferred and skipped entirely when the effective log level is above `INFO`.

**`main_inference.py`**

```python
# per-image progress (inside run_inference)
logger.info("  %s → %s", name, output_path)

# summary (inside main)
logger.info("Saved %d denoised image(s) to %s", len(saved_paths), Path(args.output))
```

`main()` calls `logging.basicConfig(level=logging.INFO, format="%(message)s")`
so the CLI output is visually identical to the previous `print` output.

**`src/evaluation/compute_metrics.py`**

```python
logger.info(dataframe.tail(1).to_string(index=False))
logger.info("Saved metrics to %s", csv_path)
logger.info("Saved plot to %s", plot_path)
```

`main()` also calls `logging.basicConfig(level=logging.INFO, format="%(message)s")`.

### Verification

```text
uv run pytest -v

1 passed in 2.54s
```
