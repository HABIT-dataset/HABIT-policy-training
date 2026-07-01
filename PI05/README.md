# Local OpenPI Training on a HuggingFace LeRobot v2 Dataset

This repository contains a single-machine training setup for
[OpenPI](openpi/README.md) (PI0 / PI0.5 / PI0-FAST) on a HuggingFace LeRobot
v2 dataset.

The example pipeline targets the
[`configint/HABIT`](https://huggingface.co/datasets/configint/HABIT)
dataset (a bimanual human-robot interaction dataset on the Hub). A small
subset (~10 episodes) is downloaded automatically for a quick local sanity
check; the same scripts work on the full dataset by raising `HF_NUM_EPISODES`.

## Layout

```
PI05/
├── openpi/                  # OpenPI source (vendored, unchanged)
├── local_training/          # Local training wrapper
│   ├── env_config.py        # Parses .env-style files into an EnvConfig
│   ├── data_config.py       # BimanualDataConfig: LeRobot v2 -> OpenPI inputs
│   ├── hf_dataset.py        # Downloads a subset of a HuggingFace LeRobot dataset
│   └── train_local.py       # Entrypoint: stage data, build config, train
└── samples/
    ├── envs/.env.habit      # Example config for HABIT (10 episodes, PI0.5)
    └── check_dataloader.py  # CPU-only data-pipeline sanity check
```

## Setup

The OpenPI dependencies (JAX-CUDA, Flax, lerobot, etc.) are pinned in
`openpi/pyproject.toml`. Install them once:

```bash
cd openpi && uv sync && cd ..
```

This creates `openpi/.venv/`. You don't need to activate it — `train_local.py`
auto-bootstraps into that venv if it's invoked under a Python that doesn't
have the deps (e.g. your `(base)` conda env). Alternatively, run
`source openpi/.venv/bin/activate` once per shell.

If you don't have `uv`: `pip install -e openpi` works too, but follow
`openpi/README.md` for the right CUDA-enabled JAX wheel.

## Quick start: 200-step PI0.5 fine-tune on a 10-episode HABIT subset

```bash
python -m local_training.train_local samples/envs/.env.habit
```

This will:

1. Stage the local cache under `~/.cache/openpi-local/` (datasets,
   checkpoints, JAX cache, weights). Override with `CACHE_ROOT=/path/to/scratch`.
2. Download 10 episodes of the HABIT `sample` subset
   (parquet + the three robot-mounted MP4 views) into
   `~/.cache/openpi-local/datasets/habit_local/`.
3. Compute norm stats from the dataset's `meta/stats.json` (fast path; no
   video decoding).
4. Pull the public PI0.5 base weights from `gs://openpi-assets/checkpoints/`.
5. Run 200 fine-tuning steps via `openpi/scripts/train.py`.

Checkpoints land under `~/.cache/openpi-local/checkpoints/bimanual_local/<EXP_NAME>/`.

## Data pipeline sanity check (no GPU)

If you only want to confirm that your `STATE_FIELDS` / `ACTION_FIELDS` /
`IMAGE_KEYS` mapping is correct without firing up training:

```bash
# After train_local has staged the dataset once:
python samples/check_dataloader.py samples/envs/.env.habit \
    --dataset-root ~/.cache/openpi-local/datasets/habit_local \
    --save-images /tmp/habit_check
```

The script prints, for one frame, the raw parquet keys, the post-repack
sample, and the final model-ready dict (`state`, `actions`, `image`,
`image_mask`, `prompt`). Camera frames are saved as PNGs if `--save-images`
is given.

## Adapting to another LeRobot v2 dataset

`local_training` is intentionally generic. To point it at a different
dataset, edit a copy of `samples/envs/.env.habit`:

| Variable | Meaning |
| --- | --- |
| `HF_DATASET_REPO` | HuggingFace dataset repo id (e.g. `lerobot/aloha_sim_transfer_cube_human`). |
| `HF_DATASET_SUBSET` | Sub-directory of the repo (e.g. `sample`, `full`, or empty). |
| `HF_NUM_EPISODES` | Number of episodes to stage locally (chunk-000 only). |
| `DATASET_REPO_ID` | Local repo_id used by LeRobot under `$HF_LEROBOT_HOME/`. |
| `IMAGE_KEYS` | `"target_name:source_path"` pairs mapping LeRobot video keys onto the model's three image slots (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). |
| `STATE_FIELDS` | Comma-separated parquet columns concatenated to form `state`. |
| `ACTION_FIELDS` | Comma-separated parquet columns concatenated to form `actions`. |
| `ACTION_DIM`, `ACTION_HORIZON` | Architectural model slots. |
| `MODEL_VARIANT` | `pi0`, `pi05`, or `pi0_fast`. |
| `WEIGHT_LOADER_PATH` | Pretrained checkpoint URI (local path, `gs://`, or empty). |

For datasets with chunks larger than 1000 episodes, extend
`local_training/hf_dataset.py::_allow_patterns` to compute the chunk index
from `info.json["chunks_size"]`.

## License

This repository vendors OpenPI under its original Apache 2.0 license (see
`openpi/LICENSE` and `openpi/LICENSE_GEMMA.txt`). The local_training/ code is
provided under the same terms.
