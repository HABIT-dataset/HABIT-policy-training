"""Local sanity check for the bimanual data pipeline.

Loads an env file (same format as samples/envs/.env.habit), runs one sample
through the BimanualDataConfig repack pipeline, and prints the resulting
keys/shapes so you can verify that:

  - STATE_FIELDS + ACTION_FIELDS columns exist in the parquet
  - ConcatColumns yields the expected per-arm concatenated shape
  - RepackTransform produces the `state`, `actions`, `images`, `prompt` keys
  - ImagesToModelFormat produces `image` + `image_mask`
  - Action horizon is populated correctly

Run:
    python samples/check_dataloader.py samples/envs/.env.habit \\
        --dataset-root ~/.cache/openpi-local/datasets/habit_local

No GPU and no model weights required — only the data-side transforms.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

import numpy as np


def _load_env_file(path: pathlib.Path) -> dict[str, str]:
    """Parse a KEY=VALUE shell-style env file into a dict."""
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _prepare_lerobot_env(dataset_root: pathlib.Path, env_repo_id: str) -> str:
    """Make LeRobotDataset resolve `repo_id` to `dataset_root` locally."""
    dataset_root = dataset_root.resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Expected meta/info.json under {dataset_root}. "
            "Point --dataset-root at a LeRobot v2 dataset directory."
        )
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root.parent)
    local_repo_id = dataset_root.name
    print(f"[env] HF_LEROBOT_HOME = {dataset_root.parent}")
    print(f"[env] local repo_id  = {local_repo_id}  (env had: {env_repo_id!r})")
    return local_repo_id


def _describe_leaf(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"tensor  shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, (str, bytes)):
        snippet = value if isinstance(value, str) else value.decode("utf-8", "replace")
        snippet = snippet.replace("\n", " ")
        return f"str     {snippet[:80]!r}"
    if isinstance(value, (int, float, bool, np.floating, np.integer)):
        return f"scalar  {value}"
    return f"{type(value).__name__}"


def _print_sample(label: str, sample: dict, *, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}{label}:")
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"{pad}  {k}:")
            for sk, sv in v.items():
                print(f"{pad}    {sk:<30} {_describe_leaf(sv)}")
        else:
            print(f"{pad}  {k:<32} {_describe_leaf(v)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env_file",
        type=pathlib.Path,
        help="Path to .env file (e.g. samples/envs/.env.habit)",
    )
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("~/.cache/openpi-local/datasets/habit_local").expanduser(),
        help="Local LeRobot v2 dataset dir (contains meta/ and data/).",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--save-images",
        type=pathlib.Path,
        default=None,
        help="Directory to save the post-repack images as PNGs (one per camera).",
    )
    args = parser.parse_args()

    if not args.env_file.exists():
        print(f"Env file not found: {args.env_file}", file=sys.stderr)
        return 1

    env_vars = _load_env_file(args.env_file)
    print(f"[env] Loaded {len(env_vars)} vars from {args.env_file}")

    for k, v in env_vars.items():
        os.environ[k] = v

    local_repo_id = _prepare_lerobot_env(
        args.dataset_root, env_vars.get("DATASET_REPO_ID", "")
    )
    os.environ["DATASET_REPO_ID"] = local_repo_id

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    for p in (repo_root, repo_root / "openpi" / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # CPU-only: this script never needs a GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from local_training.data_config import register_bimanual_config
    from local_training.env_config import parse_env
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
    from openpi import transforms as _transforms

    env = parse_env()
    register_bimanual_config(env, config_name="bimanual_local_sanity")
    train_config = _config.get_config("bimanual_local_sanity")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    print(f"\n[config] model action_dim    = {train_config.model.action_dim}")
    print(f"[config] model action_horizon = {train_config.model.action_horizon}")
    print(f"[config] action_sequence_keys = {data_config.action_sequence_keys}")
    print(f"[config] state_fields         = {env.state_fields}")
    print(f"[config] action_fields        = {env.action_fields}")
    print(f"[config] image_keys           = {env.image_keys}")

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=train_config.model.action_horizon,
        model_config=train_config.model,
    )
    print(f"\n[dataset] length = {len(dataset)} frames")

    raw = dataset[args.sample_index]
    print("\n=== Step 1: raw LeRobot sample (keys actually present in parquet) ===")
    _print_sample("raw", raw)

    print("\n=== Step 2: checking configured STATE_FIELDS / ACTION_FIELDS ===")
    state_cols = [c.strip() for c in env.state_fields.split(",") if c.strip()]
    action_cols = [c.strip() for c in env.action_fields.split(",") if c.strip()]
    missing = [c for c in (state_cols + action_cols) if c not in raw]
    if missing:
        print(f"  MISSING columns in raw sample: {missing}")
        print(f"  Available: {sorted(raw.keys())}")
        return 2
    state_dim = sum(int(np.asarray(raw[c]).shape[-1]) for c in state_cols)
    action_dim_real = sum(int(np.asarray(raw[c]).shape[-1]) for c in action_cols)
    print(f"  state columns present, concatenated dim = {state_dim}")
    print(f"  action columns present, concatenated dim (per step) = {action_dim_real}")

    print("\n=== Step 3: after repack_transforms (Concat + RepackTransform) ===")
    repacked = _transforms.compose(data_config.repack_transforms.inputs)(dict(raw))
    _print_sample("repacked", repacked)

    print("\n=== Step 4: after data_transforms (ImagesToModelFormat) ===")
    shaped = _transforms.compose(data_config.data_transforms.inputs)(repacked)
    _print_sample("model-ready", shaped)

    print("\n=== Step 5: model compatibility checks ===")
    problems: list[str] = []
    state_arr = np.asarray(shaped.get("state", []))
    if state_arr.shape != (state_dim,):
        problems.append(f"state shape {state_arr.shape} != ({state_dim},)")
    actions_arr = np.asarray(shaped.get("actions", []))
    expected_action_shape = (train_config.model.action_horizon, action_dim_real)
    if actions_arr.shape != expected_action_shape:
        problems.append(f"actions shape {actions_arr.shape} != {expected_action_shape}")
    image_dict = shaped.get("image", {})
    mask_dict = shaped.get("image_mask", {})
    expected_cams = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    missing_cams = expected_cams - set(image_dict)
    if missing_cams:
        problems.append(f"missing camera keys: {missing_cams}")
    for cam, arr in image_dict.items():
        a = np.asarray(arr)
        if a.ndim != 3 or a.shape[-1] != 3:
            problems.append(f"image[{cam}] not HWC-RGB, got shape {a.shape}")
    for cam in expected_cams & set(mask_dict):
        if not bool(mask_dict[cam]):
            problems.append(f"image_mask[{cam}] is False — camera should be present")
    if "prompt" not in shaped or not isinstance(shaped["prompt"], (str, bytes)):
        problems.append("missing or non-string `prompt`")

    print(f"  state:   shape = {state_arr.shape}           (expected ({state_dim},))")
    print(f"  actions: shape = {actions_arr.shape}  (expected {expected_action_shape})")
    print(f"  image keys: {sorted(image_dict)}")
    print(f"  image_mask keys: {sorted(mask_dict)}")
    print(f"  prompt: {shaped.get('prompt', '<missing>')!r}")

    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except ImportError:
            print("\n[warn] Pillow not installed; cannot save images. `uv pip install pillow`")
        else:
            for cam, arr in image_dict.items():
                a = np.asarray(arr).astype(np.uint8)
                out = args.save_images / f"frame{args.sample_index:06d}_{cam}.png"
                Image.fromarray(a).save(out)
                print(f"  saved {out}  ({a.shape})")

    if problems:
        print("\nFAIL — issues found:")
        for p in problems:
            print(f"  - {p}")
        return 3

    print("\nOK — data pipeline produces the shapes the model expects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
