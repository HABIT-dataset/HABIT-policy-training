"""Local training entrypoint for OpenPI on a HuggingFace LeRobot v2 dataset.

Pipeline:
    1. Parse env vars (optionally from a .env-style file)
    2. Set HF_LEROBOT_HOME / cache directories under a local root
    3. Download a small subset of a HuggingFace dataset (optional)
    4. Build an OpenPI ``TrainConfig`` for the bimanual data layout
    5. Build norm stats from the dataset's ``meta/stats.json`` (fast path)
       or fall back to ``compute_norm_stats.py``
    6. Run ``openpi/scripts/train.py`` end-to-end

Usage:
    python -m local_training.train_local samples/envs/.env.habit
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _maybe_reexec_under_openpi_venv() -> None:
    """Re-exec the current script under ``openpi/.venv/bin/python`` if needed.

    ``uv sync`` in ``openpi/`` creates an isolated virtualenv at
    ``openpi/.venv/`` with all the JAX/Flax/lerobot/etc. deps. A user running
    ``python -m local_training.train_local`` from a different interpreter
    (e.g. the ``(base)`` conda env) won't see those deps. To keep the example
    one-command, detect that situation and re-exec under the venv's python so
    the user doesn't need to remember to ``source openpi/.venv/bin/activate``.

    Skipped if:
      - the venv doesn't exist (user installed deps a different way), or
      - we're already running under that venv, or
      - flax is already importable from the current interpreter.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    venv_python = repo_root / "openpi" / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    if pathlib.Path(sys.executable).resolve() == venv_python.resolve():
        return
    # Cheap probe: if flax is already on the path, we're good.
    try:
        import flax  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    logger.info(
        "Re-executing under openpi venv: %s (current %s lacks flax)",
        venv_python, sys.executable,
    )
    os.execv(str(venv_python), [str(venv_python), "-m", "local_training.train_local", *sys.argv[1:]])


def _ensure_openpi_on_path() -> None:
    """Make ``openpi`` importable from the vendored source tree.

    The vendored OpenPI lives at ``openpi/src/openpi/`` so that ``uv sync`` in
    ``openpi/`` can install it as an editable package. To keep the example
    runnable from a clean conda env (no ``uv sync`` required for the
    no-training pieces), prepend ``openpi/src`` to ``sys.path`` if openpi
    isn't already importable.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    openpi_src = repo_root / "openpi" / "src"
    if openpi_src.is_dir() and str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))
    # scripts/ also needs to be on sys.path so DataLoader workers can re-import
    # ``train`` and ``compute_norm_stats`` after they spawn.
    scripts_dir = repo_root / "openpi" / "scripts"
    if scripts_dir.is_dir() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # OpenPI requires JAX, Flax, etc. — pinned in openpi/pyproject.toml. If the
    # current Python environment doesn't have them installed, point the user
    # at the setup instructions instead of letting them stumble through one
    # missing module at a time.
    try:
        import openpi.training.config  # noqa: F401
    except ModuleNotFoundError as e:
        msg = (
            f"\n\nFailed to import OpenPI ({e}).\n"
            "OpenPI dependencies (JAX/Flax/optax/lerobot/...) must be installed. From the repo root, run:\n"
            "    cd openpi && uv sync && cd ..\n"
            "and then either re-run this command (it will auto-bootstrap into\n"
            "openpi/.venv/) or invoke openpi/.venv/bin/python -m local_training.train_local ...\n"
            "Alternatively, with pip:  pip install -e openpi"
        )
        raise SystemExit(msg) from e


def _setup_paths(env) -> dict[str, pathlib.Path]:
    """Decide on local cache / checkpoint / asset directories and create them."""
    cache_root = pathlib.Path(env.cache_root or os.path.expanduser("~/.cache/openpi-local")).resolve()
    datasets_dir = cache_root / "datasets"
    checkpoints_dir = pathlib.Path(env.checkpoint_dir or (cache_root / "checkpoints")).resolve()
    assets_dir = pathlib.Path(env.assets_dir or (cache_root / "assets")).resolve()
    weights_dir = cache_root / "weights"
    jax_cache = cache_root / "jax"

    for d in (cache_root, datasets_dir, checkpoints_dir, assets_dir, weights_dir, jax_cache):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_LEROBOT_HOME"] = str(datasets_dir)
    os.environ.setdefault("OPENPI_DATA_HOME", str(cache_root / "openpi"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(jax_cache))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ["PYTHONUNBUFFERED"] = "1"

    return {
        "cache_root": cache_root,
        "datasets": datasets_dir,
        "checkpoints": checkpoints_dir,
        "assets": assets_dir,
        "weights": weights_dir,
    }


def _maybe_download_dataset(env) -> None:
    """Download a small subset of the HuggingFace dataset if requested."""
    if not env.hf_dataset_repo:
        logger.info(
            "HF_DATASET_REPO not set, skipping dataset download "
            "(assumes %s already exists at $HF_LEROBOT_HOME/%s/)",
            env.dataset_repo_id, env.dataset_repo_id,
        )
        return

    from local_training.hf_dataset import download_subset

    cache_root = pathlib.Path(os.environ["HF_LEROBOT_HOME"]).parent
    target_root = pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / env.dataset_repo_id
    if target_root.exists() and (target_root / "meta" / "info.json").exists():
        logger.info("Dataset already staged at %s — skipping download", target_root)
        return

    download_subset(
        hf_repo_id=env.hf_dataset_repo,
        subset=env.hf_dataset_subset,
        num_episodes=env.hf_num_episodes,
        image_keys=env.image_keys,
        cache_root=cache_root,
        local_repo_id=env.dataset_repo_id,
    )


def _build_train_config(env, paths: dict[str, pathlib.Path]):
    from openpi.training import config as _config
    from openpi.training import weight_loaders

    if env.config_name == "bimanual_local":
        from local_training.data_config import register_bimanual_config

        register_bimanual_config(env, config_name="bimanual_local")
        logger.info("Registered bimanual_local config")

    config = _config.get_config(env.config_name)
    logger.info("Loaded base config: %s", env.config_name)

    overrides: dict = {}
    if env.exp_name:
        overrides["exp_name"] = env.exp_name
    if env.batch_size is not None:
        overrides["batch_size"] = env.batch_size
    if env.num_train_steps is not None:
        overrides["num_train_steps"] = env.num_train_steps
    if env.num_workers is not None:
        overrides["num_workers"] = env.num_workers
    if env.save_interval is not None:
        overrides["save_interval"] = env.save_interval
    if env.log_interval is not None:
        overrides["log_interval"] = env.log_interval
    if env.keep_period is not None:
        overrides["keep_period"] = env.keep_period
    if env.fsdp_devices is not None:
        overrides["fsdp_devices"] = env.fsdp_devices
    if env.seed is not None:
        overrides["seed"] = env.seed
    if env.ema_decay is not None:
        overrides["ema_decay"] = env.ema_decay
    if env.wandb_project:
        overrides["project_name"] = env.wandb_project
    overrides["wandb_enabled"] = env.wandb_enabled
    overrides["resume"] = env.resume
    overrides["overwrite"] = env.overwrite
    overrides["checkpoint_base_dir"] = str(paths["checkpoints"])
    overrides["assets_base_dir"] = str(paths["assets"])

    config = dataclasses.replace(config, **overrides)

    if env.weight_loader_path:
        config = dataclasses.replace(
            config,
            weight_loader=weight_loaders.CheckpointWeightLoader(env.weight_loader_path),
        )

    lr_overrides = {}
    if env.learning_rate is not None:
        lr_overrides["peak_lr"] = env.learning_rate
    if env.warmup_steps is not None:
        lr_overrides["warmup_steps"] = env.warmup_steps
    if env.decay_steps is not None:
        lr_overrides["decay_steps"] = env.decay_steps
    if lr_overrides:
        config = dataclasses.replace(
            config,
            lr_schedule=dataclasses.replace(config.lr_schedule, **lr_overrides),
        )

    if env.dataset_repo_id:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, repo_id=env.dataset_repo_id),
        )

    if env.action_dim is not None or env.action_horizon is not None:
        model_overrides = {}
        if env.action_dim is not None:
            model_overrides["action_dim"] = env.action_dim
        if env.action_horizon is not None:
            model_overrides["action_horizon"] = env.action_horizon
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, **model_overrides),
        )

    logger.info(
        "Final config: name=%s exp=%s batch_size=%d steps=%d fsdp=%d",
        config.name, config.exp_name, config.batch_size,
        config.num_train_steps, config.fsdp_devices,
    )
    return config


def _build_norm_stats_from_meta(config, env) -> bool:
    """Build OpenPI's ``norm_stats.json`` from the dataset's ``meta/stats.json``.

    LeRobot v2 datasets include ``meta/stats.json`` with mean/std/min/max/q01/q99
    for every column. We just concatenate the columns for bimanual setups and
    save in OpenPI's expected format. Avoids the slow video-decoding loop in
    ``compute_norm_stats.py`` for the example workflow.

    Returns True on success.
    """
    import json

    dataset_dir = pathlib.Path(os.environ["HF_LEROBOT_HOME"]) / env.dataset_repo_id
    meta_stats_path = dataset_dir / "meta" / "stats.json"
    if not meta_stats_path.exists():
        logger.info("No meta/stats.json at %s, falling back to compute", meta_stats_path)
        return False

    meta_stats = json.loads(meta_stats_path.read_text())
    state_cols = [s.strip() for s in (env.state_fields or "").split(",") if s.strip()]
    action_cols = [s.strip() for s in (env.action_fields or "").split(",") if s.strip()]
    if not state_cols or not action_cols:
        logger.info("STATE_FIELDS / ACTION_FIELDS unset, falling back to compute")
        return False

    def concat_stats(cols: list[str]) -> dict:
        import numpy as np

        out = {"mean": [], "std": [], "q01": [], "q99": []}
        for col in cols:
            if col not in meta_stats:
                raise KeyError(f"Column '{col}' not in meta/stats.json")
            s = meta_stats[col]
            for k in out:
                if k in s:
                    out[k].extend(s[k])
                else:
                    fallback = s.get("min" if k == "q01" else "max", [0.0] * len(s["mean"]))
                    out[k].extend(fallback)
        return {k: np.array(v, dtype=np.float32) for k, v in out.items()}

    logger.info("Building norm stats from %s", meta_stats_path)
    logger.info("  state cols:  %s", state_cols)
    logger.info("  action cols: %s", action_cols)

    from openpi.shared import normalize as _normalize

    norm_stats = {
        "state": _normalize.NormStats(**concat_stats(state_cols)),
        "actions": _normalize.NormStats(**concat_stats(action_cols)),
    }

    asset_id = config.data.assets.asset_id or config.data.repo_id
    assets_dir = config.assets_dirs / asset_id
    _normalize.save(assets_dir, norm_stats)
    logger.info("Wrote norm stats to %s/norm_stats.json", assets_dir)
    return True


def _ensure_norm_stats(config, env) -> None:
    """Make sure norm stats exist for this config.

    Tries the meta/stats.json fast path first, then falls back to
    ``compute_norm_stats.py`` which decodes videos to compute stats from the
    full dataset.
    """
    asset_id = config.data.assets.asset_id or config.data.repo_id
    if not asset_id:
        return

    norm_stats_path = config.assets_dirs / asset_id / "norm_stats.json"

    if not env.use_full_norm_stats:
        if _build_norm_stats_from_meta(config, env):
            return

    if not env.use_full_norm_stats and norm_stats_path.exists():
        logger.info("Using cached norm stats at %s", norm_stats_path)
        return

    if env.use_full_norm_stats:
        logger.info("USE_FULL_NORM_STATS=true → running compute_norm_stats.py on full dataset")
    else:
        logger.info("Falling back to scripts/compute_norm_stats.py")

    import compute_norm_stats

    stats_workers = (
        env.norm_stats_num_workers
        if env.norm_stats_num_workers is not None
        else (env.num_workers or 0)
    )
    compute_norm_stats.main(
        config.name,
        num_workers=stats_workers,
        assets_base_dir=str(config.assets_dirs),
    )
    logger.info("Norm stats computed at %s", norm_stats_path)


def main(argv: list[str] | None = None) -> None:
    # If the user invoked us under a Python that lacks openpi's deps, hop
    # into openpi/.venv/ before doing anything else (fork-safe: this never
    # returns when it re-execs).
    _maybe_reexec_under_openpi_venv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env_file",
        nargs="?",
        default=None,
        help="Path to a .env-style file (KEY=VALUE per line). "
        "If omitted, reads from os.environ only.",
    )
    args = parser.parse_args(argv)

    from local_training.env_config import load_env_file, parse_env

    if args.env_file:
        env_vars = load_env_file(args.env_file)
        for k, v in env_vars.items():
            os.environ.setdefault(k, v)
        logger.info("Loaded %d vars from %s", len(env_vars), args.env_file)

    env = parse_env()
    logger.info("=" * 60)
    logger.info("OpenPI Local Training")
    logger.info("=" * 60)
    logger.info("Config: %s, Exp: %s", env.config_name, env.exp_name)

    _ensure_openpi_on_path()
    paths = _setup_paths(env)
    _maybe_download_dataset(env)
    config = _build_train_config(env, paths)
    _ensure_norm_stats(config, env)

    import train as openpi_train_module

    logger.info("Starting OpenPI training...")
    openpi_train_module.main(config)
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
