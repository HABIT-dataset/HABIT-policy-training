"""Parse environment variables into a structured config for local OpenPI training.

Configuration is sourced from a `.env`-style file (or the process environment)
and converted into an ``EnvConfig`` dataclass. The training entrypoint
(``local_training.train_local``) then turns this into an OpenPI ``TrainConfig``.
"""

import os
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Parsed environment configuration for local OpenPI training."""

    # OpenPI config selection
    config_name: str = "bimanual_local"
    exp_name: str = ""

    # Data
    # repo_id used by LeRobot to locate the dataset under HF_LEROBOT_HOME.
    dataset_repo_id: str = "habit_local"

    # Optional: download a small subset of a HuggingFace LeRobot v2 dataset
    # before training. Leave hf_dataset_repo empty to skip download (assumes
    # the dataset already exists at $HF_LEROBOT_HOME/<dataset_repo_id>/).
    hf_dataset_repo: str = ""
    hf_dataset_subset: str = ""
    hf_num_episodes: int = 10

    # Local cache directories
    cache_root: str = ""
    checkpoint_dir: str = ""
    assets_dir: str = ""

    # Weight loading
    weight_loader_path: str = ""

    # Training hyperparameters
    batch_size: int | None = None
    num_train_steps: int | None = None
    num_workers: int | None = None
    norm_stats_num_workers: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    keep_period: int | None = None
    fsdp_devices: int | None = None
    seed: int | None = None

    # Learning rate
    learning_rate: float | None = None
    warmup_steps: int | None = None
    decay_steps: int | None = None

    # Flags
    resume: bool = False
    overwrite: bool = False
    wandb_enabled: bool = False
    wandb_project: str = ""

    # EMA
    ema_decay: float | None = None

    # Data schema (for the bimanual data config)
    action_fields: str = ""
    state_fields: str = ""
    image_keys: str = ""
    default_prompt: str = ""
    action_dim: int | None = None
    action_horizon: int | None = None

    # Model variant: "pi0", "pi05", or "pi0_fast".
    model_variant: str = "pi05"

    # If True, force mean/std (z-score) normalization regardless of model_type.
    use_min_max_norm: bool = False

    # If True, skip the fast meta/stats.json copy and run the full
    # compute_norm_stats.py loop. Only useful for datasets where the per-file
    # sampled stats produce degenerate quantiles.
    use_full_norm_stats: bool = False


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_env() -> EnvConfig:
    """Parse environment variables into ``EnvConfig``."""
    return EnvConfig(
        config_name=os.getenv("OPENPI_CONFIG_NAME", "bimanual_local"),
        exp_name=os.getenv("EXP_NAME", ""),
        dataset_repo_id=os.getenv("DATASET_REPO_ID", "habit_local"),
        hf_dataset_repo=os.getenv("HF_DATASET_REPO", ""),
        hf_dataset_subset=os.getenv("HF_DATASET_SUBSET", ""),
        hf_num_episodes=int(os.getenv("HF_NUM_EPISODES", "10")),
        cache_root=os.getenv("CACHE_ROOT", ""),
        checkpoint_dir=os.getenv("CHECKPOINT_DIR", ""),
        assets_dir=os.getenv("ASSETS_DIR", ""),
        weight_loader_path=os.getenv("WEIGHT_LOADER_PATH", ""),
        batch_size=_parse_int(os.getenv("BATCH_SIZE")),
        num_train_steps=_parse_int(os.getenv("NUM_TRAIN_STEPS")),
        num_workers=_parse_int(os.getenv("NUM_WORKERS")),
        norm_stats_num_workers=_parse_int(os.getenv("NORM_STATS_NUM_WORKERS")),
        save_interval=_parse_int(os.getenv("SAVE_INTERVAL")),
        log_interval=_parse_int(os.getenv("LOG_INTERVAL")),
        keep_period=_parse_int(os.getenv("KEEP_PERIOD")),
        fsdp_devices=_parse_int(os.getenv("FSDP_DEVICES")),
        seed=_parse_int(os.getenv("SEED")),
        learning_rate=_parse_float(os.getenv("LEARNING_RATE")),
        warmup_steps=_parse_int(os.getenv("WARMUP_STEPS")),
        decay_steps=_parse_int(os.getenv("DECAY_STEPS")),
        resume=_parse_bool(os.getenv("RESUME", "false")),
        overwrite=_parse_bool(os.getenv("OVERWRITE", "false")),
        wandb_enabled=_parse_bool(os.getenv("WANDB_ENABLED", "false")),
        wandb_project=os.getenv("WANDB_PROJECT", ""),
        ema_decay=_parse_float(os.getenv("EMA_DECAY")),
        action_fields=os.getenv("ACTION_FIELDS", ""),
        state_fields=os.getenv("STATE_FIELDS", ""),
        image_keys=os.getenv("IMAGE_KEYS", ""),
        default_prompt=os.getenv("DEFAULT_PROMPT", ""),
        action_dim=_parse_int(os.getenv("ACTION_DIM")),
        action_horizon=_parse_int(os.getenv("ACTION_HORIZON")),
        model_variant=os.getenv("MODEL_VARIANT", "pi05"),
        use_min_max_norm=_parse_bool(os.getenv("USE_MIN_MAX_NORM", "false")),
        use_full_norm_stats=_parse_bool(os.getenv("USE_FULL_NORM_STATS", "false")),
    )


def load_env_file(path: str) -> dict[str, str]:
    """Parse a KEY=VALUE shell-style env file into a dict."""
    env: dict[str, str] = {}
    with open(path) as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env
