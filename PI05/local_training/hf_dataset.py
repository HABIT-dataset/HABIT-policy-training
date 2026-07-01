"""Download a small subset of a HuggingFace LeRobot v2 dataset for local training.

Stages a clean ``LeRobot v2`` directory tree under
``$HF_LEROBOT_HOME/<local_repo_id>/`` containing exactly ``num_episodes``
parquet files plus the matching video files for the requested image views.
``meta/info.json`` and ``meta/episodes.jsonl`` are patched so the resulting
directory looks like a real (smaller) LeRobot v2 dataset.

This is intentionally minimal and works without authentication for public
datasets.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
from typing import Iterable

logger = logging.getLogger(__name__)


def _video_keys_from_image_keys(image_keys: str) -> list[str]:
    """Extract LeRobot video keys (the ``observation.images.*`` source paths)
    from the comma-separated env-style image_keys spec.
    """
    out: list[str] = []
    for pair in image_keys.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            _, source = pair.split(":", 1)
        else:
            source = pair
        source = source.strip()
        # Only video features live under observation.images; skip anything else.
        if source.startswith("observation.images."):
            out.append(source)
    return out


def _allow_patterns(
    subset: str,
    num_episodes: int,
    video_keys: Iterable[str],
) -> list[str]:
    prefix = f"{subset}/" if subset else ""
    patterns = [
        f"{prefix}meta/info.json",
        f"{prefix}meta/episodes.jsonl",
        f"{prefix}meta/tasks.jsonl",
        f"{prefix}meta/stats.json",
        f"{prefix}meta/modality.json",
    ]
    for i in range(num_episodes):
        # NOTE: chunks_size on HABIT is 1000, so episodes 0..9 all live in
        # chunk-000. If a dataset uses smaller chunks, this would need to be
        # generalized via info.json — overkill for the example.
        patterns.append(f"{prefix}data/chunk-000/episode_{i:06d}.parquet")
        for vk in video_keys:
            patterns.append(f"{prefix}videos/chunk-000/{vk}/episode_{i:06d}.mp4")
    return patterns


def _stage_lerobot_root(
    snapshot_dir: pathlib.Path,
    subset: str,
    target_root: pathlib.Path,
    num_episodes: int,
    keep_video_keys: list[str],
) -> None:
    """Symlink the staged subset/* into a clean LeRobot v2 layout at target_root.

    ``keep_video_keys`` is the list of ``observation.images.*`` features that
    were actually downloaded; any other video features declared in
    ``info.json`` are dropped from the staged copy. LeRobot's
    ``LeRobotDataset.__init__`` asserts that every video feature has a
    matching MP4 on disk, so we must not advertise features whose files we
    deliberately skipped.
    """
    if target_root.exists():
        # Wipe stale staging. We never wipe the snapshot — that's the cache.
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    src_root = snapshot_dir / subset if subset else snapshot_dir

    # Copy meta/ verbatim, then patch info.json + episodes.jsonl below.
    (target_root / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "tasks.jsonl", "stats.json", "modality.json"):
        src = src_root / "meta" / name
        if src.exists():
            shutil.copy(src, target_root / "meta" / name)

    # data/ and videos/ symlinks (skip videos we didn't download).
    keep_video_set = set(keep_video_keys)
    for kind in ("data", "videos"):
        src_kind = src_root / kind
        if not src_kind.exists():
            continue
        for sub in src_kind.rglob("*"):
            if not sub.is_file():
                continue
            rel = sub.relative_to(src_root)
            # rel for videos is "videos/chunk-XXX/<video_key>/<file>".
            if kind == "videos":
                parts = rel.parts
                if len(parts) >= 3 and parts[2] not in keep_video_set:
                    continue
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(sub.resolve(), dst)
            except FileExistsError:
                pass

    # Patch info.json: truncate episode count and drop unused video features.
    info_path = target_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    if keep_video_set:
        features = {
            k: v
            for k, v in features.items()
            if not (isinstance(v, dict) and v.get("dtype") == "video" and k not in keep_video_set)
        }
        info["features"] = features

    info["total_episodes"] = num_episodes
    info["splits"] = {"train": f"0:{num_episodes}"}

    # Recompute total_frames from episodes.jsonl (kept consistent — not
    # strictly read during training, but lerobot may inspect it).
    src_episodes_path = src_root / "meta" / "episodes.jsonl"
    kept_lines: list[str] = []
    total_frames = 0
    if src_episodes_path.exists():
        with src_episodes_path.open() as f:
            for i, raw in enumerate(f):
                if i >= num_episodes:
                    break
                kept_lines.append(raw)
                try:
                    total_frames += int(json.loads(raw).get("length", 0))
                except (json.JSONDecodeError, ValueError):
                    pass
    info["total_frames"] = total_frames
    num_video_features = sum(
        1
        for v in features.values()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )
    info["total_videos"] = num_episodes * num_video_features
    info_path.write_text(json.dumps(info, indent=2))

    # Truncate episodes.jsonl to the kept range.
    if kept_lines:
        (target_root / "meta" / "episodes.jsonl").write_text("".join(kept_lines))

    # Patch stats.json the same way: drop entries for video features we no
    # longer advertise. LeRobot's older code paths may try to read every key
    # listed in features against stats — keeping them in sync is safer.
    stats_path = target_root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        # Only video stats are tied to specific keys; numeric stats stay.
        stats = {
            k: v
            for k, v in stats.items()
            if not k.startswith("observation.images.") or k in keep_video_set
        }
        stats_path.write_text(json.dumps(stats))


def download_subset(
    *,
    hf_repo_id: str,
    subset: str,
    num_episodes: int,
    image_keys: str,
    cache_root: pathlib.Path,
    local_repo_id: str,
) -> pathlib.Path:
    """Download a small subset of a HuggingFace LeRobot v2 dataset.

    Returns the path to the staged LeRobot v2 root (``$HF_LEROBOT_HOME/<repo>/``).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download datasets. "
            "Install with `uv pip install huggingface_hub` or set "
            "HF_DATASET_REPO='' to skip download."
        ) from e

    video_keys = _video_keys_from_image_keys(image_keys)
    if not video_keys:
        logger.warning(
            "No `observation.images.*` keys found in IMAGE_KEYS — videos will "
            "not be downloaded. Make sure your IMAGE_KEYS spec is correct."
        )

    snapshot_dir = cache_root / "_hf_snapshots" / hf_repo_id.replace("/", "__")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    patterns = _allow_patterns(subset, num_episodes, video_keys)
    logger.info(
        "Downloading %d episodes of %s (subset=%r, %d video keys) -> %s",
        num_episodes, hf_repo_id, subset or "<root>", len(video_keys), snapshot_dir,
    )
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        allow_patterns=patterns,
    )

    lerobot_home = pathlib.Path(os.environ["HF_LEROBOT_HOME"])
    target_root = lerobot_home / local_repo_id
    _stage_lerobot_root(
        snapshot_dir, subset, target_root, num_episodes, keep_video_keys=video_keys
    )
    logger.info("Staged LeRobot v2 dataset at %s", target_root)
    return target_root
