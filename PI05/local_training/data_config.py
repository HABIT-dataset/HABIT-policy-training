"""DataConfigFactory for bimanual LeRobot v2 datasets.

Bridges LeRobot v2 dataset column names (e.g. ``robot0.observation.state.*``,
``observation.images.*``) to the OpenPI model input format (``state``,
``actions``, ``image``, ``image_mask``, ``prompt``) using ``RepackTransform``.

Column names are passed in as constructor args (driven by env vars in the
training entrypoint) so a wide range of bimanual / multi-camera datasets can
be configured without code changes.
"""

import dataclasses
import os
import pathlib
from typing import Sequence

import numpy as np

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
)


@dataclasses.dataclass(frozen=True)
class ConcatColumns(_transforms.DataTransformFn):
    """Concatenate multiple parquet columns into one array along the last axis.

    Useful for bimanual setups where state/action is split across per-arm
    columns (e.g. ``robot0.action.delta_action`` + ``robot1.action.delta_action``).
    The concatenated array is written to ``output_key``; the source columns are
    left intact so other transforms can still reference them.
    """

    output_key: str
    source_keys: tuple[str, ...]

    def __call__(self, data):
        # At inference, action source columns aren't present (only state is).
        # If output_key is already populated or any source key is missing,
        # skip concatenation so the same transform works in both modes.
        if self.output_key in data:
            return data
        if not all(k in data for k in self.source_keys):
            return data
        arrays = [np.asarray(data[k]) for k in self.source_keys]
        data[self.output_key] = np.concatenate(arrays, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class ImagesToModelFormat(_transforms.DataTransformFn):
    """Convert RepackTransform output (``images``) to model format (``image`` + ``image_mask``).

    OpenPI's built-in policies (Aloha/Libero) have their own input transforms
    that do this. We emulate the minimal necessary conversion here so
    ``ModelTransformFactory`` (ResizeImages, TokenizePrompt, etc.) can consume
    the data.

    - Renames ``images`` -> ``image`` (plural to singular)
    - Generates ``image_mask`` (True for each view present)
    - Converts torch.Tensor frames from torchcodec to numpy uint8 HWC
      (OpenPI's image_tools.resize_with_pad expects numpy arrays).
    """

    def __call__(self, data):
        if "images" in data:
            images_dict = data.pop("images")
            converted = {}
            for k, v in images_dict.items():
                arr = v
                if hasattr(arr, "numpy"):
                    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr.numpy()
                arr = np.asarray(arr)
                # CHW float [0,1] -> HWC uint8 [0,255]
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = arr.astype(np.uint8)
                converted[k] = arr
            data["image"] = converted
            data["image_mask"] = {k: np.True_ for k in converted}
        return data


def _build_repack_structure(
    image_keys: str,
    state_field: str,
    action_field: str,
) -> dict:
    """Build a RepackTransform structure dict from env-driven field names.

    Args:
        image_keys: Comma-separated pairs of ``"target_name:source_path"``,
            e.g. ``"base_0_rgb:observation.images.front_view,..."``. If a pair
            has no ``:``, the source path is used as both target name and path.
        state_field: Dot-separated path to a single state column. For
            multi-column setups (bimanual) the caller should first inject a
            ``ConcatColumns`` and pass the resulting concat key here.
        action_field: Same as ``state_field`` but for actions.
    """
    images = {}
    if image_keys:
        for pair in image_keys.split(","):
            pair = pair.strip()
            if ":" in pair:
                target, source = pair.split(":", 1)
            else:
                target = pair.rsplit(".", 1)[-1] if "." in pair else pair
                source = pair
            images[target.strip()] = source.strip()

    structure: dict = {}
    if images:
        structure["images"] = images
    if state_field:
        structure["state"] = state_field
    if action_field:
        structure["actions"] = action_field
    # Preserve the prompt key added by PromptFromLeRobotTask upstream — without
    # this, RepackTransform drops it and TokenizePrompt fails.
    structure["prompt"] = "prompt"
    return structure


@dataclasses.dataclass(frozen=True)
class BimanualDataConfig(DataConfigFactory):
    """DataConfigFactory for bimanual LeRobot v2 datasets.

    Builds the repack/data/model transforms needed to bridge LeRobot v2 column
    names to OpenPI's expected input dict. Column names are configured via
    constructor args (set from env vars in ``train_local``).
    """

    image_keys: str = "base_0_rgb:observation.images.front_view"
    state_field: str = "observation.state"
    action_field: str = "action"
    action_sequence_keys: Sequence[str] = ("action",)
    default_prompt: str | None = None
    use_delta_actions: bool = False
    delta_action_mask_spec: str = ""
    prompt_from_task: bool = True

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        state_cols = [s.strip() for s in self.state_field.split(",") if s.strip()]
        action_cols = [s.strip() for s in self.action_field.split(",") if s.strip()]

        pre_transforms: list = []
        state_source = state_cols[0] if len(state_cols) == 1 else "_state_concat"
        action_source = action_cols[0] if len(action_cols) == 1 else "_actions_concat"

        if len(state_cols) > 1:
            pre_transforms.append(
                ConcatColumns(output_key=state_source, source_keys=tuple(state_cols))
            )
        if len(action_cols) > 1:
            pre_transforms.append(
                ConcatColumns(output_key=action_source, source_keys=tuple(action_cols))
            )

        repack_structure = _build_repack_structure(
            self.image_keys, state_source, action_source
        )

        repack_transform = _transforms.Group(
            inputs=[*pre_transforms, _transforms.RepackTransform(repack_structure)]
        )

        # Convert "images" (plural) -> "image" + "image_mask" expected by openpi.
        data_transforms = _transforms.Group(inputs=[ImagesToModelFormat()])

        if self.use_delta_actions and self.delta_action_mask_spec:
            mask = _parse_delta_mask(self.delta_action_mask_spec)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(mask)],
                outputs=[_transforms.AbsoluteActions(mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        # PI0.5's default is quantile (q01/q99) normalization, which collapses
        # a degenerate column (e.g. a gripper that never moves on a small
        # subset) to a constant -1.0 and destabilizes training. Setting
        # USE_MIN_MAX_NORM=true forces mean/std (z-score) normalization, which
        # has a built-in epsilon that tolerates degenerate columns.
        use_quantile_override = base.use_quantile_norm
        if os.getenv("USE_MIN_MAX_NORM", "").lower() in ("1", "true", "yes"):
            use_quantile_override = False

        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
            use_quantile_norm=use_quantile_override,
        )


def _parse_delta_mask(spec: str) -> list[bool]:
    """Parse a delta-mask spec string.

    Format: comma-separated integers with optional minus prefix.
    Positive value N -> N True entries (apply delta), negative -> N False
    entries (keep absolute). Example: ``"6,-1"`` -> ``[True]*6 + [False]``.
    """
    mask: list[bool] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val > 0:
            mask.extend([True] * val)
        else:
            mask.extend([False] * abs(val))
    return mask


def register_bimanual_config(env, config_name: str = "bimanual_local") -> str:
    """Register a TrainConfig for the bimanual data layout into OpenPI's registry.

    Builds the config from env vars and adds it to OpenPI's
    ``_CONFIGS_DICT`` so ``get_config(config_name)`` can find it.
    """
    import openpi.models.pi0_config as pi0_config
    import openpi.models.pi0_fast as pi0_fast
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders
    from openpi.training.config import (
        DataConfig,
        TrainConfig,
        _CONFIGS,
        _CONFIGS_DICT,
    )

    model_variant = (env.model_variant or "pi05").lower()
    action_dim = env.action_dim or 32
    action_horizon = env.action_horizon or 50

    if model_variant == "pi0":
        model = pi0_config.Pi0Config(action_dim=action_dim, action_horizon=action_horizon)
    elif model_variant == "pi05":
        model = pi0_config.Pi0Config(pi05=True, action_dim=action_dim, action_horizon=action_horizon)
    elif model_variant == "pi0_fast":
        model = pi0_fast.Pi0FASTConfig(
            action_dim=action_dim, action_horizon=action_horizon, max_token_len=180
        )
    else:
        raise ValueError(f"Unknown MODEL_VARIANT: {model_variant}")

    # Action sequence keys: use ALL action columns so LeRobot expands each
    # over action_horizon before they're concatenated.
    action_cols = tuple(
        s.strip() for s in (env.action_fields or "action").split(",") if s.strip()
    )

    data_factory = BimanualDataConfig(
        repo_id=env.dataset_repo_id,
        image_keys=env.image_keys
        or "base_0_rgb:observation.images.front_view",
        state_field=env.state_fields or "observation.state",
        action_field=env.action_fields or "action",
        action_sequence_keys=action_cols,
        default_prompt=env.default_prompt or None,
        prompt_from_task=True,
        base_config=DataConfig(prompt_from_task=True),
    )

    if env.weight_loader_path:
        weight_loader = weight_loaders.CheckpointWeightLoader(env.weight_loader_path)
    else:
        weight_loader = weight_loaders.NoOpWeightLoader()

    train_config = TrainConfig(
        name=config_name,
        model=model,
        data=data_factory,
        weight_loader=weight_loader,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=env.warmup_steps or 1_000,
            peak_lr=env.learning_rate or 2.5e-5,
            decay_steps=env.decay_steps or 30_000,
            decay_lr=(env.learning_rate or 2.5e-5) / 10,
        ),
        num_train_steps=env.num_train_steps or 30_000,
        batch_size=env.batch_size or 32,
    )

    if config_name not in _CONFIGS_DICT:
        _CONFIGS.append(train_config)
        _CONFIGS_DICT[config_name] = train_config

    return config_name
