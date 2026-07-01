from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


habit_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_wrist_view", "left_wrist_view", "front_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cartesian_position_xyz_left",       # 3D
            "cartesian_position_rot_left",       # 3D
            "gripper_position_left",             # 1D
            "cartesian_position_xyz_right",      # 3D
            "cartesian_position_rot_right",      # 3D
            "gripper_position_right",            # 1D  -> total 14D
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "delta_action_eef_left",             # 6D (xyz + rpy delta)
            "delta_action_gripper_left",         # 1D
            "delta_action_eef_right",            # 6D (xyz + rpy delta)
            "delta_action_gripper_right",        # 1D  -> total 14D
        ],
        # NOTE on rep/type: HABIT actions are already stored as per-step deltas
        # (see the `delta_action_*` keys). We therefore use rep=ABSOLUTE so the
        # processor does NOT apply its own absolute->relative conversion — the
        # RELATIVE code path (state_action_processor / stats.py) only fires for
        # keys with rep=RELATIVE, so leaving these ABSOLUTE feeds the stored
        # deltas straight through to min/max normalization. Using RELATIVE here
        # would double-difference the actions. Because that path is skipped,
        # `type` is not consulted for these keys; it is left as EEF for the arm
        # entries for readability and is a no-op for the gripper entries.
        # `action_configs` is positional 1:1 with `modality_keys` above.
        action_configs=[
            # delta_action_eef_left (arm, 6D end-effector delta)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # delta_action_gripper_left (1D gripper)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # delta_action_eef_right (arm, 6D end-effector delta)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # delta_action_gripper_right (1D gripper)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(habit_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
