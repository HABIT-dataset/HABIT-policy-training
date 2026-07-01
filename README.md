# HABIT Policy Training

Training code for the two vision-language-action (VLA) policies fine-tuned on the
[**HABIT**](https://huggingface.co/datasets/configinc/HABIT) dataset — a
large-scale bimanual, human-present robot manipulation dataset in LeRobot v2
format.

This repository packages two independent, self-contained training setups, each
vendoring its upstream policy codebase and adding a thin HABIT-specific
configuration layer. Both have been adapted to run on a **single local
machine** (no cluster/orchestration dependencies).

| Model | Directory | Upstream | HABIT entrypoint |
|---|---|---|---|
| **π0.5** (OpenPI) | [`PI05/`](PI05/) | [openpi](https://github.com/Physical-Intelligence/openpi) | `python -m local_training.train_local samples/envs/.env.habit` |
| **GR00T N1.6** | [`Gr00t-N1.6/`](Gr00t-N1.6/) | [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `bash examples/HABIT/finetune_habit_1gpu.sh` |

Each directory has its own README with full setup and usage instructions:

- **[`PI05/README.md`](PI05/README.md)** — OpenPI (π0 / π0.5 / π0-FAST) local training.
- **[`Gr00t-N1.6/README.md`](Gr00t-N1.6/README.md)** — GR00T N1.6 fine-tuning (see the *Fine-tune HABIT* section).

---

## The HABIT data layout both models consume

HABIT is a **bimanual Franka Research 3** dataset. Both training setups read the
same LeRobot v2 fields:

- **State (14D):** per-arm end-effector Cartesian position (xyz 3D + rotation 3D)
  + gripper (1D), for the left and right arm.
- **Action (14D):** per-arm end-effector **delta action** (7D [xyz 3D, rotation 3D, gripper 1D]), left and right.
- **Cameras (robot-side):** `front_view`, `left_wrist_view`, `right_wrist_view`.

> The full HABIT dataset additionally ships two human-side camera streams
> (`human_front_view`, `exo_view`); the policy-training configs here use only
> the three robot-side views, matching the experiments in the paper.

The two policies expose these fields differently:

- **π0.5** maps LeRobot columns to OpenPI inputs via env vars in
  `PI05/samples/envs/.env.habit` (`STATE_FIELDS`, `ACTION_FIELDS`, `IMAGE_KEYS`).
  It can auto-download a subset of `configinc/HABIT` from the Hub.
- **GR00T** reads a local LeRobot v2 directory via `--dataset_path` and maps
  fields through `Gr00t-N1.6/examples/HABIT/habit_config.py`. Download the
  dataset yourself first (e.g. `git clone` / `huggingface-cli download
  configinc/HABIT`) and point `--dataset_path` at the resulting directory.

---

## Quick start

### π0.5 (OpenPI) — auto-downloads a 10-episode HABIT subset

```bash
cd PI05
cd openpi && uv sync && cd ..          # install OpenPI deps once
python -m local_training.train_local samples/envs/.env.habit
```

This stages 10 episodes of the HABIT `sample` subset under
`~/.cache/openpi-local/`, pulls the public π0.5 base weights, and runs a
200-step fine-tune. See [`PI05/README.md`](PI05/README.md) for scaling to the
full dataset and adapting to other LeRobot v2 datasets.

### GR00T N1.6 — point it at a local HABIT checkout

```bash
cd Gr00t-N1.6
# install deps per Gr00t-N1.6/README.md (uv / pip), then:
# edit --dataset_path in the script to your local HABIT directory
bash examples/HABIT/finetune_habit_1gpu.sh      # single GPU
bash examples/HABIT/finetune_habit_1node.sh     # 8 GPUs, single node (torchrun)
```

Checkpoints are written to `Gr00t-N1.6/outputs/habit_finetune/` by default.

---

## Repository notes

- **Vendored upstreams.** `PI05/openpi/` and `Gr00t-N1.6/` contain their
  respective upstream sources under their original licenses (Apache 2.0). The
  HABIT-specific additions are limited to `PI05/local_training/`,
  `PI05/samples/`, and `Gr00t-N1.6/examples/HABIT/`.
- **Local-only.** Both entrypoints run on one machine (single- or multi-GPU via
  `torchrun`); there are no cloud/cluster orchestration dependencies in the
  training path.
- **License.** Each vendored codebase retains its upstream license; the
  HABIT integration code is released under the same terms.
