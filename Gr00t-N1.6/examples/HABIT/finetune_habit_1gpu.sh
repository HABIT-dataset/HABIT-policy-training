set -x -euo pipefail

CUDA_VISIBLE_DEVICES=0 python \
    -m gr00t.experiment.launch_finetune \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /path/to/your/habit_dataset \
    --modality_config_path examples/HABIT/habit_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus 1 \
    --output_dir ./outputs/habit_finetune \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 5000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 128 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use_wandb
