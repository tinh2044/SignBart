#!/bin/bash

CONDA_ENV_NAME="base"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate

EXPERIMENT_NAME="signBart-KArSL01-190"
CONFIG_PATH="./configs/KArSL01-190.yaml"
PRETRAINED_PATH="./pretrained_models/KArSL01-190.pth"
SEED=379
TASK="eval"
DATA_PATH="./data/KArSL01-190"
EPOCHS=200
LEARNING_RATE=0.01
RESUME_CHECKPOINTS=""
SCHEDULER_FACTOR=0.1
SCHEDULER_PATIENCE=5

python -m main \
    --experiment_name "$EXPERIMENT_NAME" \
    --config_path "$CONFIG_PATH" \
    --pretrained_path "$PRETRAINED_PATH" \
    --seed $SEED \
    --task "$TASK" \
    --data_path "$DATA_PATH" \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --resume_checkpoints "$RESUME_CHECKPOINTS" \
    --scheduler_factor $SCHEDULER_FACTOR \
    --scheduler_patience $SCHEDULER_PATIENCE

conda deactivate
