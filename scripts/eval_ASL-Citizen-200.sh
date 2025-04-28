#!/bin/bash

CONDA_ENV_NAME="base"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate

EXPERIMENT_NAME="signBart-ASL-Citizen-200"
CONFIG_PATH="./configs/ASL-Citizen-200.yaml"
PRETRAINED_PATH=""./pretrained_models/ASL-Citizen-200.pth""
SEED=379
TASK="eval"
DATA_PATH="./data/ASL-Citizen-200"
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
