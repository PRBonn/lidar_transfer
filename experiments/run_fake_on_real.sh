#!/bin/bash

# Parameter
EXP="2048@0.04/"
VERSION="log2048"
TYPE="fake_on_real"
PRETRAINED="~/workspace/msc/lidar-bonnetal/train/models/darknet53/"

PREDICTION="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/predictions/$TYPE"
MODEL="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/logs/$VERSION"
DATA="/media/flanger/SAMS1TB_0/kitti-odometry/dataset/"
cd ~/workspace/msc
# Training
# ./lidar-bonnetal/train/tasks/semantic/train.py \
#     -d "~/workspace/msc/lidar_transfer/output/" \
#     -p "$PRETRAINED" \
#     -ac "lidar-bonnetal/train/tasks/semantic/config/arch/darknet53.yaml" \
#     -dc "lidar-bonnetal/train/tasks/semantic/config/labels/semantic-kitti.yaml" \
#     -l "$MODEL"

# Infer
./lidar-bonnetal/train/tasks/semantic/infer.py \
    -d "$DATA" \
    -m "$MODEL" \
    -l "$PREDICTION"

# Evaluate
./semantic-kitti-api/evaluate_semantics.py \
    -d "$DATA" \
    -p "$PREDICTION" \
    -s valid
