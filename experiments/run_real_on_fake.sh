#!/bin/bash

# Parameter
EXP="2048\@0.04/"
VERSION="log2048"
TYPE="real_inf_on_fake"
PRETRAINED="~/workspace/msc/lidar-bonnetal/train/models/darknet53/"

PREDICTION="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/predictions/$TYPE"
MODEL="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/logs/$VERSION"
DATA="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/dataset/"
cd ~/workspace/msc
# Training
# use pretrained model on real data

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
