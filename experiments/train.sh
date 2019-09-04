#!/bin/bash

# Parameter
EXP="2048@0.05x3"
VERSION="log_dropout_u0.1_l0.1"
PRETRAINED="/automount_home_students/flanger/workspace/msc/lidar-bonnetal/train/models/darknet53/"

MODEL="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/logs/$VERSION"
DATA="/media/flanger/SAMS1TB_0/msc/experiments/$EXP"
source ~/workspace/msc/venv/bin/activate

# Training
cd ~/workspace/msc/lidar-bonnetal/train/tasks/semantic/
./train.py \
    -d "$DATA/dataset/" \
    -p "$PRETRAINED" \
    -ac "$DATA/arch_cfg_dropout.yaml" \
    -dc "$DATA/data_cfg.yaml" \
    -l "$MODEL"

TITLE="$EXP/$VERSION done"
MSG=""
curl -u $PUSHBULLET_TOKEN: https://api.pushbullet.com/v2/pushes \
    -d type=note -d title="$TITLE" -d body="$MSG" >/dev/null 2>&1
