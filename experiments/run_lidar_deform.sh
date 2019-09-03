#!/bin/bash

# Parameter
EXP="2048@0.05x3"

SRC="/media/flanger/SAMS1TB_0/kitti-odometry/dataset/"
TARG="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/dataset"
CONF="/media/flanger/SAMS1TB_0/msc/experiments/$EXP/lidar_transfer.yaml"

# Generate new datasets from all sequences
cd ~/workspace/msc/lidar_transfer/
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 00 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 01 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 02 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 03 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 04 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 05 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 06 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 07 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 08 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 09 -p "$TARG" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 10 -p "$TARG" || \
    MSG="FAILED!!!" && MSG="Success!!"

MSG="Create dataset $MSG"
curl -u $PUSHBULLET_TOKEN: https://api.pushbullet.com/v2/pushes \
    -d type=note -d title="$TITLE" -d body="$MSG" >/dev/null 2>&1
