#!/bin/bash
bagname=_2023_06_26_11_08_53

python evaluation_scripts/test_obvi.py \
    --imagedir=/home/tiejean/Documents/mnt/oslam/droid_slam_in/$bagname/image_0 \
    --right_imagedir=/home/tiejean/Documents/mnt/oslam/droid_slam_in/$bagname/image_1 \
    --calib=calib/obvi_slam.txt \
    --node_ids_and_timestamps=/home/tiejean/Documents/mnt/oslam/droid_slam_in/$bagname/node_ids_and_timestamps.txt \
    --reconstruction_path=/home/tiejean/Documents/mnt/oslam/droid_slam_out/$bagname/ \
    --stride=3