#!/bin/bash
bagname=_2023_06_26_11_08_53
droid_slam_in_dir=/home/tiejean/Documents/mnt/oslam/droid_slam_in
droid_slam_out_dir=/home/tiejean/Documents/mnt/oslam/droid_slam_out

python evaluation_scripts/test_obvi.py \
    --imagedir=$droid_slam_in_dir/$bagname/image_0 \
    --right_imagedir=$droid_slam_in_dir/$bagname/image_1 \
    --calib=calib/obvi_slam.txt \
    --node_ids_and_timestamps=$droid_slam_in_dir/$bagname/node_ids_and_timestamps.txt \
    --reconstruction_path=$droid_slam_out_dir/$bagname/ \
    --stride=1