import sys
sys.path.append('droid_slam')

import json
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse
import pandas as pd

from functools import partial
from torch.multiprocessing import Process
import evo
from evo.core.trajectory import PoseTrajectory3D
from droid import Droid

import torch.nn.functional as F


kTrajFilename = "Trajectory.csv"
kNodeIdsAndTimestampsFilename = "timestamps/node_ids_and_timestamps.txt"
kLeftImageDirName = "image_0"
kRightImageDirName = "image_1"
kOrbOutDirName = "orb_out"
kDroidInDirName = "droid_slam_in"
kDroidOutDirName = "droid_slam_out"

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics

# TODO using length of calib to understand what to do
def rectified_stereo_image_stream(datapath, right_data_path, calib, image_size=[280, 512], stride=1):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    base_images_left = sorted(os.listdir(datapath))[::stride]
    images_left = [os.path.join(datapath, leftFileName) for leftFileName in base_images_left]
    images_right = [os.path.join(right_data_path, leftFileName) for leftFileName in base_images_left]
    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if (not os.path.isfile(imgL)) or (not os.path.isfile(imgR)):
            continue
        leftImg = cv2.imread(imgL); rightImg = cv2.imread(imgR)
        h0, w0, _ = leftImg.shape
        # h1 = int(h0 * np.sqrt((image_size[0] * image_size[1]) / (h0 * w0)))
        # w1 = int(h0 * np.sqrt((image_size[0] * image_size[1]) / (h0 * w0)))

        # leftImg = cv2.resize(leftImg, (w1, h1))
        # rightImg = cv2.resize(rightImg, (w1, h1))
        images =[leftImg]; images += [rightImg]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)

        intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()
        intrinsics[0] *= image_size[1] / w0
        intrinsics[1] *= image_size[0] / h0
        intrinsics[2] *= image_size[1] / w0
        intrinsics[3] *= image_size[0] / h0
        # intrinsics[0] *= (w1 / w0)
        # intrinsics[1] *= (h1 / h0)
        # intrinsics[2] *= (w1 / w0)
        # intrinsics[3] *= (h1 / h0)

        yield stride * t, images, intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("{}/images.npy".format(reconstruction_path), images)
    np.save("{}/disps.npy".format(reconstruction_path), disps)
    np.save("{}/poses.npy".format(reconstruction_path), poses)
    np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    
    return tstamps

def parse_timestamps(filepath):
    df = pd.read_csv(filepath)
    timestamps = []
    for _, row in df.iterrows():
        timestamps.append((row[" seconds"], row[" nanoseconds"]))
    return timestamps

def parseSeqIdAndBagnamesFromSequenceFile(path):
    fp = open(path, "r")
    if fp.closed:
        raise FileNotFoundError("Fail to open file " + path)
    data = json.load(fp)["sequence_info"]
    seq_id = data["seq_id"]
    bagnames = []
    for baginfo in data["sequence"]:
        bagnames.append(baginfo["bag_base_name"])
    fp.close()
    return seq_id, bagnames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqfile", required=True, type=str, help="sequence file")
    parser.add_argument("--datadir", required=True, type=str, help="root")
    # parser.add_argument("--node_ids_and_timestamps", required=True, type=str, help="path to the file that indicate nodes and their associated timestamp")
    # parser.add_argument("--imagedir", type=str, help="path to image directory")
    # parser.add_argument("--right_imagedir", default=None, help="optional (supply for stereo), path to directory for right images")
    parser.add_argument("--calib", required=True, type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    # parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    if not os.path.exists(args.datadir):
        print("Data root directory " + args.datadir + " doesn't exists!")
        exit(1)
    seq_ids, bagnames = parseSeqIdAndBagnamesFromSequenceFile(args.seqfile)
    for seq_id, bagname in zip(seq_ids, bagnames):
        args.node_ids_and_timestamps = os.path.join(args.datadir, kOrbOutDirName, bagname, kNodeIdsAndTimestampsFilename)
        if not os.path.exists(args.node_ids_and_timestamps):
            print("Timestamp file " + args.node_ids_and_timestamps + "doesn't exists!")
            exit(1)
        droid_input_dir = os.path.join(args.datadir, kDroidInDirName, bagname)
        if not os.path.exists(droid_input_dir):
            print("Droid slam input directory " + droid_input_dir + "doesn't exists!")
            exit(1)
        args.imagedir = os.path.join(droid_input_dir, kLeftImageDirName)
        if not os.path.exists(args.imagedir):
            print("Left Image directory " + args.imagedir + "doesn't exists!")
            exit(1)
        args.right_imagedir = os.path.join(droid_input_dir, kRightImageDirName)
        if not os.path.exists(args.right_imagedir):
            print("Right Image directory " + args.right_imagedir + "doesn't exists!")
            exit(1)
        droid_output_dir = os.path.join(args.datadir, kDroidOutDirName)
        if not os.path.exists(droid_output_dir):
            os.mkdir(args.reconstruction_path)
        args.reconstruction_path = os.path.join(droid_output_dir, str(seq_id)+"_"+bagname)
        if not os.path.exists(args.reconstruction_path):
            os.mkdir(args.reconstruction_path)
        
        timestamps = parse_timestamps(args.node_ids_and_timestamps)

        if (args.right_imagedir is None):
            args.stereo = False
            imgStreamFunc = partial(image_stream, imagedir=args.imagedir, calib=args.calib, stride=args.stride)
        else:
            args.stereo = True
            imgStreamFunc = partial(rectified_stereo_image_stream, datapath=args.imagedir, right_data_path=args.right_imagedir, calib=args.calib, stride=args.stride)
        torch.multiprocessing.set_start_method('spawn')

        droid = None

        # need high resolution depths
        if args.reconstruction_path is not None:
            args.upsample = True

        tstamps = []
        for (t, image, intrinsics) in tqdm(imgStreamFunc()):
            if t < args.t0:
                continue
            tstamps.append(t)

            if not args.disable_vis:
                show_image(image[0])

            if droid is None:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)

            droid.track(t, image, intrinsics=intrinsics)

        if args.reconstruction_path is not None:
            print("Initiating save reconstruction")
            save_reconstruction(droid, args.reconstruction_path)
            print("Done with saving reconstruction")

        print("Initiating terminate function")
        traj_est = droid.terminate(rectified_stereo_image_stream(datapath=args.imagedir, right_data_path=args.right_imagedir, calib=args.calib, stride=1))
        print("Done with terminate function")
        
        savepath = os.path.join(args.reconstruction_path, kTrajFilename)
        fp = open(savepath, "w")
        if fp.closed:
            print("Failed to open file " + savepath)
            exit(1)
        fp.write("seconds, nanoseconds, lost, transl_x, transl_y, transl_z, quat_x, quat_y, quat_z, quat_w\n")
        for i, pose in enumerate(traj_est):
            fp.write(str(timestamps[i][0]) + ", " + str(timestamps[i][1]))
            fp.write(", 0") # hardcoding lost to be 0
            fp.write(", " + str(pose[0]) + ", " + str(pose[1]) + ", " + str(pose[2]) \
                    + ", " + str(pose[6]) + ", " + str(pose[4]) + ", " + str(pose[5]) + ", " + str(pose[6]) )
            fp.write("\n")
        fp.close()
        print("Done with saving trajectory")

        print(traj_est)
        print(traj_est.shape)

        # traj_est = PoseTrajectory3D(
        #     positions_xyz=traj_est[:,:3],
        #     orientations_quat_wxyz=traj_est[:,3:],
        #     timestamps=np.array(tstamps))