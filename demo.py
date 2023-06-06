import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse

from functools import partial
from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


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


def stereo_image_stream(datapath, right_data_path, orb_calib_file, image_size=[320, 512], stride=1):
    """ image generator """

    # GUIDANCE: For a different dataset, replace the following blocks with the appropriate camera parameters
    # TODO read from calib file

    K_l = np.array([527.873518, 0.000000, 482.823413, 0.000000, 527.276819, 298.033945, 0.000000, 0.000000, 1.000000]).reshape(3, 3)
    d_l = np.array([0, 0, 0, 0, 0])
    R_l = np.array([0.999940, -0.003244, -0.010471, 0.003318, 0.999970, 0.007064, 0.010448, -0.007098, 0.999920]).reshape(3, 3)

    P_l = np.array(
        [528.955512, 0.000000, 479.748173, 0.000000, 0.000000, 528.955512, 298.607571, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]).reshape(3,4)

    K_r = np.array([530.158021, 0.000000, 475.540633, 0.000000, 529.682234, 299.995465, 0.000000, 0.000000, 1.000000]).reshape(3, 3)
    d_r = np.array([0, 0, 0, 0, 0]).reshape(5)
    R_r = np.array([0.999661, -0.024534, 0.008699, 0.024595, 0.999673, -0.006974, -0.008525, 0.007186, 0.999938]).reshape(3, 3)

    P_r = np.array(
        [528.955512, 0.000000, 479.748173, -69.690815, 0.000000, 528.955512, 298.607571, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]).reshape(3, 4)

    intrinsics_vec = [528.955512, 528.955512, 479.748173, 298.607571] # (fx fy cx cy)
    # End of dataset specific code -----------------------------------------------------------------------------

    # read all png images in folder
    base_images_left = sorted(os.listdir(datapath))[::stride]
    # images_right = sorted(os.listdir(right_data_path))[::stride]

    images_left = [os.path.join(datapath, leftFileName) for leftFileName in base_images_left]
    images_right = [os.path.join(right_data_path, leftFileName) for leftFileName in base_images_left]

    map_l = None

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):

        if not os.path.isfile(imgR):
            continue

        leftImg = cv2.imread(imgL)

        h0, w0, _ = leftImg.shape

        if (map_l is None):
            map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3, :3], (w0, h0), cv2.CV_32F)
            map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3, :3], (w0, h0), cv2.CV_32F)

        images = [cv2.remap(leftImg, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)

        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / w0
        intrinsics[1] *= image_size[0] / h0
        intrinsics[2] *= image_size[1] / w0
        intrinsics[3] *= image_size[0] / h0

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

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--right_imagedir", default=None, help="optional (supply for stereo), path to directory for right images")
    parser.add_argument("--calib", type=str, help="path to calibration file")
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
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()


    if (args.right_imagedir is None):
        args.stereo = False
        imgStreamFunc = partial(image_stream, imagedir=args.imagedir, calib=args.calib, stride=args.stride)
    else:
        args.stereo = True
        imgStreamFunc = partial(stereo_image_stream, datapath=args.imagedir, right_data_path=args.right_imagedir, orb_calib_file=args.calib, stride=args.stride)
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(imgStreamFunc()):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    print("Initiating terminate function")
    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    print("Done with terminate function")
