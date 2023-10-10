# ObVi-SLAM Evaluation

This branch is for ObVi-SLAM Evaluation of DROID-SLAM.

## Set up DROID-SLAM
Here's a setup guide for DROID-SLAM Evaluation. If you encountered any questions during package installation, refer to the original [Getting Started](#getting-started) Page of DROID-SLAM or their [GitHub Issues](https://github.com/princeton-vl/DROID-SLAM/issues).

1. Clone the repo using the `--recursive` flag
```Bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
git checkout ObViSLAMEvaluation
```

2. Creating a new anaconda environment using the provided .yaml file. Using the following command, you'll get a conda environment named `droidenv_obvislam_eval`.
```Bash
conda env create -f environment_obvislam_eval.yaml
conda activate droidenv_obvislam_eval
pip install evo --upgrade --no-binary evo
pip install gdown
```

3. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
```

4. Download the model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing)

## Generate Input Data
The evaluation script accepts KITTI-formatted inputs. You can convert your bagfiles to KITTI format yourself. Here we provide a few alternatives if you don't have a script available. One thing to note is that the evaluation script expects a "node_ids_and_timestamps.txt" file to provide the timestamps for evaluation. You can refer to README in [ObVi-SLAM](https://github.com/ut-amrl/ObVi-SLAM.git) Repository to see how to obtain this file or specify the timestamps you want to evaluatio

If you already have the the [ObVi-SLAM](https://github.com/ut-amrl/ObVi-SLAM.git) Repository installed, you can run the following command inside the podman container:
```Bash
bash convenience_scripts/podman/droid_slam_data_generator.sh
```
The script will process all `bagnames` you specified under the `$root_data_dir`, and dump the output to `$droid_slam_data_output_directory` in KITTI format.

You can also clone the [SLAMUtilsScripts](https://github.com/ut-amrl/SLAMUtilsScripts.git) repository and then run
```Bash
python data_processing/parse_bag_to_kitti.py \
  --bagfile <path_to_bagfile> \
  --outputdir <path_to_output_directory> \
  --outputname <output_name; use bagname by default> \
  --left_img_topic <left_ros_image_topic> \
  --right_img_topic <right_ros_image_topic>
```

## Run Evaluation
We provide an Example script to help you run the evaluation:
```Bash
bash evaluation_scripts/test_obvi.sh
```

Specifically, you can call `evaluation_scripts/test_obvi.py` to run the evaluation on a single bag in the following way:
```Bash
python evaluation_scripts/test_obvi.py \
  --imagedir <left_image_directory; a folder named "image_0" in KITTI> \
  --right_imagedir <left_image_directory; a folder named "image_1" in KITTI> \
  --calib=calib/obvi_slam.txt \
  --node_ids_and_timestamps <path_to_node_ids_and_timestamps_file> \
  --reconstruction_path <directory_to_save_the_maps> \
  --stride=1
```


# DROID-SLAM


<!-- <center><img src="misc/DROID.png" width="640" style="center"></center> -->


[![IMAGE ALT TEXT HERE](misc/screenshot.png)](https://www.youtube.com/watch?v=GG78CSlSHSA)



[DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869)  
Zachary Teed and Jia Deng

```
@article{teed2021droid,
  title={{DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras}},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in neural information processing systems},
  year={2021}
}
```

**Initial Code Release:** This repo currently provides a single GPU implementation of our monocular, stereo, and RGB-D SLAM systems. It currently contains demos, training, and evaluation scripts. 


## Requirements

To run the code you will need ...
* **Inference:** Running the demos will require a GPU with at least 11G of memory. 

* **Training:** Training requires a GPU with at least 24G of memory. We train on 4 x RTX-3090 GPUs.

## Getting Started
1. Clone the repo using the `--recursive` flag
```Bash
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
```

2. Creating a new anaconda environment using the provided .yaml file. Use `environment_novis.yaml` to if you do not want to use the visualization
```Bash
conda env create -f environment.yaml
pip install evo --upgrade --no-binary evo
pip install gdown
```

3. Compile the extensions (takes about 10 minutes)
```Bash
python setup.py install
```


## Demos

1. Download the model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing)

2. Download some sample videos using the provided script.
```Bash
./tools/download_sample_data.sh
```

Run the demo on any of the samples (all demos can be run on a GPU with 11G of memory). While running, press the "s" key to increase the filtering threshold (= more points) and "a" to decrease the filtering threshold (= fewer points). To save the reconstruction with full resolution depth maps use the `--reconstruction_path` flag.


```Python
python demo.py --imagedir=data/abandonedfactory --calib=calib/tartan.txt --stride=2
```

```Python
python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt
```

```Python
python demo.py --imagedir=data/Barn --calib=calib/barn.txt --stride=1 --backend_nms=4
```

```Python
python demo.py --imagedir=data/mav0/cam0/data --calib=calib/euroc.txt --t0=150
```

```Python
python demo.py --imagedir=data/rgbd_dataset_freiburg3_cabinet/rgb --calib=calib/tum3.txt
```


**Running on your own data:** All you need is a calibration file. Calibration files are in the form 
```
fx fy cx cy [k1 k2 p1 p2 [ k3 [ k4 k5 k6 ]]]
```
with parameters in brackets optional.

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, and TUM. EuRoC and TUM can be run on a 1080Ti. The TartanAir and ETH will require 24G of memory.

### TartanAir (Mono + Stereo)
Download the [TartanAir](https://theairlab.org/tartanair-dataset/) dataset using the script `thirdparty/tartanair_tools/download_training.py` and put them in `datasets/TartanAir`
```Bash
./tools/validate_tartanair.sh --plot_curve            # monocular eval
./tools/validate_tartanair.sh --plot_curve  --stereo  # stereo eval
```

### EuRoC (Mono + Stereo)
Download the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) sequences (ASL format) and put them in `datasets/EuRoC`
```Bash
./tools/evaluate_euroc.sh                             # monocular eval
./tools/evaluate_euroc.sh --stereo                    # stereo eval
```

### TUM-RGBD (Mono)
Download the fr1 sequences from [TUM-RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) and put them in `datasets/TUM-RGBD`
```Bash
./tools/evaluate_tum.sh                               # monocular eval
```

### ETH3D (RGB-D)
Download the [ETH3D](https://www.eth3d.net/slam_datasets) dataset
```Bash
./tools/evaluate_eth3d.sh                             # RGB-D eval
```

## Training

First download the TartanAir dataset. The download script can be found in `thirdparty/tartanair_tools/download_training.py`. You will only need the `rgb` and `depth` data.

```
python download_training.py --rgb --depth
```

You can then run the training script. We use 4x3090 RTX GPUs for training which takes approximatly 1 week. If you use a different number of GPUs, adjust the learning rate accordingly.

**Note:** On the first training run, covisibility is computed between all pairs of frames. This can take several hours, but the results are cached so that future training runs will start immediately. 


```
python train.py --datapath=<path to tartanair> --gpus=4 --lr=0.00025
```


## Acknowledgements
Data from [TartanAir](https://theairlab.org/tartanair-dataset/) was used to train our model. We additionally use evaluation tools from [evo](https://github.com/MichaelGrupp/evo) and [tartanair_tools](https://github.com/castacks/tartanair_tools).
