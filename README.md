# Progressively Optimized Local Radiance Fields for Robust View Synthesis

We present an algorithm for reconstructing the radiance field of a large-scale scene from a single casually captured video. The task poses two core challenges. First, most existing radiance field reconstruction approaches rely on accurate pre-estimated camera poses from Structure-from-Motion algorithms, which frequently fail on in-the-wild videos. Second, using a single, global radiance field with finite representational capacity does not scale to longer trajectories in an unbounded scene. For handling unknown poses, we jointly estimate the camera poses with radiance field in a progressive manner. We show that progressive optimization significantly improves the robustness of the reconstruction. For handling large unbounded scenes, we dynamically allocate new local radiance fields trained with frames within a temporal window. This further improves robustness (e.g., performs well even under moderate pose drifts) and allows us to scale to large scenes. Our extensive evaluation on the Tanks and Temples dataset and our collected outdoor dataset, Static Hikes, show that our approach compares favorably with the state-of-the-art.

### [Project page](https://localrf.github.io/) | [Paper](https://localrf.github.io/localrf.pdf) | [Data](https://drive.google.com/drive/folders/1kGY-VijIbXNsNb7ghEywi1fvkH4BaIEz?usp=share_link)
[Andreas Meuleman](https://ameuleman.github.io), 
[Yu-Lun Liu](https://www.cmlab.csie.ntu.edu.tw/~yulunliu), 
[Chen Gao](http://chengao.vision), 
[Jia-Bin Huang](https://jbhuang0604.github.io), 
[Changil Kim](https://changilkim.com), 
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim), 
[Johannes Kopf](http://johanneskopf.de)

## Setup
Tested with Pytorch 2.0 and CUDA 11.8 and ROCm 5.4.2 compute platforms.
```
git clone --recursive https://github.com/facebookresearch/localrf && cd localrf
conda create -n localrf python=3.8 -y
conda activate localrf
pip install torch torchvision # Replace here with the command from https://pytorch.org/ corresponding to your compute platform
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict matplotlib scipy==1.6.1 plyfile joblib timm
```

## Preprocessing
Download the [hike scenes](https://drive.google.com/file/d/1DngTRNuZZXpho8-2cjpToa3lGWzxgwqL/view?usp=drive_link).

We use [RAFT](https://github.com/princeton-vl/RAFT) and [DPT](https://github.com/isl-org/DPT) for flow and monocular depth prior.

Get pretrained weights.
```
bash scripts/download_weights.sh
```

Run flow and depth estimation (assuming sorted image files in `${SCENE_DIR}/images`).
```
python scripts/run_flow.py --data_dir ${SCENE_DIR}
python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large
```

Alternatively, run `scripts/preprocess_all.sh` to preprocess all hike scenes.

## Optimization
```
python localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${LOG_DIR} --fov ${FOV}
```
After training completion, test views and smoothed trajectories will be stored in `${LOG_DIR}`. We also provide `scripts/train_all.sh` to initiate optimization on several scenes.

## Citation
```
@inproceedings{meuleman2023localrf,
  author    = {Meuleman, Andreas and Liu, Yu-Lun and Gao, Chen and Huang, Jia-Bin and Kim, Changil and Kim, Min H. and Kopf, Johannes},
  title     = {Progressively Optimized Local Radiance Fields for Robust View Synthesis},
  booktitle = {CVPR},
  year      = {2023},
}
```

## Acknowledgements
The code is available under the MIT license and draws from [TensoRF](https://github.com/apchenstu/TensoRF) and [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF), which are also licensed under the MIT license.
Licenses for these projects can be found in the `licenses/` folder.

We use [RAFT](https://github.com/princeton-vl/RAFT) and [DPT](https://github.com/isl-org/DPT) for flow and monocular depth prior.