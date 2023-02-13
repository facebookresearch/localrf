# Mega-NeRF

## Prepare Custom Data
SCENE_NAME=train_small/Caterpillar
rm -rf ../../data/logs_eval/meganerf/${SCENE_NAME}
python scripts/colmap_to_mega_nerf.py --model_path ../../data/sequenced/${SCENE_NAME}/sparse/0 --images_path ../../data/sequenced/${SCENE_NAME}/images --output_path ../../data/logs_eval/meganerf/${SCENE_NAME} --scale 6

## Training 1
<!-- rm -rf ../../data/logs_eval/meganerf/${SCENE_NAME}/masks -->
python scripts/create_cluster_masks.py --config configs/mega-nerf/quad.yaml --dataset_path ../../data/logs_eval/meganerf/${SCENE_NAME}  --output ../../data/logs_eval/meganerf/${SCENE_NAME}/masks --grid_dim 2 2


## Training 2
for CENTROID in {0..3}
do
    python mega_nerf/train.py \
        --config_file configs/mega-nerf/quad.yaml \
        --exp_name ../../data/logs_eval/meganerf/${SCENE_NAME}/exp-${CENTROID} \
        --dataset_path ../../data/logs_eval/meganerf/${SCENE_NAME}  \
        --chunk_paths ../../data/logs_eval/meganerf/${SCENE_NAME}/chunk-${CENTROID} \
        --cluster_mask_path ../../data/logs_eval/meganerf/${SCENE_NAME}/masks/${CENTROID} \
        --train_iterations 100
done

## Training 3:
python scripts/merge_submodules.py \
    --config_file configs/mega-nerf/quad.yaml \
    --ckpt_prefix ../../data/logs_eval/meganerf/${SCENE_NAME}/exp- \
    --centroid_path ../../data/logs_eval/meganerf/${SCENE_NAME}/masks/params.pt \
    --output ../../data/logs_eval/meganerf/${SCENE_NAME}/results \
    --train_iterations 100

## Evaluation
python mega_nerf/eval.py \
    --config_file configs/mega-nerf/quad.yaml \
    --exp_name ../../data/logs_eval/meganerf/${SCENE_NAME}/exp- \
    --dataset_path ../../data/logs_eval/meganerf/${SCENE_NAME}  \
    --container_path ../../data/logs_eval/meganerf/${SCENE_NAME}/results






This repository contains the code needed to train [Mega-NeRF](https://meganerf.cmusatyalab.org/) models and generate the sparse voxel octrees used by the Mega-NeRF-Dynamic viewer.

The codebase for the Mega-NeRF-Dynamic viewer can be found [here](https://github.com/cmusatyalab/mega-nerf-viewer).

**Note:** This is a preliminary release and there may still be outstanding bugs.

## Citation

```
@InProceedings{Turki_2022_CVPR,
    author    = {Turki, Haithem and Ramanan, Deva and Satyanarayanan, Mahadev},
    title     = {Mega-NERF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12922-12931}
}
```

## Demo
![](demo/rubble-orbit.gif)
![](demo/building-orbit.gif)

## Setup

```
conda env create -f environment.yml
conda activate mega-nerf
```

The codebase has been mainly tested against CUDA >= 11.1 and V100/2080 Ti/3090 Ti GPUs. 1080 Ti GPUs should work as well although training will be much slower.

## Pretrained Models

### Trained with 8 submodules (to compare with main paper)

- Rubble: [model](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-grid-8.tgz)
- Building: [model](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-grid-8.tgz)
- Quad: [model](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-grid-8.tgz)
- Residence: [model](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-grid-8.tgz)
- Sci-Art: [model](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-grid-8.tgz)
- Campus: [model](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-8.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-grid-8.tgz)

### Larger models (trained with 25 submodules with 512 channels each)

- Rubble: [model](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-25-w-512.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-grid-25.tgz)
- Building: [model](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-25-w-512.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-grid-25.tgz)
- Residence: [model](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-25-w-512.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-grid-25.tgz)
- Sci-Art: [model](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-25-w-512.pt) [cluster masks](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-grid-25.tgz)

## Data

### Mill 19

- The Building scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz).
- The Rubble scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz).

### UrbanScene 3D

1. Download the raw photo collections from the [UrbanScene3D](https://vcc.tech/UrbanScene3D/) dataset
2. Download the refined camera poses for one of the scenes below:
  - [Residence](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm.tgz)
  - [Sci-Art](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz)
  - [Campus](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm.tgz)
4. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Quad 6k Dataset

1. Download the raw photo collections from [here](http://vision.soic.indiana.edu/disco_files/ArtsQuad_dataset.tar).
2. Download [the refined camera poses](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm.tgz)
3. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Custom Data

We strongly recommend using [PixSFM](https://github.com/cvg/pixel-perfect-sfm) to refine camera poses for your own datasets. Mega-NeRF also assumes that the dataset is properly geo-referenced/aligned such that the second value of its `ray_altitude_range` parameter properly corresponds to ground level. If using PixSFM/COLMAP the [model_aligner](https://colmap.github.io/faq.html#geo-registration) utility might be helpful, with [Manhattan world alignment](https://colmap.github.io/faq.html#manhattan-world-alignment) being a possible fallback option if GPS alignment is not possible. We provide a [script](https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py) to convert from PixSFM/COLMAP output to the format Mega-NeRF expects.

If creating a custom dataset manually, the expected directory structure is:
- /coordinates.pt: [Torch file](https://pytorch.org/docs/stable/generated/torch.save.html) that should contain the following keys:
  - 'origin_drb': Origin of scene in real-world units
  - 'pose_scale_factor': Scale factor mapping from real-world unit (ie: meters) to [-1, 1] range
- '/{val|train}/rgbs/': JPEG or PNG images
- '/{val|train}/metadata/': Image-specific image metadata saved as a torch file. Each image should have a corresponding metadata file with the following file format: {rgb_stem}.pt. Each metadata file should contain the following keys:
  - 'W': Image width
  - 'H': Image height
  - 'intrinsics': Image intrinsics in the following form: [fx, fy, cx, cy]
  - 'c2w': Camera pose. 3x3 camera matrix with the convention used in the original [NeRF repo](https://github.com/bmild/nerf), ie: x: down, y: right, z: backwards, followed by the following transformation: ```torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1)```

## Training

1. Generate the training partitions for each submodule: ```python scripts/create_cluster_masks.py --config configs/mega-nerf/${DATASET_NAME}.yml --dataset_path $DATASET_PATH  --output $MASK_PATH --grid_dim $GRID_X $GRID_Y```
    - **Note:** this can be run across multiple GPUs by instead running ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS --max_restarts 0 scripts/create_cluster_masks.py <args>```
2. Train each submodule: ```python mega_nerf/train.py --config_file configs/mega-nerf/${DATASET_NAME}.yml --exp_name $EXP_PATH --dataset_path $DATASET_PATH --chunk_paths $SCRATCH_PATH --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX}```
    - **Note:** training with against full scale data will write hundreds of GBs / several TBs of shuffled data to disk. You can downsample the training data using ```train_scale_factor``` option.
    - **Note:** we provide [a utility script](parscripts/run_8.txt) based on [parscript](https://github.com/mtli/parscript) to start multiple training jobs in parallel. It can run through the following command: ```CONFIG_FILE=configs/mega-nerf/${DATASET_NAME}.yaml EXP_PREFIX=$EXP_PATH DATASET_PATH=$DATASET_PATH CHUNK_PREFIX=$SCRATCH_PATH MASK_PATH=$MASK_PATH python -m parscript.dispatcher parscripts/run_8.txt -g $NUM_GPUS```
3. Merge the trained submodules into a unified Mega-NeRF model: ```python scripts/merge_submodules.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml  --ckpt_prefix ${EXP_PREFIX}- --centroid_path ${MASK_PATH}/params.pt --output $MERGED_OUTPUT```

## Evaluation

Single-GPU evaluation: ```python mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml  --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT```

Multi-GPU evaluation: ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml  --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT```

## Octree Extraction (for use by Mega-NeRF-Dynamic viewer)

```
python scripts/create_octree.py --config configs/mega-nerf/${DATASET_NAME}.yaml --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT --output $OCTREE_PATH
 ```

## Acknowledgements

Large parts of this codebase are based on existing work in the [nerf_pl](https://github.com/kwea123/nerf_pl), [NeRF++](https://github.com/Kai-46/nerfplusplus), and [Plenoctree](https://github.com/sxyu/plenoctree) repositories. We use [svox](https://github.com/sxyu/svox) to serialize our sparse voxel octrees and the generated structures should be largely compatible with that codebase.
