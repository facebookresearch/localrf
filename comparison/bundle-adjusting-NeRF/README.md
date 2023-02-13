## BARF :vomiting_face:: Bundle-Adjusting Neural Radiance Fields
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/),
[Wei-Chiu Ma](http://people.csail.mit.edu/weichium/),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/),
and [Simon Lucey](http://ci2cv.net/people/simon-lucey/)  
IEEE International Conference on Computer Vision (ICCV), 2021 (**oral presentation**) 

Project page: https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF  
Paper: https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/paper.pdf  
arXiv preprint: https://arxiv.org/abs/2104.06405  

We provide PyTorch code for all experiments: planar image alignment, NeRF/BARF on both synthetic (Blender) and real-world (LLFF) datasets, and a template for BARFing on your custom sequence.

--------------------------------------

### Prerequisites

- Note: for Azure ML support for this repository, please consider checking out [this branch](https://github.com/szymanowiczs/bundle-adjusting-NeRF/tree/azureml_training_script) by Stan Szymanowicz.

This code is developed with Python3 (`python3`). PyTorch 1.9+ is required.  
It is recommended use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Install the dependencies and activate the environment `barf-env` with
```bash
conda env create --file requirements.yaml python=3
conda activate barf-env
```
Initialize the external submodule dependencies with
```bash
git submodule update --init --recursive
```

--------------------------------------

### Dataset

- #### Synthetic data (Blender) and real-world data (LLFF)
    Both the Blender synthetic data and LLFF real-world data can be found in the [NeRF Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
For convenience, you can download them with the following script: (under this repo)
  ```bash
  # Blender
  gdown --id 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG # download nerf_synthetic.zip
  unzip nerf_synthetic.zip
  rm -f nerf_synthetic.zip
  mv nerf_synthetic data/blender
  # LLFF
  gdown --id 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g # download nerf_llff_data.zip
  unzip nerf_llff_data.zip
  rm -f nerf_llff_data.zip
  mv nerf_llff_data data/llff
  ```
  The `data` directory should contain the subdirectories `blender` and `llff`.
  If you already have the datasets downloaded, you can alternatively soft-link them within the `data` directory.

- #### <span style="color:red">Test your own sequence!</span>
  If you want to try BARF on your own sequence, we provide a template data file in `data/iphone.py`, which is an example to read from a sequence captured by an iPhone 12.
  You should modify `get_image()` to read each image sample and set the raw image sizes (`self.raw_H`, `self.raw_W`) and focal length (`self.focal`) according to your camera specs.  
  You may ignore the camera poses as they are assumed unknown in this case, which we simply set to zero vectors.

--------------------------------------

### Running the code

- #### BARF models
  To train and evaluate BARF:
  ```bash
  # <GROUP> and <NAME> can be set to your likes, while <SCENE> is specific to datasets
  
  # Blender (<SCENE>={chair,drums,ficus,hotdog,lego,materials,mic,ship})
  python3 train.py --group=<GROUP> --model=barf --yaml=barf_blender --name=<NAME> --data.scene=<SCENE> --barf_c2f=[0.1,0.5]
  python3 evaluate.py --group=<GROUP> --model=barf --yaml=barf_blender --name=<NAME> --data.scene=<SCENE> --data.val_sub= --resume
  
  # LLFF (<SCENE>={fern,flower,fortress,horns,leaves,orchids,room,trex})
  python3 train.py --group=<GROUP> --model=barf --yaml=barf_llff --name=<NAME> --data.scene=<SCENE> --barf_c2f=[0.1,0.5]
  python3 evaluate.py --group=<GROUP> --model=barf --yaml=barf_llff --name=<NAME> --data.scene=<SCENE> --resume
  ```
  All the results will be stored in the directory `output/<GROUP>/<NAME>`.
  You may want to organize your experiments by grouping different runs in the same group.

  To train baseline models:
  - Full positional encoding: omit the `--barf_c2f` argument.
  - No positional encoding: add `--arch.posenc!`.
  
  If you want to evaluate a checkpoint at a specific iteration number, use `--resume=<ITER_NUMBER>` instead of just `--resume`.

- #### Training the original NeRF
  If you want to train the reference NeRF models (assuming known camera poses):
  ```bash
  # Blender
  python3 train.py --group=<GROUP> --model=nerf --yaml=nerf_blender --name=<NAME> --data.scene=<SCENE>
  python3 evaluate.py --group=<GROUP> --model=nerf --yaml=nerf_blender --name=<NAME> --data.scene=<SCENE> --data.val_sub= --resume
  
  # LLFF
  python3 train.py --group=<GROUP> --model=nerf --yaml=nerf_llff --name=<NAME> --data.scene=<SCENE>
  python3 evaluate.py --group=<GROUP> --model=nerf --yaml=nerf_llff --name=<NAME> --data.scene=<SCENE> --resume
  ```
  If you wish to replicate the results from the original NeRF paper, use `--yaml=nerf_blender_repr` or `--yaml=nerf_llff_repr` instead for Blender or LLFF respectively.
  There are some differences, e.g. NDC will be used for the LLFF forward-facing dataset.
  (The reference NeRF models considered in the paper do not use NDC to parametrize the 3D points.)

- #### Planar image alignment experiment
  If you want to try the planar image alignment experiment, run:
  ```bash
  python3 train.py --group=<GROUP> --model=planar --yaml=planar --name=<NAME> --seed=3 --barf_c2f=[0,0.4]
  ```
  This will fit a neural image representation to a single image (default to `data/cat.jpg`), which takes a couple of minutes to optimize on a modern GPU.
  The seed number is set to reproduce the pre-generated warp perturbations in the paper.
  For the baseline methods, modify the arguments similarly as in the NeRF case above:
  - Full positional encoding: omit the `--barf_c2f` argument.
  - No positional encoding: add `--arch.posenc!`.

  A video `vis.mp4` will also be created to visualize the optimization process.

- #### Visualizing the results
  We have included code to visualize the training over TensorBoard and Visdom.
  The TensorBoard events include the following:
  - **SCALARS**: the rendering losses and PSNR over the course of optimization. For BARF, the rotational/translational errors with respect to the given poses are also computed.
  - **IMAGES**: visualization of the RGB images and the RGB/depth rendering.
  
  We also provide visualization of 3D camera poses in Visdom.
  Run `visdom -port 9000` to start the Visdom server.  
  The Visdom host server is default to `localhost`; this can be overridden with `--visdom.server` (see `options/base.yaml` for details).
  If you want to disable Visdom visualization, add `--visdom!`.

  The `extract_mesh.py` script provides a simple way to extract the underlying 3D geometry using marching cubes. Run as follows:
  ```bash
  python3 extract_mesh.py --group=<GROUP> --model=barf --yaml=barf_blender --name=<NAME> --data.scene=<SCENE> --data.val_sub= --resume
  ```
  This works for both BARF and the original NeRF (by modifying the command line accordingly). This is currently supported only for the Blender dataset.

--------------------------------------
### Codebase structure

The main engine and network architecture in `model/barf.py` inherit those from `model/nerf.py`.
This codebase is structured so that it is easy to understand the actual parts BARF is extending from NeRF.
It is also simple to build your exciting applications upon either BARF or NeRF -- just inherit them again!
This is the same for dataset files (e.g. `data/blender.py`).

To understand the config and command lines, take the below command as an example:
```bash
python3 train.py --group=<GROUP> --model=barf --yaml=barf_blender --name=<NAME> --data.scene=<SCENE> --barf_c2f=[0.1,0.5]
```
This will run `model/barf.py` as the main engine with `options/barf_blender.yaml` as the main config file.
Note that `barf` hierarchically inherits `nerf` (which inherits `base`), making the codebase customizable.  
The complete configuration will be printed upon execution.
To override specific options, add `--<key>=value` or `--<key1>.<key2>=value` (and so on) to the command line. The configuration will be loaded as the variable `opt` throughout the codebase.  
  
Some tips on using and understanding the codebase:
- The computation graph for forward/backprop is stored in `var` throughout the codebase.
- The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
- If you are using a multi-GPU machine, you can add `--gpu=<gpu_number>` to specify which GPU to use. Multi-GPU training/evaluation is currently not supported.
- To resume from a previous checkpoint, add `--resume=<ITER_NUMBER>`, or just `--resume` to resume from the latest checkpoint.
- (to be continued....)
  
--------------------------------------

If you find our code useful for your research, please cite
```
@inproceedings{lin2021barf,
  title={BARF: Bundle-Adjusting Neural Radiance Fields},
  author={Lin, Chen-Hsuan and Ma, Wei-Chiu and Torralba, Antonio and Lucey, Simon},
  booktitle={IEEE International Conference on Computer Vision ({ICCV})},
  year={2021}
}
```

Please contact me (chlin@cmu.edu) if you have any questions!
