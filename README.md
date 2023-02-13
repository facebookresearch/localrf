# uberlapse

## Setup

* 

## Data
```
rsync -a /mnt/datassd/ameuleman/videos/ ameuleman@vcgpuserver1.kaist.ac.kr:/mnt/datassd/ameuleman/videos
rsync -a /mnt/datassd/ameuleman/preprocessed/ ameuleman@vcgpuserver1.kaist.ac.kr:/mnt/datassd/ameuleman/preprocessed
rsync -a /mnt/datassd/ameuleman/sequenced/ ameuleman@vcgpuserver1.kaist.ac.kr:/mnt/datassd/ameuleman/sequenced
rsync -a ameuleman@vcgpuserver1.kaist.ac.kr:/mnt/datassd/ameuleman/ /mnt/datassd/ameuleman/
//vcserver2.kaist.ac.kr/vcpaper4 /home/ameuleman/network_drives/vcpaper4 cifs -o username=ameuleman -o uid=ameuleman
```

## Docker
```
docker build -t localrf --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
docker build -t mipnerf -f mipnerf.Dockerfile .
docker build -t nerfacto -f nerfacto.Dockerfile .
docker build -t colmap -f colmap.Dockerfile .

docker logs --follow 

docker kill $(docker ps -q)

JOB_COMPLETION_INDEX=0
SCENE=sd1
docker run -it -e NVIDIA_VISIBLE_DEVICES=$JOB_COMPLETION_INDEX --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --network=host --runtime=nvidia --ipc=host localrf /bin/sh -c \
    "cd /host/; \
    python preprocess_video.py --video_name $SCENE"
```

## conda
```
conda create -n localrf python=3.8
conda activate localrf
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard imageio easydict matplotlib scipy==1.9.1 kornia plyfile joblib timm
```


## tensorboard
```
tensorboard --logdir="data/logs/" --port=1285 --bind_all
http://localhost:1285/
```

## Preprocessing
Set sequences names in those files before doing anything
```
kubectl apply -f kube/preprocess_job.yaml
## Wait for it to finish
kubectl apply -f kube/colmap_multinerf_job.yaml
kubectl apply -f kube/colmap_ngp_job.yaml
## Wait for those to finish
bash run_nerfpp_preproc.sh
python split_train_test.py
## Wait for run_nerfpp_preproc
python split_train_test_npp.py
```

```
SCENE=office
python preprocess_video.py --video_name $SCENE --scale 1 --multiGPU 1
cd DPT
python run_monodepth.py --input_path ../data/sequenced/$SCENE/images --output_path ../data/sequenced/$SCENE/depth --model_type dpt_large
```

## FuSta
```
python main.py /mnt/datassd/ameuleman/preprocessed/sequenced/ours/hike_07_08_gopro_4/images/ /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced/ours/hike_07_08_gopro_4/skip_2/ /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced/ours/hike_07_08_gopro_4/skip_2/wf/

python run_FuSta.py --load FuSta_model/checkpoint/model_epoch050.pth --input_frames_path /mnt/datassd/ameuleman/preprocessed/sequenced/ours/hike_07_08_gopro_4/images/ --warping_field_path /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced/ours/hike_07_08_gopro_4/skip_2/wf/ --output_path /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced/ours/hike_07_08_gopro_4/fusta/ --temporal_width 41 --temporal_step 4


python main.py /mnt/datassd/ameuleman/preprocessed/sequenced/ours/hike_07_08_gopro_4/images/ /mnt/datassd/ameuleman/logs_eval/FuSta2/sequenced/ours/hike_07_08_gopro_4/skip_2/ /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced2/ours/hike_07_08_gopro_4/skip_2/wf/

python run_FuSta.py --load FuSta_model/checkpoint/model_epoch050.pth --input_frames_path /mnt/datassd/ameuleman/preprocessed/sequenced/ours/hike_07_08_gopro_4/images/ --warping_field_path /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced2/ours/hike_07_08_gopro_4/skip_2/wf/ --output_path /mnt/datassd/ameuleman/logs_eval/FuSta/sequenced/ours/hike_07_08_gopro_4/fusta2/ --temporal_width 41 --temporal_step 4
```

## Video and ffmpeg
For website
```-frames:v 900
~/ffmpeg/ffmpeg -r 30 -i website_full/videos/input/hike2.mp4 -crf 28  -preset veryslow -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p website/videos/input/hike2.mp4
```
```
~/ffmpeg/ffmpeg -framerate 30 -pattern_type glob -i 'data/sequenced/images/*.jpg' -vf scale=480:272 -c:v libx264 -pix_fmt yuv420p data/scaled/hike_07_08_gopro_2.mp4

~/ffmpeg/ffmpeg -i input0.mp4 -i input1.mp4 -filter_complex hstack=inputs=2 output.mp4

SCENE=office_gopro
~/ffmpeg/ffmpeg -i data/videos/$SCENE.mp4 -vf scale=480:272 data/scaled/$SCENE.mp4
~/ffmpeg/ffmpeg -r 30 -i data/scaled/${SCENE}.mp4 -i tmp/${SCENE}_smooth.mp4 -filter_complex hstack=inputs=2 tmp/stacked/$SCENE.mp4

~/ffmpeg/ffmpeg -r 2 -i data/scaled/$SCENE.mp4 -filter:v select="mod(n-1\,2)" -r 1 data/scaled/${SCENE}_s.mp4
~/ffmpeg/ffmpeg -r 2 -i data/scaled/${SCENE}_s.mp4 -filter:v select="mod(n-1\,2)" -r 1 data/scaled/${SCENE}_q.mp4
SCENE=${SCENE}_q


ffmpeg -framerate 30 -i uw2/render/%05d.png -c:v libx264 -pix_fmt yuv420p uw2/university2.mp4
```

## Misc
```
kubectl-login --env sea112-hpcc --namespace compphoto --username $(id -un)
kubectl-login --env ash6-hpcc --namespace compphoto --username $(id -un)
python localTensoRF/train.py --config localTensoRF/configs/uberlapse.txt --multiGPU 1 --n_iters 6000 --n_max_frames 50
```

https://www.internalfb.com/intern/wiki/Computational_Photography/Kubernetes_Quick_Start_Guide/
https://www.internalfb.com/intern/wiki/High_Performance_cloud_Computing/Kubernetes/Accessing_Kubernetes/

https://sanda-dev.medium.com/ssh-into-kubernetes-pod-without-public-ip-access-fbb9da7f7b26

https://confluence.jaytaala.com/display/TKB/Create+a+persistent+SSH+tunnel+between+servers+with+systemd
