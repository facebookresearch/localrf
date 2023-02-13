# uberlapse

## Setup proxy TODO: make it permanent
```
ssh -D 8080 -N ameuleman@ameuleman.sb.facebook.com
```

## Update and create pod
```
docker build -t uberlapse .
docker rm uberlapse
docker run --name uberlapse uberlapse
docker login dtr.thefacebook.com
docker commit -m "Update dependencies" -a ameuleman uberlapse dtr.thefacebook.com/ameuleman/uberlapse
docker push dtr.thefacebook.com/ameuleman/uberlapse

kubectl delete pod uberlapse
kubectl apply -f kube/pod.yaml
kubectl describe pod uberlapse
```
docker build -t meganerf -f meganerf.Dockerfile .
docker rm meganerf
docker run --name meganerf meganerf
docker commit -m "Update dependencies" -a ameuleman meganerf dtr.thefacebook.com/ameuleman/meganerf
docker push dtr.thefacebook.com/ameuleman/meganerf


## Allow ssh into pods
```
kubectl exec -it uberlapse -- bash -c "mkdir -p /home/docker/.ssh"
kubectl cp ~/.ssh/id_rsa.pub uberlapse:/home/docker/.ssh/authorized_keys_tmp
kubectl exec -it uberlapse -- bash -c "cat /home/docker/.ssh/authorized_keys_tmp >> /home/docker/.ssh/authorized_keys"
kubectl exec -it uberlapse -- bash -c "sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5"
kubectl exec -it uberlapse -- bash -c "sudo service ssh start"
kubectl exec -it uberlapse -- bash -c "sudo mkdir -p /home/docker/.cache/torch/hub/checkpoints/"
kubectl exec -it uberlapse -- bash -c "sudo chown docker /home/docker/.cache/torch/hub/checkpoints/ -R"
kubectl exec -it uberlapse -- bash -c "cp /mnt/uberlapse/ameuleman/checkpoints/alexnet-owt-7be5be79.pth /home/docker/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
kubectl exec -it uberlapse -- bash -c "cp -r /mnt/uberlapse/.vscode-server /home/docker/.vscode-server"
kubectl port-forward uberlapse 2301:22
ssh docker@localhost -p 2301
```
kubectl exec -it uberlapse-no-gpu -- bash -c "mkdir -p /home/docker/.ssh"
kubectl cp ~/.ssh/id_rsa.pub uberlapse-no-gpu:/home/docker/.ssh/authorized_keys_tmp
kubectl exec -it uberlapse-no-gpu -- bash -c "cat /home/docker/.ssh/authorized_keys_tmp >> /home/docker/.ssh/authorized_keys"
kubectl exec -it uberlapse-no-gpu -- bash -c "sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5"
kubectl exec -it uberlapse-no-gpu -- bash -c "sudo service ssh start"
kubectl exec -it uberlapse-no-gpu -- bash -c "cp -r /mnt/uberlapse/.vscode-server /home/docker/.vscode-server"
----
kubectl exec -it meganerf -- bash -c "mkdir -p /home/docker/.ssh"
kubectl cp ~/.ssh/id_rsa.pub meganerf:/home/docker/.ssh/authorized_keys_tmp
kubectl exec -it meganerf -- bash -c "cat /home/docker/.ssh/authorized_keys_tmp >> /home/docker/.ssh/authorized_keys"
kubectl exec -it meganerf -- bash -c "sudo service ssh start"
kubectl exec -it meganerf -- bash -c "cp -r /mnt/uberlapse/.vscode-server /home/docker/.vscode-server"

## Repeate every lost connexion:
```
ssh -D 8080 -N ameuleman@ameuleman.sb.facebook.com
kubectl port-forward uberlapse 2301:22
kubectl port-forward uberlapse-no-gpu 2302:22
kubectl port-forward meganerf 2303:22
```

## Sync local files
```
scp -r -P 2301 docker@localhost:/mnt/uberlapse/ameuleman/localTensoRF .

kubectl cp uberlapse:/mnt/uberlapse/ameuleman/localTensoRF localTensoRF

scp -r localTensoRF ameuleman@ameuleman.sb.facebook.com:~/uberlapse
scp kube/job.yaml ameuleman@ameuleman.sb.facebook.com:~/uberlapse/
kubectl cp uberlapse:/mnt/uberlapse/ameuleman/save_eval.py save_eval.py
kubectl cp uberlapse:/mnt/uberlapse/ameuleman/avg_eval.py avg_eval.py
scp save_eval.py ameuleman@ameuleman.sb.facebook.com:~/uberlapse/save_eval.py
scp avg_eval.py ameuleman@ameuleman.sb.facebook.com:~/uberlapse/avg_eval.py
#kubectl cp localTensoRF uberlapse:/mnt/uberlapse/ameuleman/localTensoRF
#kubectl cp save_eval.py uberlapse:/mnt/uberlapse/ameuleman/save_eval.py
#kubectl cp avg_eval.py uberlapse:/mnt/uberlapse/ameuleman/avg_eval.py
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
