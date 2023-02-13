SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4 forest5 forest6 forest7 forest8 forest9)
SCENES=(forest_short forest3 forest4)
SCENES=(sequenced/ours/uw1/skip_0 sequenced/ours/uw2/skip_0  sequenced/ours/uw3/skip_0 sequenced/ours/lake22_1/skip_0 sequenced/ours/lake22_3/skip_0 sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2 sequenced/ours/hike_1008_2/skip_2) 
# SCENES=(sequenced/ours/hike_09_26_1/skip_0)
# SCENES=(sequenced/intermediate/M60/skip_4 sequenced/intermediate/Panther/skip_4 sequenced/intermediate/Train/skip_4 sequenced/advanced/Auditorium/skip_4 sequenced/advanced/Ballroom/skip_4 sequenced/advanced/Courtroom/skip_4 sequenced/advanced/Museum/skip_4)
# SCENES=(sequenced/train/Caterpillar/skip_4 sequenced/train/Church/skip_4)
SCENES=( sequenced/advanced/Auditorium/skip_4 )
SCENES=(garden1 sd1 snow1 snow2 snow3)
SCENES=(garden2 garden3 snow4)
SCENES=(forest_short forest3 forest4 forest5)
SCENES=(forest6 forest7 forest8 forest9)

cd comparison/nerfstudio

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    echo "$SCENE ${GPU_ID}"
    { export CUDA_VISIBLE_DEVICES=${GPU_ID}; ns-train nerfacto --data /mnt/datassd/ameuleman/preprocessed/${SCENE} --output-dir /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE} --viewer.websocket-port 800${GPU_ID} --viewer.quit-on-train-completion True --experiment-name . --timestamp 0 nerfstudio-data; ns-render --load-config /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format images --output-path /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}/test --traj filename --camera-path-filename /mnt/datassd/ameuleman/preprocessed/${SCENE}/transforms.json; ns-render --load-config /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format video --output-path /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}.mp4 --traj filename --camera-path-filename /mnt/datassd/ameuleman/preprocessed/${SCENE}/transforms.json; } & 
    # ns-eval --load-config /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-path /mnt/datassd/ameuleman/logs_eval/nerfacto/${SCENE}/test/metrics.json
done
#     export CUDA_VISIBLE_DEVICES=${GPU_ID}; ns-train nerfacto --data /data/preprocessed/${SCENE} --output-dir /data/logs_eval/nerfacto/${SCENE} --viewer.quit-on-train-completion True --experiment-name . --timestamp 0 nerfstudio-data; ns-render --load-config /data/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format images --output-path /data/logs_eval/nerfacto/${SCENE}/test --traj filename --camera-path-filename /data/preprocessed/${SCENE}/transforms.json; ns-render --load-config /data/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format video --output-path /data/logs_eval/nerfacto/${SCENE} --traj filename --camera-path-filename /data/preprocessed/${SCENE}/transforms.json & GPU_ID=$!
# done
# # python -m pip install setuptools; python -m pip install -e .; python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# docker run -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --ipc=private --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --runtime=nvidia --ipc=private localrf /bin/sh -c "cd /host/comparison/nerfstudio; python -m pip install setuptools; python -m pip install -e .; python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch; ns-train nerfacto --data /data/preprocessed/${SCENE} --output-dir /data/logs_eval/nerfacto/${SCENE} --viewer.quit-on-train-completion True --experiment-name . --timestamp 0 nerfstudio-data;  ns-render --load-config /data/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format images --output-path /data/logs_eval/nerfacto/${SCENE}/test --traj filename --camera-path-filename /data/preprocessed/${SCENE}/transforms.json; ns-render --load-config /data/logs_eval/nerfacto/${SCENE}/nerfacto/0/config.yml --output-format video --output-path /data/logs_eval/nerfacto/${SCENE} --traj filename --camera-path-filename /data/preprocessed/${SCENE}/transforms.json"
