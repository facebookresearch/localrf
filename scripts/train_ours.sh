SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow4)
# SCENES=(forest_short forest3 forest4 snow3)
SCENES=(sequenced/ours/uw1/skip_0 sequenced/ours/uw2/skip_0  sequenced/ours/uw3/skip_0 sequenced/ours/lake22_1/skip_0 sequenced/ours/lake22_3/skip_0 sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2)
# SCENES=(sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
# SCENES=(sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2 sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
# sequenced/intermediate/M60/skip_4 sequenced/intermediate/Panther/skip_4 sequenced/intermediate/Train/skip_4 sequenced/advanced/Auditorium/skip_4 sequenced/advanced/Ballroom/skip_4 sequenced/advanced/Courtroom/skip_4 sequenced/advanced/Museum/skip_4 sequenced/train/Caterpillar/skip_4 sequenced/train/Church/skip_4)
SCENES=( sequenced/ours/uw1/skip_0 )

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    echo "$SCENE on GPU $GPU_ID"
    docker run -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --runtime=nvidia localrf /bin/sh -c "cd /host; python localTensoRF/train.py --datadir /data/preprocessed/${SCENE} --logdir /data/logs/${SCENE}"
done
