# SCENES=(sequenced/ours/uw1/skip_0 sequenced/ours/uw2/skip_0  sequenced/ours/uw3/skip_0 sequenced/ours/lake22_1/skip_0 sequenced/ours/lake22_3/skip_0 sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2)
# SCENES=(sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
# SCENES=(sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2 sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
# sequenced/intermediate/M60/skip_4 sequenced/intermediate/Panther/skip_4 sequenced/intermediate/Train/skip_4 sequenced/advanced/Auditorium/skip_4 sequenced/advanced/Ballroom/skip_4 sequenced/advanced/Courtroom/skip_4 sequenced/advanced/Museum/skip_4 sequenced/train/Caterpillar/skip_4 sequenced/train/Church/skip_4)
SCENES=(garden1 garden2 garden3 sd1 snow1 snow2) 
SCENES=(snow3 snow4 forest_short forest3 forest4 forest5)
# SCENES=(forest6 forest7 forest8 forest9)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    echo "$SCENE"
    docker run -d -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --user "$(id -u):$(id -g)" --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --runtime=nvidia mipnerf /bin/sh -c "cd /host/comparison/multinerf; python -m train --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir = '/data/preprocessed/${SCENE}'\" --gin_bindings=\"Config.checkpoint_dir = '/data/logs_eval/mipnerf360/${SCENE}/checkpoints'\" --logtostderr; python -m render --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir = '/data/preprocessed/${SCENE}'\" --gin_bindings=\"Config.checkpoint_dir = '/data/logs_eval/mipnerf360/${SCENE}/checkpoints'\" --gin_bindings=\"Config.render_dir = '/data/logs_eval/mipnerf360/${SCENE}/render'\" --gin_bindings=\"Config.render_path = False\" --logtostderr; python -m render --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir = '/data/preprocessed/${SCENE}'\" --gin_bindings=\"Config.checkpoint_dir = '/data/logs_eval/mipnerf360/${SCENE}/checkpoints'\" --gin_bindings=\"Config.render_dir = '/data/logs_eval/mipnerf360/${SCENE}/render_smooth4'\" --gin_bindings=\"Config.render_path = True\" --logtostderr"
    sleep 100
done
# SCENE=sequenced/ours/hike_07_08_gopro_4/skip_2
# docker run -it -e NVIDIA_VISIBLE_DEVICES=4 --user "$(id -u):$(id -g)" --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --runtime=nvidia mipnerf /bin/sh -c "cd /host/comparison/multinerf; python -m render --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir = '/data/preprocessed/${SCENE}'\" --gin_bindings=\"Config.checkpoint_dir = '/data/logs_eval/mipnerf360/${SCENE}/checkpoints'\" --gin_bindings=\"Config.render_dir = '/data/logs_eval/mipnerf360/${SCENE}/render_r'\" --gin_bindings=\"Config.render_path = True\" --logtostderr"
