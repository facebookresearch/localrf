SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short)
SCENES=(forest3 forest4)
SCENES=( sequenced/ours/pg/skip_0 )
SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4 sequenced/ours/uw1/skip_0 sequenced/ours/uw2/skip_0  sequenced/ours/uw3/skip_0 sequenced/ours/lake22_1/skip_0 sequenced/ours/lake22_3/skip_0 sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2 sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
SCENES=( indoor )
SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4 indoor)
SCENES=(forest5 forest6 forest7 forest8 forest9)

SUFIX=
for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    docker run -d -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --network=host --runtime=nvidia --ipc=host localrf /bin/sh -c "cd /host/; mkdir /data/preprocessed/${SCENE}/text${SUFIX}; colmap model_converter --input_path /data/preprocessed/${SCENE}/sparse${SUFIX}/0 --output_path /data/preprocessed/${SCENE}/text${SUFIX} --output_type TXT; python colmap2nerf.py --images /data/preprocessed/${SCENE}/images --aabb_scale 1 --colmap_db /data/preprocessed/${SCENE}/database${SUFIX}.db --text  /data/preprocessed/${SCENE}/text${SUFIX} --out /data/preprocessed/${SCENE}/transforms${SUFIX}.json; "
    
    # python preprocess_video.py --video_name $SCENE; cd DPT; python run_monodepth.py --input_path /data/preprocessed/${SCENE}/images --output_path /data/preprocessed/${SCENE}/depth --model_type dpt_large"
done
