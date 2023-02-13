SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short)
SCENES=(forest3 forest4)
SCENES=( sequenced/ours/pg2/skip_0 )

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    docker run  -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --network=host --runtime=nvidia --ipc=host localrf /bin/sh -c "cd /host/; python colmap2nerf.py --images /data/preprocessed/${SCENE}/images --aabb_scale 1 --colmap_db /data/preprocessed/${SCENE}/database.db --text  /data/preprocessed/${SCENE}/text --out /data/preprocessed/${SCENE}/transforms.json --run_colmap"
    
    # python preprocess_video.py --video_name $SCENE; cd DPT; python run_monodepth.py --input_path /data/preprocessed/${SCENE}/images --output_path /data/preprocessed/${SCENE}/depth --model_type dpt_large"
done
