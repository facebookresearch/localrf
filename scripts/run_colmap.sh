SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4)
# SCENES=(forest3 forest4)
SCENES=( sequenced/ours/pg/skip_02 )
SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4 indoor sequenced/ours/uw1/skip_0 sequenced/ours/uw2/skip_0  sequenced/ours/uw3/skip_0 sequenced/ours/lake22_1/skip_0 sequenced/ours/lake22_3/skip_0 sequenced/ours/pg/skip_0 sequenced/ours/hike_07_08_gopro_4/skip_2 sequenced/ours/hike_1008_2/skip_2 sequenced/ours/hike_09_26_1/skip_0)
SCENES=(garden1 garden2 garden3 sd1 snow1 snow2 snow3 snow4 forest_short forest3 forest4 indoor)
SCENES=(forest5 forest6 forest7 forest8 forest9)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    # if [ $(expr $JOB_COMPLETION_INDEX % 3) = 2 ]
    # then
    #     BACKGROUND=
    # else
    #     BACKGROUND=-d
    # fi
    BACKGROUND=-d
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    docker run $BACKGROUND -m 12g --cpus 16 -it -e NVIDIA_VISIBLE_DEVICES=$GPU_ID --user "$(id -u):$(id -g)" --rm -v $PWD:/host -v /mnt/datassd/ameuleman:/data --user "$(id -u):$(id -g)" --network=host --runtime=nvidia --ipc=host colmap /bin/sh -c "cd /host/; sudo strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5; bash comparison/multinerf/scripts/local_colmap_and_resize.sh /data/preprocessed/${SCENE}"
done

# -d -m 6g --cpus 12 