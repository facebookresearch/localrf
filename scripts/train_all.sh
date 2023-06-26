# Copyright (c) Meta Platforms, Inc. and affiliates.
DATA_DIR=data
EXP=logs
N_GPU=8

DATA_PREFIX=s_h
SCENES=(forest1 forest2 forest3 garden1 garden2 garden3 indoor playground university1 university2 university3 university4)
FOVS=(59 89 69 59 69 69 69 69 85 73 73 69)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    SCENE=${DATA_PREFIX}/${SCENES[$JOB_COMPLETION_INDEX]}
    FOV=${FOVS[$JOB_COMPLETION_INDEX]}
    echo "$SCENE on GPU $GPU_ID with FoV $FOV"

    SCENE_DIR=${DATA_DIR}/${SCENE}
    LOG_DIR=${DATA_DIR}/${EXP}/${SCENE}
    mkdir -p ${LOG_DIR}

    nohup python -u localTensoRF/train.py --datadir ${SCENE_DIR} --logdir ${LOG_DIR} --fov ${FOV} --device cuda:$GPU_ID > ${LOG_DIR}/logs.out &
done
