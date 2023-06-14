# Copyright (c) Meta Platforms, Inc. and affiliates.
DATA_DIR=data/s_h

SCENES=(forest1 forest2 forest3 garden1 garden2 garden3 indoor playground university1 university2 university3 university4)

for (( JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++ )) 
do
    SCENE_DIR=${DATA_DIR}/${SCENES[$JOB_COMPLETION_INDEX]}
    echo "Preprocessing $SCENE_DIR"

    python scripts/run_flow.py --data_dir ${SCENE_DIR}
    python DPT/run_monodepth.py --input_path ${SCENE_DIR}/images --output_path ${SCENE_DIR}/depth --model_type dpt_large
done
