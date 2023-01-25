# SCENES=(intermediate/M60 intermediate/Panther intermediate/Playground intermediate/Train advanced/Auditorium advanced/Ballroom advanced/Courtroom advanced/Museum train/Barn train/Caterpillar train/Church train/Courthouse train/Ignatius train/Meetingroom train/Truck)
# SKIPS=(4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3 3 3)

SCENES=(ours/uw1 ours/uw2  ours/uw3 ours/uw4 ours/lake22_1 ours/lake22_2 ours/lake22_3  ours/lake22_4 ours/pg ours/hike_07_08_gopro_4 ours/hike_09_26_7 ours/hike_1008_4 ours/hike_1008_6 ours/hike_1008_2 ours/hike_09_26_1)
SKIPS=(0 0 0 0 0 0 0 2 0 2 0 3 3 2 0)

for JOB_COMPLETION_INDEX in {0..14}
do
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    SKIP=${SKIPS[$JOB_COMPLETION_INDEX]}
    SCENE_NAME=${SCENE}/skip_${SKIP}
    echo "$SCENE_NAME"
    kubectl cp uberlapse-no-gpu:/mnt/uberlapse/ameuleman/data/sequenced/${SCENE_NAME}/in.mp4 ../data/input/${SCENE}.mp4
done
