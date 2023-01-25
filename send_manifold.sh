mkdir data/tmp

SCENES=(intermediate/M60 intermediate/Panther intermediate/Playground intermediate/Train advanced/Auditorium advanced/Ballroom advanced/Courtroom advanced/Museum train/Barn train/Caterpillar train/Church train/Courthouse train/Ignatius train/Meetingroom train/Truck)
SKIPS=(4 4 4 4 4 4 4 4 4 4 4 4 4 4 4)

for JOB_COMPLETION_INDEX in {0..14}
do
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    SKIP=${SKIPS[$JOB_COMPLETION_INDEX]}
    SCENE_NAME=${SCENE}/skip_${SKIP}
    echo "$SCENE_NAME"

    kubectl cp uberlapse:/mnt/uberlapse/ameuleman/data/sequenced/ data/tmp/

done