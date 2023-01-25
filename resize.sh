#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


SCENES=(intermediate/M60 intermediate/Panther intermediate/Playground intermediate/Train advanced/Auditorium advanced/Ballroom advanced/Courtroom advanced/Museum train/Barn train/Caterpillar train/Church train/Courthouse train/Ignatius train/Meetingroom train/Truck)
SKIPS=(4 4 4 4 4 4 4 4 4 4 4 4 4 4 4)

for JOB_COMPLETION_INDEX in {0..14}
do
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    SKIP=${SKIPS[$JOB_COMPLETION_INDEX]}
    SCENE_NAME=${SCENE}/skip_${SKIP}
    echo "$SCENE_NAME"
    DATASET_PATH=data/sequenced/${SCENE_NAME}

    rm -rf "$DATASET_PATH"/images_2
    cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_2
    pushd "$DATASET_PATH"/images_2
    ls | xargs -P 8 -I {} mogrify -resize 50% {}
    popd

    rm -rf "$DATASET_PATH"/images_4
    cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4
    pushd "$DATASET_PATH"/images_4
    ls | xargs -P 8 -I {} mogrify -resize 25% {}
    popd

    rm -rf "$DATASET_PATH"/images_8
    cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_8
    pushd "$DATASET_PATH"/images_8
    ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
    popd
done

