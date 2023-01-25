import json
import copy
import os
import shutil

scenes = [
    # "train_small/Caterpillar"
    "intermediate/M60",
    "intermediate/Panther",
    "intermediate/Playground",
    "intermediate/Train",
    "advanced/Auditorium",
    "advanced/Ballroom",
    "advanced/Courtroom",
    "advanced/Museum",
    "train/Barn",
    "train/Caterpillar",
    "train/Church",
    "train/Courthouse",
    "train/Ignatius",
    "train/Meetingroom",
    "train/Truck",
]
suffix = "/skip_4"
# suffix = ""

for scene in scenes:
    workspace = f"data/sequenced/{scene}{suffix}"

    with open(f"{workspace}/transforms.json", 'r') as f:
        in_json = json.loads(f.read())

    train_json = copy.deepcopy(in_json)
    train_json["frames"] = []
    all_json = copy.deepcopy(in_json)
    all_json["frames"] = []
    test_json = copy.deepcopy(in_json)
    test_json["frames"] = []

    for frame_metat in in_json["frames"]:
        frame_meta = copy.deepcopy(frame_metat)
        file_path = frame_meta["file_path"]
        file_path = os.path.basename(frame_meta["file_path"])
        frame_meta["file_path"] = f"images/{file_path}"
        
        all_json["frames"].append(frame_meta)
        
        index = int(os.path.splitext(file_path)[0])

        if index % 10:
            train_json["frames"].append(frame_meta)
        else:
            test_json["frames"].append(frame_meta)

    ## nerf_synthetic style
    with open(f"{workspace}/transforms_train.json", "w") as f:
        json.dump(train_json, f, indent=2)
    with open(f"{workspace}/transforms_test.json", "w") as f:
        json.dump(test_json, f, indent=2)
    with open(f"{workspace}/transforms_val.json", "w") as f:
        json.dump(test_json, f, indent=2)
