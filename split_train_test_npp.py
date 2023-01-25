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
    os.makedirs(f"{workspace}/train", exist_ok=True)
    os.makedirs(f"{workspace}/test", exist_ok=True)
    os.makedirs(f"{workspace}/train/rgb", exist_ok=True)
    os.makedirs(f"{workspace}/test/rgb", exist_ok=True)
    os.makedirs(f"{workspace}/train/intrinsics", exist_ok=True)
    os.makedirs(f"{workspace}/test/intrinsics", exist_ok=True)
    os.makedirs(f"{workspace}/train/pose", exist_ok=True)
    os.makedirs(f"{workspace}/test/pose", exist_ok=True)

    with open(f"{workspace}/posed_images/kai_cameras_normalized.json", 'r') as f:
        in_json = json.loads(f.read())

    for file_path in in_json:
        frame_meta = in_json[file_path]
        
        base_name = os.path.splitext(file_path)[0]
        index = int(base_name)

        if index % 10:
            split = "train"
        else:
            split = "test"
        
        shutil.copy(f"{workspace}/images/{file_path}", f"{workspace}/{split}/rgb/{file_path}")
        with open(f"{workspace}/{split}/intrinsics/{base_name}.txt", 'w') as f:
            f.write(" ".join([str(v) for v in frame_meta["K"]]))
        with open(f"{workspace}/{split}/pose/{base_name}.txt", 'w') as f:
            f.write(" ".join([str(v) for v in frame_meta["W2C"]]))
            