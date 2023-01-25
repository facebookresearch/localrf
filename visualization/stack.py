import os

preset="fast"
crt = 28

scenes = [
    # "university1", 
    # "university2", 
    # "hike2", 
    # "playground", 
    "hike1",
    ]
for scene in scenes:
    orientation = "h" if scene == "university1" else "v"
    frame_crop = " -frames:v 900" if scene == "hike1" else ""
    # os.system(f"ffmpeg -y -r 30 -i ../../Downloads/{scene}_p.mp4 -crf {crt} -preset {preset} -c:v libx264 -pix_fmt yuv420p {frame_crop} ../../Downloads/{scene}_p1.mp4")
    os.system(f"ffmpeg -y -r 30 -i data/2831/videos/{scene}_input.mp4 -i data/2831/videos/{scene}_ours.mp4 -i ../../Downloads/{scene}_p.mp4 -crf {crt} -preset {preset}  -c:v libx264 -pix_fmt yuv420p {frame_crop} -filter_complex {orientation}stack=inputs=3 ../../Downloads/{scene}_stacked.mp4")

