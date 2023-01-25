import os

preset="slow"

def compress_batch(crt, methods):
    for method in methods:
        os.system(f"ffmpeg -y -r 30 -i data/website_full/videos/{method}/hike1.mp4 -crf {crt} -preset {preset} -frames:v 900 -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/hike1_{method}.mp4")

        os.system(f"ffmpeg -y -r 30 -i data/website_full/videos/{method}/university1.mp4 -crf {crt} -preset {preset} -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website_full/videos/{method}/university1tmp.mp4")
        os.system(f"ffmpeg -y -r 30 -i data/website_full/videos/{method}/university1tmp.mp4 -crf {crt} -preset {preset} -vf transpose=2 -c:v libx264 -pix_fmt yuv420p data/website/videos/university1_{method}.mp4")

        scenes = ["hike2", "university2", "hike2", "playground"]
        for scene in scenes:
            os.system(f"ffmpeg -y -r 30 -i data/website_full/videos/{method}/{scene}.mp4 -crf {crt} -preset {preset} -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/{scene}_{method}.mp4")

os.system(f"ffmpeg -y -r 60 -i data/prog_opt.mp4 -crf 26 -preset {preset} -c:v libx264 -pix_fmt yuv420p data/website/videos/prog_opt.mp4")
compress_batch(32, ["input"])
compress_batch(26, ["mipnerf360", "nerfacto", "ours", "no_prog", "no_loc"])
