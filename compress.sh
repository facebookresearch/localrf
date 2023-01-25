PRESET=veryfast

CRT=32
METHODS=(input)
for IDX in {0..0}
do
    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/hike1.mp4 -crf $CRT -preset $PRESET -frames:v 900 -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/hike1_${METHODS[$IDX]}.mp4

    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/university1.mp4 -crf $CRT -preset $PRESET -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website_full/videos/${METHODS[$IDX]}/university1tmp.mp4
    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/university1tmp.mp4 -crf $CRT -preset $PRESET -vf transpose=2 -c:v libx264 -pix_fmt yuv420p data/website/videos/university1_${METHODS[$IDX]}.mp4
    SCENES=(hike2 university2 hike2 playground)
    for SCENE_IDX in {0..3}
    do
        ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/${SCENES[$SCENE_IDX]}.mp4 -crf $CRT -preset $PRESET -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/${SCENES[$SCENE_IDX]}_${METHODS[$IDX]}.mp4
    done
done


CRT=28
METHODS=(mipnerf360 nerfacto ours no_prog no_loc)
for IDX in {0..4}
do
    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/hike1.mp4 -crf $CRT -preset $PRESET -frames:v 900 -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/hike1_${METHODS[$IDX]}.mp4

    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/university1.mp4 -crf $CRT -preset $PRESET -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website_full/videos/${METHODS[$IDX]}/university1tmp.mp4
    ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/university1tmp.mp4 -crf $CRT -preset $PRESET -vf transpose=2 -c:v libx264 -pix_fmt yuv420p data/website/videos/university1_${METHODS[$IDX]}.mp4
    SCENES=(hike2 university2 hike2 playground)
    for SCENE_IDX in {0..3}
    do
        ~/ffmpeg/ffmpeg -y -r 30 -i data/website_full/videos/${METHODS[$IDX]}/${SCENES[$SCENE_IDX]}.mp4 -crf $CRT -preset $PRESET -vf scale=1280:720 -c:v libx264 -pix_fmt yuv420p data/website/videos/${SCENES[$SCENE_IDX]}_${METHODS[$IDX]}.mp4
    done
done

# ffmpeg -framerate 30 -pattern_type glob -i 'hike_09_26_1/skip_0/render_smooth4/path_renders_step_200000/color_*.png' -c:v libx264 -pix_fmt yuv420p hike1.mp4
