import cv2
import numpy as np

folder = "/mnt/datassd/ameuleman/logs_eval/FuSta/fusta"

cap = cv2.VideoCapture("/mnt/datassd/ameuleman/logs_eval/FuSta/hike2.mp4")

slice_fusta = []
slice_ours = []

for i in range(93):
    fusta = cv2.imread(f"{folder}/{i+1:05d}.png")
    fusta = cv2.resize(fusta, None, fx=0.5, fy=0.5)
    slice_fusta.append(fusta[:, 250])
    
    ours = cap.read()[1]
    ours = cv2.resize(ours, fusta.shape[1::-1])
    slice_ours.append(ours[:, 250])

    fusta = cv2.resize(fusta, None, fx=0.25, fy=0.25)
    ours = cv2.resize(ours, fusta.shape[1::-1])
    frame = np.hstack([fusta, ours])
    if i >= 5:
        cv2.imwrite(f"{folder}_comparison/{i-5}.jpg", frame)
    
slice_fusta = np.stack(slice_fusta, axis=1)
slice_ours = np.stack(slice_ours, axis=1)

slice_fusta = cv2.resize(slice_fusta, (slice_fusta.shape[0], slice_fusta.shape[0]), interpolation=cv2.INTER_NEAREST)
slice_ours = cv2.resize(slice_ours, (slice_fusta.shape[0], slice_fusta.shape[0]), interpolation=cv2.INTER_NEAREST)

cv2.imwrite(f"{folder}_comparison/slice_fusta.jpg", slice_fusta)
cv2.imwrite(f"{folder}_comparison/slice_ours.jpg", slice_ours)
