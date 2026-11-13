"""Early prototype: YOLO person heatmap over a whole video file (crowd_monitor.py replaces it).

    python camera_index_test.py path/to/video.mp4
"""
import sys

import cv2
import numpy as np
from ultralytics import YOLO

if len(sys.argv) != 2:
    sys.exit("usage: python camera_index_test.py VIDEO")
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"can't open {video_path}")

model = YOLO("yolov8n.pt")  # lightweight and fast

heatmap = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))

    if heatmap is None:
        heatmap = np.zeros((frame_resized.shape[0], frame_resized.shape[1]), dtype=np.float32)

    results = model(frame_resized, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # add intensity around center
            cv2.circle(heatmap, (cx, cy), 15, 1, -1)

cap.release()

heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = heatmap.astype(np.uint8)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite("crowd_density_heatmap.png", heatmap_color)
print("Saved: crowd_density_heatmap.png")
