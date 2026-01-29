import cv2
import numpy as np
from ultralytics import YOLO

video_path = r"C:\Users\vicky\Downloads\14422586_1920_1080_25fps.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Video not opening. Check path!")
    exit()

model = YOLO("yolov8n.pt")

W, H = 640, 360
heatmap = np.zeros((H, W), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(heatmap, (cx, cy), 15, 1, -1)

cap.release()

heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite("crowd_density_heatmap.png", heatmap_color)
print("✅ Saved: crowd_density_heatmap.png")
