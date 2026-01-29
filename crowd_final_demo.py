from inference import get_model
import cv2
import numpy as np
import time
from collections import deque

# ---------------- SETTINGS ----------------
API_KEY = "PASTE_YOUR_PRIVATE_API_KEY_HERE"
MODEL_ID = "crowd-detection-n30gw-isumb/2"

CAM_INDEX = 1          # try 0 / 1 / 2
CONF_TH = 0.40

HEAT_DECAY = 0.93      # 0.90 fast fade, 0.97 slow fade
HEAT_ADD = 25          # heat intensity per detection
HEAT_RADIUS = 30       # blob size

LOW_TH = 5             # <= 5 people -> GREEN
MID_TH = 12            # 6-12 -> YELLOW
# > 12 -> RED

REDIRECT_MARGIN = 2    # difference needed to redirect

COOLDOWN = 2.0         # seconds (decision won't change quickly)
# ------------------------------------------

print("Loading Roboflow model...")
model = get_model(model_id=MODEL_ID, api_key=API_KEY)
print("✅ Model loaded successfully!")

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ Camera not opened. Try CAM_INDEX = 0 or 2")
    exit()

heatmap = None

# smoothing people count
people_hist = deque(maxlen=10)

# cooldown decision lock
last_decision = "HOLD / BALANCED"
last_time = time.time()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # fade old heat
    heatmap *= HEAT_DECAY

    # inference
    result = model.infer(frame)

    # sometimes returns list
    if isinstance(result, list):
        result = result[0]

    preds = []
    if hasattr(result, "dict"):
        preds = result.dict().get("predictions", [])

    people = 0
    left = 0
    right = 0
    mid_x = w // 2

    for p in preds:
        if p.get("confidence", 0) < CONF_TH:
            continue

        people += 1

        x = int(p["x"])
        y = int(p["y"])
        bw = int(p["width"])
        bh = int(p["height"])

        # left/right split
        if x < mid_x:
            left += 1
        else:
            right += 1

        # bbox draw
        x1, y1 = x - bw // 2, y - bh // 2
        x2, y2 = x + bw // 2, y + bh // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # heat blob
        cv2.circle(heatmap, (x, y), HEAT_RADIUS, HEAT_ADD, -1)

    # smooth people count
    people_hist.append(people)
    people_avg = int(sum(people_hist) / len(people_hist))

    # heatmap smoothing
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), 7)

    # normalize heatmap
    heat_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heat_norm = heat_norm.astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    output = cv2.addWeighted(frame, 0.55, heat_color, 0.45, 0)

    # divider line
    cv2.line(output, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    # ---------------- Density Classification ----------------
    if people_avg <= LOW_TH:
        density = "LOW"
        signal = "GREEN"
        box_color = (0, 255, 0)
    elif people_avg <= MID_TH:
        density = "MEDIUM"
        signal = "YELLOW"
        box_color = (0, 255, 255)
    else:
        density = "HIGH"
        signal = "RED"
        box_color = (0, 0, 255)

    # ---------------- Redirection Decision (with cooldown) ----------------
    new_decision = "HOLD / BALANCED"

    if left > right + REDIRECT_MARGIN:
        new_decision = "REDIRECT -> RIGHT"
    elif right > left + REDIRECT_MARGIN:
        new_decision = "REDIRECT -> LEFT"

    # apply cooldown lock
    if time.time() - last_time > COOLDOWN:
        last_decision = new_decision
        last_time = time.time()

    # ---------------- Arrow Overlay ----------------
    if last_decision == "REDIRECT -> LEFT":
        cv2.arrowedLine(output, (w - 120, h // 2), (120, h // 2), (0, 0, 255), 8, tipLength=0.3)
    elif last_decision == "REDIRECT -> RIGHT":
        cv2.arrowedLine(output, (120, h // 2), (w - 120, h // 2), (0, 0, 255), 8, tipLength=0.3)
    else:
        cv2.putText(output, "HOLD", (mid_x - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # ---------------- FPS ----------------
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    # ---------------- UI Text ----------------
    cv2.putText(output, f"People: {people_avg}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(output, f"Left: {left}  Right: {right}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(output, f"Decision: {last_decision}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(output, f"Decision: {last_decision}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(output, f"Density: {density}", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)

    cv2.putText(output, f"Signal: {signal}", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)

    cv2.putText(output, f"FPS: {fps:.1f}", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Crowd Heatmap + Redirection (Final Demo)", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
