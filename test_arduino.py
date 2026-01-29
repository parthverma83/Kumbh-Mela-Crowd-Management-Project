from inference import get_model
import cv2
import numpy as np
import serial
import time

# ---------------- SETTINGS ----------------
API_KEY = "Ce92tZce5xzX69O9Kx7p"
MODEL_ID = "crowd-detection-n30gw-isumb/2"

CAM_INDEX = 1        # try 0/1/2
CONF_TH = 0.40

HEAT_DECAY = 0.95
HEAT_ADD = 30

LOW_TH = 5           # <= 5 people => GREEN
MID_TH = 12          # 6 to 12 => YELLOW
# > 12 => RED

SERIAL_PORT = "COM7" # check Arduino port in IDE
BAUD = 9600
# ------------------------------------------

print("Loading Roboflow model...")
model = get_model(model_id=MODEL_ID, api_key=API_KEY)
print("Model loaded successfully!")

# Arduino serial connect
arduino = None
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)
    print("✅ Arduino connected!")
except:
    print("⚠️ Arduino not connected (Skipping LED control).")

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("❌ Camera not opened. Try CAM_INDEX = 0 or 2")
    exit()

heatmap = None
last_signal = None

def send_signal(sig):
    global last_signal
    if arduino is None:
        return
    if sig != last_signal:
        arduino.write(sig.encode())
        last_signal = sig

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # decay heat
    heatmap *= HEAT_DECAY

    # inference
    result = model.infer(frame)
    if isinstance(result, list):
        result = result[0]

    preds = []
    if hasattr(result, "dict"):
        preds = result.dict().get("predictions", [])

    people = 0
    left = 0
    right = 0
    mid_x = w // 2

    # divider line
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    for p in preds:
        if p.get("confidence", 0) < CONF_TH:
            continue

        people += 1
        x = int(p["x"])
        y = int(p["y"])
        bw = int(p["width"])
        bh = int(p["height"])

        # left/right count
        if x < mid_x:
            left += 1
        else:
            right += 1

        # bbox
        x1, y1 = x - bw // 2, y - bh // 2
        x2, y2 = x + bw // 2, y + bh // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # heat add
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] += HEAT_ADD

    # heatmap overlay
    heat_norm = np.clip(heatmap, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    output = cv2.addWeighted(frame, 0.65, heat_color, 0.35, 0)

    # ---------------- Density classification ----------------
    if people <= LOW_TH:
        density = "LOW"
        signal = "G"
    elif people <= MID_TH:
        density = "MEDIUM"
        signal = "Y"
    else:
        density = "HIGH"
        signal = "R"

    send_signal(signal)

    # ---------------- Crowd redirection decision ----------------
    if left > right + 2:
        decision = "REDIRECT -> RIGHT"
        arrow_dir = "RIGHT"
    elif right > left + 2:
        decision = "REDIRECT -> LEFT"
        arrow_dir = "LEFT"
    else:
        decision = "HOLD / BALANCED"
        arrow_dir = "HOLD"

    # ---------------- Draw arrows overlay ----------------
    if arrow_dir == "LEFT":
        cv2.arrowedLine(output, (w - 100, h // 2), (100, h // 2), (0, 0, 255), 8, tipLength=0.3)
    elif arrow_dir == "RIGHT":
        cv2.arrowedLine(output, (100, h // 2), (w - 100, h // 2), (0, 0, 255), 8, tipLength=0.3)
    else:
        cv2.putText(output, "HOLD", (mid_x - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # ---------------- Text overlay ----------------
    cv2.putText(output, f"People: {people}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(output, f"Left: {left}  Right: {right}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(output, f"Density: {density} ({signal})", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(output, f"Decision: {decision}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(output, f"Decision: {decision}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Crowd Heatmap + Arduino Signals + Redirection", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()
