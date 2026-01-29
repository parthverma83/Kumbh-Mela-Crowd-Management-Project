from inference import get_model
import cv2
import numpy as np

# ----------- SETTINGS -----------
API_KEY = "PASTE_YOUR_PRIVATE_API_KEY_HERE"
MODEL_ID = "crowd-detection-n30gw-isumb/2"

CAM_INDEX = 1   # Try 0, 1, or 2
CONF_TH = 0.40

HEAT_DECAY = 0.95   # heat fades speed
HEAT_ADD = 30       # heat added per person detection
ALERT_COUNT = 12    # crowd threshold
# --------------------------------

print("Loading Roboflow model...")
try:
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    print("Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"❌ Camera at index {CAM_INDEX} not opened. Try CAM_INDEX = 0 or 2")
    exit()

heatmap = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # Initialize heatmap based on frame size
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # Decay old heat
    heatmap *= HEAT_DECAY

    # Inference - using the .infer() method
    results = model.infer(frame)
    
    # Handle both list and object responses
    if isinstance(results, list):
        preds = results[0].predictions if hasattr(results[0], 'predictions') else results[0]["predictions"]
    else:
        preds = results.predictions if hasattr(results, 'predictions') else results["predictions"]

    people = 0
    left = 0
    right = 0
    mid_x = w // 2

    # Draw divider line
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    for p in preds:
        # Depending on SDK version, p might be a dict or an object
        conf = p.confidence if hasattr(p, 'confidence') else p["confidence"]
        if conf < CONF_TH:
            continue

        people += 1
        
        # Get coordinates
        px = int(p.x if hasattr(p, 'x') else p["x"])
        py = int(p.y if hasattr(p, 'y') else p["y"])
        pw = int(p.width if hasattr(p, 'width') else p["width"])
        ph = int(p.height if hasattr(p, 'height') else p["height"])

        # left/right count logic
        if px < mid_x:
            left += 1
        else:
            right += 1

        # Calculate bounding box corners
        x1, y1 = int(px - pw // 2), int(py - ph // 2)
        x2, y2 = int(px + pw // 2), int(py + ph // 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add heat at the center point (ensure within bounds)
        if 0 <= px < w and 0 <= py < h:
            heatmap[py, px] += HEAT_ADD

    # Heatmap Processing
    heat_norm = np.clip(heatmap, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    
    # Merge heatmap and original frame
    output = cv2.addWeighted(frame, 0.65, heat_color, 0.35, 0)

    # Redirection Decision Logic
    if people >= ALERT_COUNT:
        if left > right + 2:
            decision = "REDIRECT -> RIGHT"
        elif right > left + 2:
            decision = "REDIRECT -> LEFT"
        else:
            decision = "HOLD / BALANCED"
    else:
        decision = "NORMAL FLOW"

    # UI Overlay
    cv2.putText(output, f"People: {people}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, f"Left: {left}  Right: {right}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(output, f"Decision: {decision}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(output, f"Decision: {decision}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Crowd Heatmap + Redirection", output)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()