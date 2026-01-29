Background & Motivation

Our initial implementation focused on deploying a YOLO-based object detection model on an IoT surveillance setup to count people in real-world environments.

While YOLO performs well in sparse to moderately crowded scenes, it showed critical limitations when applied to extremely dense crowds, such as religious gatherings, festivals, and mass events.

This project documents our engineering transition from traditional object detection to density map regression using CSRNet, and highlights the improvements achieved.

🧪 Phase 1: YOLO-based Crowd Counting on IoT
🔧 Approach

Model: YOLO (bounding-box based object detection)

Deployment: IoT camera / edge device

Output: Person count via detected bounding boxes

❌ Limitations Observed

When tested on high-density environments:

Severe occlusion between people

Extremely small visible head regions

Bounding boxes overlapped or disappeared

False negatives increased rapidly

Count accuracy degraded beyond usable limits

📉 Result:
YOLO consistently underestimated the crowd size and failed in scenarios with 500+ people per frame.

🔍 Key Insight

In extreme crowd density, individual detection is not feasible.
The problem must be reframed from “detecting people” to “estimating crowd density”.

This insight led us to explore Density Map Regression, which is specifically designed for high-occlusion scenarios.

🧠 Phase 2: Transition to CSRNet (Density Map Regression)
🔧 New Approach

Model: CSRNet (Convolutional Neural Network for Crowd Counting)

Core Idea:

Each person contributes to a continuous density map

No bounding boxes required

Output:

Density heatmap

Total crowd count = sum of density values

🚀 Why CSRNet Works Better Than YOLO
Aspect	YOLO (Object Detection)	CSRNet (Density Map)
Handles occlusion	❌ Poor	✅ Excellent
Works in dense crowds	❌ Fails	✅ Designed for it
Bounding boxes needed	✅ Yes	❌ No
Accuracy in >1000 people	❌ Very low	✅ High
Suitable for Kumbh Mela-like scenes	❌ No	✅ Yes
📊 Improvements Achieved

📈 Significant increase in count accuracy

🌡️ Reliable density heatmap visualization

🧩 Region-wise crowd density estimation

🧠 Robust performance under extreme occlusion

🔁 Scalable to real-world surveillance feeds

🏗️ System Architecture Evolution
YOLO-Based IoT Pipeline (Initial)
Camera → YOLO Detection → Bounding Boxes → Count

CSRNet-Based Pipeline (Improved)
Camera → Frame Extraction → CSRNet → Density Map → Count + Heatmap

📂 Datasets Used

To align with real-world dense crowd scenarios, we relied on datasets specifically designed for extreme density:

UCF-QNRF (primary inspiration)

ShanghaiTech Part A

NWPU-Crowd (optional benchmarking)

🎯 Real-World Impact

This transition enables:

Accurate crowd estimation in mass gatherings

Early identification of high-risk dense zones

Improved decision-making for crowd management and safety

Feasibility for smart city and public safety IoT deployments

🔮 Future Work

Optimize CSRNet for edge/IoT inference

Integrate real-time alerts for critical density thresholds

Multi-camera crowd fusion

Edge–cloud hybrid deployment

📝 Conclusion

By moving from object detection (YOLO) to density map regression (CSRNet), this project demonstrates a fundamental shift in problem formulation that enables accurate crowd estimation in environments where traditional methods fail.

This approach is scalable, robust, and suited for real-world extreme crowd scenarios.
