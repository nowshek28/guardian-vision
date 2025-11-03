# guardian-vision
Real-time CCTV Pose & Threat Detection for Cashier Safety (POC)

---

# Guardian Vision

**Guardian Vision** is an AI-powered surveillance assistant designed to enhance safety and security in public and private spaces. It leverages video feeds to detect suspicious activities, identify potential threats, and trigger timely alerts to ensure quick human intervention.

---

## 🌟 Key Applications

- **Retail & Cashier Safety**
  - Monitor cashier–customer interactions.
  - Detect dangerous items in customer hands (e.g., firearms).
  - Identify face coverings or suspicious gestures.
  - Alert when the cashier shows signs of distress.

- **Public Transport (Railway Stations & Subways)**
  - Detect people falling on the tracks or platform edges.
  - Raise quick alerts to staff for immediate rescue actions.

- **Museums & Cultural Sites**
  - Alert guards if someone behaves suspiciously near valuable exhibits.
  - Provide early warnings for object tampering or unsafe crowd movement.

- **Crowded Public Spaces**
  - Spot accidents like falls in congested areas.
  - Track unusual movements or sudden risks in gatherings and events.

---

## ⚙️ System Workflow

### 1️⃣ Frame Capture & Pose Estimation
- The video is processed using YOLOv8-Pose to detect people, customers, and cashiers.
- The model identifies left and right hand keypoints (wrists) for every person in the frame.
- Region zones (cashier, customer, table) are predefined for role detection.

---

### 2️⃣ Hand Cropping & Dataset Creation
- From each frame, hand regions are cropped using wrist coordinates and padding (20–50%) to include surrounding context.
- Each cropped image is saved into a dataset folder structure for classification.
- The dataset is organized as:
  Dataset/
    train/
      Danger/
      NoDanger/
    val/
      Danger/
      NoDanger/
- This dataset represents two classes:
  - Danger – hands holding firearms or threatening items.
  - NoDanger – normal or safe hand actions.

---

### 3️⃣ Training & Validation of the Firearm Classification Model
- The classification model is trained using YOLOv8-Classification on CPU for accessibility.
- Training command example:
  yolo classify train model=yolov8n-cls.pt data=Dataset imgsz=224 epochs=15 batch=32 device=cpu
- The model learns to differentiate between Danger and NoDanger classes.
- Validation ensures balanced accuracy and performance across lighting and angle variations.
- The model can be exported in ONNX or OpenVINO format for faster CPU inference.

---

### 4️⃣ Real-time Classification & Integration
- Each hand ROI (Region of Interest) detected by YOLOv8-Pose is passed to the trained classifier.
- The classifier outputs:
  - Danger → Firearm detected in hand.
  - NoDanger → Normal hand or harmless object.
- The output label and confidence score are displayed directly on the bounding box in the live video.
- Classification is performed every few frames for CPU optimization.
- Temporal smoothing and confidence thresholds reduce flicker and false alerts.

---

### 5️⃣ Output Generation
- The output video displays:
  - Cashier, customer, and table zones.
  - Wrist tracking trails.
  - Real-time classification results (Danger / NoDanger) with probabilities.
- The system can optionally trigger alerts or save incident frames when a dangerous object is detected.

---

## 🚨 How Guardian Vision Helps

- Real-time Monitoring: Continuous video stream analysis for early detection.
- Early Warnings: Alerts triggered before a threat escalates.
- Incident Evidence: Key frames captured automatically for later review.
- Flexible Deployment: Works across retail, transport, museums, and public areas.

---

## 🧩 Technologies Used

| Component | Technology |
|----------|------------|
| Pose & Detection | YOLOv8-Pose |
| Classification | YOLOv8-Classification |
| Programming | Python, OpenCV, NumPy |
| Dataset Management | Roboflow |
| Hardware | AMD Ryzen CPU (CPU-only inference) |
| Model Export | ONNX / OpenVINO |

---

## 📊 Results & Observations

- The classification model performed strongly on diverse datasets with minor frame defects due to variable lighting and object shape.
- The system achieved real-time inference on CPU by limiting classification to every few frames.
- Successfully identifies and labels dangerous vs. safe hand actions in live CCTV footage.

---

## 🧭 Next Steps

- Expand the dataset with hard negatives (phones, tools, and remotes).
- Add multi-class threat detection (knives, sharp objects, etc.).
- Integrate alert systems for real-time notifications.
- Optimize further using OpenVINO acceleration for CPU speedup.
- Deploy across multiple environments (retail, transport, public safety).

---

## 🔒 Why It Matters

Guardian Vision is not just about surveillance—it’s about proactive safety.
By detecting risks early, it empowers security teams and frontline workers to respond faster, protect lives, and reduce harm before it happens.
