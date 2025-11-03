import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque


trail = deque(maxlen=100)   # keeps the last N centers
current_cashier = None      # to reset trail if cashier changes


# ---------- Define the imaginary box (polygon coordinates) ----------
TABLEZONE = [(830, 0), (1150, 0), (1450, 1080), (750, 1080)]
CASHIERZONE = [(0, 0), (830, 0), (750, 1080), (0, 1080)]
CUSTOMERZONE = [(1150, 0), (1920, 0), (1920, 1080), (1450, 1080)]

# ---------- Load model ----------
input_path = os.path.join(".", "Test_video.mp4")
model = YOLO("yolov8n-pose.pt")

#--------------Firearm Danger checck model loading --------------
# Load your trained classifier (use .pt or exported .onnx)
cls_model = YOLO("runs/classify/train/weights/best.pt")  # or "best.onnx"

IMG_SIZE = 224  # same as training

def classify_hand_crop(bgr_crop: np.ndarray):
    """
    Returns (label, prob). Assumes BGR crop from OpenCV.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None, 0.0
    # YOLOv8 classify can take numpy images directly
    r = cls_model.predict(bgr_crop, imgsz=IMG_SIZE, device="cpu", verbose=False)
    # r[0].probs: class probabilities
    probs = r[0].probs.data.cpu().numpy()
    idx = int(np.argmax(probs))
    names = r[0].names  # e.g., {0:'Danger', 1:'NoDanger'}
    return names[idx], float(probs[idx])

# --- Video writer setup ---
cap_info = cv2.VideoCapture(input_path)
src_fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
cap_info.release()

# If you use vid_stride=2 and write every processed frame,
# choose ONE of these depending on how you want playback:

playback_fps = src_fps / 2.0   # A) time-accurate (reflects skipped frames)
# playback_fps = src_fps       # B) same FPS as source (video looks “sped up”)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # or "avc1" if supported
out_path = os.path.join(".", "output_annotated.mp4")
out = None  # will init after we see the first frame (to get width/height)

Output_Path = "C:/Users/nowsh/AppData/Roaming/ABHISHEKS PROJECT/guardian-vision/Crop_images/"
FRame = 0
# ---------- Cashier registration state ----------
FRAMES_TO_LOCK = 15     # consecutive frames in zone to lock cashier
CONF_MIN = 0.45         # min detection confidence
MISSING_GRACE = 60      # frames to wait before unlocking if cashier disappears

cashier_id = None
in_zone_count = {}      # tid -> consecutive frames inside CASHIERZONE
missing_frames = 0

#--------------Coustmer registration state --------------
HAND_K = 0.3      # hand box size factor relative to person height (0.12–0.18 works)
KP_CONF_MIN = 0.40  # minimum wrist keypoint confidence

# ---------- Run YOLO tracking ----------
results = model.track(
    source=input_path,
    stream=True,
    show=False,
    save=False,
    imgsz=960,
    vid_stride=2,
    conf=0.25,
    persist=True,
    device="cpu"
)

# ---------- Loop through frames ----------
for res in results:
    #frame = res.plot()  # YOLO annotated frame
    frame = res.orig_img.copy()   # clean frame, no YOLO annotations

    # Draw zones
    cv2.polylines(frame, [np.array(TABLEZONE,    np.int32).reshape((-1,1,2))], True, (0, 0,255), 3)
    cv2.polylines(frame, [np.array(CASHIERZONE,  np.int32).reshape((-1,1,2))], True, (0,255, 0), 2)
    cv2.polylines(frame, [np.array(CUSTOMERZONE, np.int32).reshape((-1,1,2))], True, (255,0, 0), 2)

    # ---------- Cashier registration logic ----------
    if res.boxes is not None and res.boxes.id is not None:
        ids  = res.boxes.id.int().tolist()
        boxes = res.boxes.xyxy.tolist()
        confs = res.boxes.conf.tolist()

        # update consecutive-in-zone counts, try to lock if none
        for tid, box, conf in zip(ids, boxes, confs):
            if conf < CONF_MIN:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            inside = cv2.pointPolygonTest(np.array(CASHIERZONE, np.int32), (cx, cy), False) >= 0
            in_zone_count[tid] = (in_zone_count.get(tid, 0) + 1) if inside else 0

            if cashier_id is None and in_zone_count[tid] >= FRAMES_TO_LOCK:
                cashier_id = tid
                missing_frames = 0
                # print(f"Cashier registered: ID {cashier_id}")

        # maintain/clear lock
        if cashier_id is not None:
            if cashier_id in ids:
                missing_frames = 0
            else:
                missing_frames += 1
                if missing_frames > MISSING_GRACE:
                    # print("Cashier left — unlocking")
                    cashier_id = None
                    missing_frames = 0

        # (optional) highlight cashier box
        if cashier_id is not None:
            for tid, box in zip(ids, boxes):
                if tid == cashier_id:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center coordinates
                     # reset trail if cashier changed
                    if current_cashier != cashier_id:
                        trail.clear()
                        current_cashier = cashier_id

                    # add current center to trail
                    trail.append((cx, cy))

                    # draw trail (polyline)
                    if len(trail) > 1:
                        pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

                    # draw current point and box/label
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 3)
                    cv2.putText(frame, f"Cashier ID: {cashier_id}", (x1, max(30, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                    break

                # ---- Customer wrist boxes (magenta) ----
            # Only if pose keypoints are available
            if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
                kp_xy = res.keypoints.xy                      # shape: (N, K, 2)
                kp_conf = getattr(res.keypoints, "conf", None)  # shape: (N, K) or None

                # COCO-17 indices for YOLOv8-pose wrists
                L_WRIST, R_WRIST = 9, 10

                for i, (tid, box) in enumerate(zip(ids, boxes)):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # consider only people in CUSTOMER zone
                    in_customer = cv2.pointPolygonTest(np.array(CUSTOMERZONE, np.int32), (cx, cy), False) >= 0
                    if not in_customer:
                        continue

                    h = max(1.0, (y2 - y1))
                    half = int(max(8, HAND_K * h / 2.0))  # half-size of the square around wrist

                    for w_idx in (L_WRIST, R_WRIST):
                        wx, wy = float(kp_xy[i, w_idx, 0]), float(kp_xy[i, w_idx, 1])

                        # keypoint confidence gate (if available)
                        if kp_conf is not None and float(kp_conf[i, w_idx]) < KP_CONF_MIN:
                            continue

                        if wx <= 0 or wy <= 0:
                            continue  # invalid wrist

                        # clamp the box to frame bounds
                        xA = max(0, int(wx - half))
                        yA = max(0, int(wy - half))
                        xB = min(frame.shape[1] - 1, int(wx + half))
                        yB = min(frame.shape[0] - 1, int(wy + half))

                        # draw wrist box and dot (magenta)
                        cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 255), 2)
                        #save_path = os.path.join(Output_Path, f"Frame_{FRame}_ID_{tid}_Wrist_{'L' if w_idx == L_WRIST else 'R'}.jpg")
                        #FRame += 1
                        #cv2.imwrite(save_path, frame[yA:yB, xA:xB])
                        label , prob = classify_hand_crop(frame[yA:yB, xA:xB])
                        if label == "Danger" and prob > 0.7:
                            cv2.putText(frame, f"!! {label} {prob:.2f} !!", (xA, max(45, yA - 25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, f"{label} {prob:.2f}", (xA, max(45, yA - 25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, (int(wx), int(wy)), 3, (255, 0, 255), -1)

                        # small label
                        hand_label = "L-hand" if w_idx == L_WRIST else "R-hand"
                        #cv2.putText(frame, hand_label, (xA, max(15, yA - 6)),
                                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # lazy-init writer with the first frame's size
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(out_path, fourcc, playback_fps, (w, h))

    out.write(frame)  # write annotated frame

    # ---------- Display the frame ----------
    cv2.imshow("Guardian-Vision (Box + Cashier Lock)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if out is not None:
    out.release()
print(f"✅ Saved video to: {os.path.abspath(out_path)}")

cv2.destroyAllWindows()
print("✅ Box + cashier registration running.")
