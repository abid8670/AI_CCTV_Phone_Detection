import cv2
import time
from ultralytics import YOLO
from threading import Thread

# ===============================
# RTSP STREAM
# ===============================
RTSP_URL = "rtsp://admin:admin%40123@192.168.53.55:554/cam/realmonitor?channel=1&subtype=1"

# ===============================
# YOLO MODEL
# ===============================
model = YOLO("yolov8n.pt")
print("✅ YOLO nano model loaded (CPU)")

# ===============================
# MULTI-THREAD FRAME CAPTURE
# ===============================
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera stream")
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cap = VideoStream(RTSP_URL)

alert_timer = None
frame_count = 0
print("✅ AI CCTV started... Press ESC to exit")

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame receive nahi ho raha")
        break

    frame_count += 1

    # Only run detection every 3 frames (CPU-friendly)
    if frame_count % 3 == 0:
        # Resize for CPU speed
        small_frame = cv2.resize(frame, (640, 360))
        results = model(small_frame, conf=0.2, verbose=False)

        person_detected = False
        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale back to original frame
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                if label == "person":
                    person_detected = True
                    color = (0, 255, 0)
                elif label == "cell phone":
                    phone_detected = True
                    color = (0, 0, 255)
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # ALERT LOGIC
        if person_detected and phone_detected:
            if alert_timer is None:
                alert_timer = time.time()
            elif time.time() - alert_timer >= 3:
                cv2.putText(
                    frame,
                    "🚨 PHONE USAGE DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )
                print("🚨 ALERT: Phone usage detected")
                alert_timer = None
        else:
            alert_timer = None

    # SHOW LIVE STREAM
    cv2.imshow("AI CCTV - Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
