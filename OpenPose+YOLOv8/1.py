import cv2
import mediapipe as mp
import time
import numpy as np
from ultralytics import YOLO

class HandObjectTracking:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize Mediapipe Hand Tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        # Load YOLOv8 Model for Segmentation-based Object Detection
        self.model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentation model

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  

        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def detectObjects(self, frame):
        results = self.model(frame)

        for r in results:
            for box, mask in zip(r.boxes, r.masks.xy):  # Use segmentation mask
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Object bounds
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{self.model.names[cls]} {conf:.2f}"

                # Create a mask to highlight the detected object
                mask_img = np.zeros_like(frame, dtype=np.uint8)
                pts = np.array(mask, dtype=np.int32)  # Convert mask to polygon
                cv2.fillPoly(mask_img, [pts], (0, 255, 0))  # Fill detected object

                # Blend the mask with the original frame
                frame = cv2.addWeighted(frame, 1, mask_img, 0.5, 0)

                # Calculate label position (center top of the object)
                text_x = x1 + 10
                text_y = max(y1 - 10, 20)

                # Draw label on top of the detected object
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return frame

    def main(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Faster capture

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        ptime = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.findFingers(frame)
            frame = self.detectObjects(frame)  # Run YOLOv8 object detection

            # Calculate FPS
            ctime = time.time()
            fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
            ptime = ctime

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv2.imshow("Hand & Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = HandObjectTracking()
    tracker.main()
