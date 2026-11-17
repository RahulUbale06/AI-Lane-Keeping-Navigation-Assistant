
from ultralytics import YOLO
import cv2


class ObjectDetector:

    def __init__(self):
        # Load lightweight YOLO model
        self.model = YOLO("yolov8n.pt")

        # COCO class IDs
        self.target_classes = {
            0: "Person",
            1: "Bicycle",
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    def detect(self, frame):

        # Run object detection
        results = self.model.predict(
            source=frame,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        annotated_frame = frame.copy()
        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in self.target_classes:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                object_name = self.target_classes[class_id]

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw object label
                label = f"{object_name}: {confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                detections.append({
                    "name": object_name,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

        return annotated_frame, detections
