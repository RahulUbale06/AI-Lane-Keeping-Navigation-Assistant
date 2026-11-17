
import cv2
from yolo_detector import ObjectDetector

VIDEO_SOURCE = "videos/mixkit-cars-driving-by-on-road-2022-hd-ready.mp4"

detector = ObjectDetector()
camera = cv2.VideoCapture(VIDEO_SOURCE)

if not camera.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break

    annotated_frame, detections = detector.detect(frame)

    print("Detected objects:", detections)

    cv2.imshow("YOLO Object Detection - Version 7.0",
               annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
