import cv2
import mediapipe as mp
from time import time

cap = cv2.VideoCapture(0)
face_detection = mp.solutions.face_detection
draw = mp.solutions.drawing_utils

detector = face_detection.FaceDetection(0.75)
p_time = 0


def put_fps():
    global p_time

    c_time = time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


def put_detection_score(detection):
    cv2.putText(img, str('{:.2f}% face').format(detection.score[0]), (190, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)


while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.process(img_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            draw.draw_detection(img, detection)

            put_detection_score(detection)

            print(id, detection)

    put_fps()

    cv2.imshow('Face-reconition', img)
    cv2.waitKey(1)
