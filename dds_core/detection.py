import cv2
import dlib
from imutils import face_utils
import numpy as np

class DrowsinessDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.counter = 0
        self.status = "Active"

    def eye_aspect_ratio(self, eye):
        # Compute Eye Aspect Ratio (EAR)
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[36:42]
            rightEye = shape[42:48]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Visualize eyes
            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

            if ear < self.EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.status = "SLEEPING !!!"
                    cv2.putText(frame, self.status, (150, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                self.counter = 0
                self.status = "Active"
                cv2.putText(frame, self.status, (150, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return frame, self.status
