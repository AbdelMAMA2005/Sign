import cv2
import mediapipe as mp
import numpy as np


class holisticDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.results = None

    # --------------------------------------------------
    def findHolistic(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(img_rgb)

        if draw and self.results:
            if self.results.face_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    self.results.face_landmarks,
                    self.mp_holistic.FACEMESH_TESSELATION
                )

            if self.results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    self.results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS
                )

            if self.results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    self.results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS
                )

        return img

    # --------------------------------------------------
    def extract_landmarks(self):
        """Retourne TOUJOURS un dictionnaire avec des shapes fixes"""
        if self.results is None:
            return None

        data = {
            "left_hand": np.zeros((21, 3)),
            "right_hand": np.zeros((21, 3)),
            "face": np.zeros((468, 3))
        }

        if self.results.left_hand_landmarks:
            data["left_hand"] = self._lm_to_array(
                self.results.left_hand_landmarks
            )

        if self.results.right_hand_landmarks:
            data["right_hand"] = self._lm_to_array(
                self.results.right_hand_landmarks
            )

        if self.results.face_landmarks:
            data["face"] = self._lm_to_array(
                self.results.face_landmarks
            )

        return data

    # --------------------------------------------------
    def _lm_to_array(self, landmark_list):
        """Convertit les landmarks Mediapipe en numpy (N, 3)"""
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark],
            dtype=np.float32
        )