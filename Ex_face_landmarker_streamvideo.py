import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import time
import cv2
import mediapipe as mp
import numpy as np

MP_TASK_FILE = "face_landmarker.task"

class FaceMeshDetector:

    def __init__(self):
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mp_callback)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)

        self.landmarks = None
        self.blendshapes = None
        self.latest_time_ms = 0

    def mp_callback(self, mp_result, output_image, timestamp_ms: int):
        if len(mp_result.face_landmarks) >= 1 and len(mp_result.face_blendshapes) >= 1:

            self.landmarks = mp_result.face_landmarks
            self.blendshapes = [b.score for b in mp_result.face_blendshapes[0]]

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            return

        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.model.detect_async(frame_mp, t_ms)
        self.latest_time_ms = t_ms

    def get_results(self):
        return self.landmarks, self.blendshapes

    def draw_landmarks_on_image(self,rgb_image, landmarks):
      face_landmarks_list = landmarks
      annotated_image = np.copy(rgb_image)

      # Loop through the detected faces to visualize.
      for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        ##人脸的关键点连线
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        #眉毛/嘴巴/眼睛/脸部轮廓的识别及连线
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        #人眼睛的识别及连线
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())

      return annotated_image


def main():
    facemesh_detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facemesh_detector.update(frame)
        landmarks, blendshapes = facemesh_detector.get_results()
        print(landmarks)
        if (landmarks is None) or (blendshapes is None):
            continue
        annotated_image = facemesh_detector.draw_landmarks_on_image(imgrgb, landmarks)
        cv2.imshow("annotated", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    exit()

main()