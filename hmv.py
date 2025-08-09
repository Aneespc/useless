import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- Setup Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# --- Setup volume control ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Calibrated max mouth open distance in pixels 
MAX_OPEN_DIST = 50

def mouth_open_percentage(landmarks, img_h):
    # MediaPipe Face Mesh landmarks for upper and lower inner lips
    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]

    upper_y = int(upper_lip.y * img_h)
    lower_y = int(lower_lip.y * img_h)

    open_dist = lower_y - upper_y
    percentage = (open_dist / MAX_OPEN_DIST) * 100
    percentage = min(max(percentage, 0), 100)  # Clamp between 0 and 100
    return percentage

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    vol_percentage = 0
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            vol_percentage = mouth_open_percentage(face_landmarks, h)

            # Map percentage (0-100) to system volume scalar (0.0-1.0)
            vol_scalar = vol_percentage / 100.0
            volume.SetMasterVolumeLevelScalar(vol_scalar, None)

            # Draw volume bar
            bar_height = int((vol_percentage / 100) * 300)
            cv2.rectangle(frame, (50, 400 - bar_height), (100, 400), (0, 255, 0), -1)
            cv2.putText(frame, f"Volume: {int(vol_percentage)}%", (50, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Optionally, draw lips points for visualization
            upper_lip_coords = (int(face_landmarks.landmark[13].x * w), int(face_landmarks.landmark[13].y * h))
