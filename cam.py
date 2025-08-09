
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Audio volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Open video file instead of webcam
#video_path = 'input_video.mp4'  # <-- Replace with your video file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error opening video file {video_path}")
    exit()

# Get frame dimensions for output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Prepare VideoWriter to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Mouth landmark indices (outer lips)
mouth_landmark_ids = [61, 291, 78, 308, 13, 14, 17, 0]  # Some key mouth points

def mouth_open_ratio(landmarks, img_w, img_h):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    upper_y = int(upper_lip.y * img_h)
    lower_y = int(lower_lip.y * img_h)

    open_dist = lower_y - upper_y
    return open_dist

MAX_OPEN_DIST = 50  # Adjust according to your calibration

def mouth_open_percentage(landmarks, img_h):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    upper_y = int(upper_lip.y * img_h)
    lower_y = int(lower_lip.y * img_h)

    open_dist = lower_y - upper_y
    percentage = (open_dist / MAX_OPEN_DIST) * 100
    percentage = min(max(percentage, 0), 100)
    return percentage

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Flip frame if needed
    # frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Get mouth landmark coordinates in pixels
        mouth_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in mouth_landmark_ids]

        # Bounding box around mouth
        x_vals = [p[0] for p in mouth_points]
        y_vals = [p[1] for p in mouth_points]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # Draw mouth rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # Mouth open distance
        open_dist = mouth_open_ratio(face_landmarks, w, h)
        mouth_height = y_max - y_min
        ratio = open_dist / mouth_height if mouth_height != 0 else 0
        ratio = np.clip(ratio, 0, 1)

        # Set system volume based on ratio (optional, will affect your system)
        volume.SetMasterVolumeLevelScalar(ratio, None)

        # Draw volume bar on frame
        bar_height = int(ratio * 300)
        cv2.rectangle(frame, (50, 400 - bar_height), (100, 400), (0, 255, 0), -1)
        cv2.putText(frame, f"Mouth Open: {int(ratio * 100)}%", (50, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Debug prints in console
        open_percentage = mouth_open_percentage(face_landmarks, h)
        print(f"Mouth open ratio: {ratio:.2f}, Mouth Open Percentage: {open_percentage:.1f}%")

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Mouth Volume Control from Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
out.release()
cv2.destroyAllWindows()
