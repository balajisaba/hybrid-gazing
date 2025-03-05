import cv2
import dlib
import numpy as np
import pyautogui
import mediapipe as mp
import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk
from pynput.mouse import Controller
from imutils import face_utils
from filterpy.kalman import KalmanFilter
import time

# Initialize modules
mouse = Controller()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.7,
                                  min_tracking_confidence=0.7)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# Screen properties
screen_w, screen_h = pyautogui.size()
cam_w, cam_h = 640, 480
unit_w, unit_h = screen_w / cam_w, screen_h / cam_h

# Kalman filter setup
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.P *= 1000
kf.R = np.eye(2) * 10
kf.Q = np.eye(4) * 0.1
kf.x = np.array([screen_w // 2, screen_h // 2, 0, 0])


# Adaptive Noise Reduction (EMA Smoothing)
def ema_filter(value, prev_value, alpha=0.3):
    return alpha * value + (1 - alpha) * prev_value


# Initialize GUI
root = tk.Tk()
root.title("Hybrid Face Tracking System")
root.geometry("800x600")
root.configure(bg="#2C2F33")

# GUI Elements
Label(root, text="Eye Tracking Cursor Control", font=("Arial", 16, "bold"), fg="white", bg="#2C2F33").pack(pady=10)
status_label = Label(root, text="Status: Not Running", font=("Arial", 12), fg="white", bg="#2C2F33")
status_label.pack()
preview_label = Label(root, bg="#2C2F33")
preview_label.pack(pady=10)

# Video Capture
global vid, running, last_fatigue_check, blink_count, start_time, fatigue_detected, prev_x, prev_y
vid = cv2.VideoCapture(0)
vid.set(3, cam_w)
vid.set(4, cam_h)
running = False
last_fatigue_check = time.time()
blink_count = 0
start_time = time.time()
fatigue_detected = False
prev_x, prev_y = screen_w // 2, screen_h // 2


# Gaze Estimation (Uses Dlib)
def get_gaze_ratio(eye_points, landmarks):
    left = landmarks.part(eye_points[0]).x
    right = landmarks.part(eye_points[3]).x
    top = (landmarks.part(eye_points[1]).y + landmarks.part(eye_points[2]).y) // 2
    bottom = (landmarks.part(eye_points[4]).y + landmarks.part(eye_points[5]).y) // 2
    return (right - left) / (bottom - top)


# Face Tracking
def track_face():
    global running, blink_count, prev_x, prev_y
    if not running:
        return
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hybrid Face Detection (MediaPipe + Dlib)
    results = face_mesh.process(rgb_frame)
    faces = detector(gray)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            eye_x, eye_y = int(face_landmarks.landmark[468].x * screen_w), int(
                face_landmarks.landmark[468].y * screen_h)

            # Adaptive Kalman Filtering
            kf.predict()
            kf.update([eye_x, eye_y])
            kalman_x, kalman_y = kf.x[:2]

            # Apply EMA for smoothing
            smooth_x = ema_filter(kalman_x, prev_x)
            smooth_y = ema_filter(kalman_y, prev_y)
            prev_x, prev_y = smooth_x, smooth_y

            # Move mouse
            mouse.position = (int(smooth_x), int(smooth_y))

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        avg_gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if avg_gaze_ratio < 0.9:
            blink_count += 1

    # GUI Preview Update
    frame = cv2.resize(frame, (200, 150))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    preview_label.configure(image=img)
    preview_label.image = img

    root.after(10, track_face)


# Fatigue Detection
def check_fatigue():
    global last_fatigue_check, blink_count, fatigue_detected
    elapsed_time = time.time() - last_fatigue_check
    if elapsed_time >= 30:
        if blink_count < 5 or blink_count > 30:
            fatigue_detected = True
            messagebox.showwarning("Fatigue Alert", "You seem fatigued! Take a break.")
            fatigue_detected = False
        blink_count = 0
        last_fatigue_check = time.time()
    root.after(10000, check_fatigue)  # Check every 10 seconds


# Start and Stop Functions
def start_tracking():
    global running
    running = True
    status_label.config(text="Status: Running", fg="green")
    track_face()
    check_fatigue()


def stop_tracking():
    global running
    running = False
    status_label.config(text="Status: Stopped", fg="red")


# Buttons
Button(root, text="Start Tracking", font=("Arial", 12, "bold"), bg="#43A047", fg="white", command=start_tracking).pack(
    pady=5)
Button(root, text="Stop Tracking", font=("Arial", 12, "bold"), bg="#D32F2F", fg="white", command=stop_tracking).pack(
    pady=5)
Button(root, text="Exit", font=("Arial", 12, "bold"), bg="#757575", fg="white", command=root.quit).pack(pady=20)

root.mainloop()
vid.release()
cv2.destroyAllWindows()
