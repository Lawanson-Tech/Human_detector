"""
face_detector.py
Face detection with logging + voice + sound alerts ONLY when a human face is detected.
Requirements: opencv-python, simpleaudio, pyttsx3
"""

import cv2
import simpleaudio as sa
import threading
import time
import os
import csv
import pyttsx3
from datetime import datetime

# ---------- Configuration ----------
CAMERA_INDEX = 0              # change if webcam index is different
ALERT_SOUND = "alert.wav"     # sound to play when a human face is detected
REPORT_CSV = "report.csv"     # log file
ALERT_COOLDOWN = 5            # seconds minimum between alerts
MIN_FACE_SIZE = (30, 30)      # smallest face size to detect
HUMAN_ALERT_DURATION = 30     # seconds to repeat alert sound for human detection
VOICE_MESSAGE = "Warning! Intruder detected!"  # customizable message
# -----------------------------------

def ensure_report_exists(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "detection", "face_count", "notes"])

def load_wave(path):
    if not os.path.exists(path):
        print(f"[WARN] Sound file not found: {path}. Continuing without that sound.")
        return None
    try:
        return sa.WaveObject.from_wave_file(path)
    except Exception as e:
        print(f"[WARN] Could not load sound {path}: {e}")
        return None

def play_wave(wave, repeat_for=0):
    """Play a wave sound. If repeat_for > 0, loop until time expires."""
    if not wave:
        return
    end_time = time.time() + repeat_for if repeat_for > 0 else None
    while True:
        wave.play()
        if not end_time:
            break
        if time.time() >= end_time:
            break
        time.sleep(1)

def speak_message_then_alarm(message, wave, duration, repeat=2):
    """Speak message multiple times, then start alarm for given duration."""
    try:
        engine = pyttsx3.init()
        for _ in range(repeat):
            engine.say(message)
        engine.runAndWait()  # blocks until speech is done
    except Exception as e:
        print("[WARN] TTS failed:", e)
    # After speaking, play alarm sound
    play_wave(wave, duration)

def log_event(path, detection, face_count, notes=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, detection, face_count, notes])

def main():
    ensure_report_exists(REPORT_CSV)

    # Load Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar cascade. Check your OpenCV installation.")
        return

    # Load alert sound
    alert_wave = load_wave(ALERT_SOUND)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        return

    prev_state = None
    last_alert_time = 0

    print("🔍 Security camera started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame from camera. Exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE
            )
            face_count = len(faces)
            state = "human" if face_count > 0 else "no_face"
            now = time.time()

            if state != prev_state:
                if state == "human":
                    notes = "Faces detected"
                    log_event(REPORT_CSV, state, face_count, notes)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -> HUMAN DETECTED (faces: {face_count})")

                    if now - last_alert_time >= ALERT_COOLDOWN:
                        # Speak twice, THEN play alarm for 30s
                        threading.Thread(
                            target=speak_message_then_alarm,
                            args=(VOICE_MESSAGE, alert_wave, HUMAN_ALERT_DURATION, 2),
                            daemon=True
                        ).start()
                        last_alert_time = now
                else:
                    # No logging, no sound — just print
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -> NO FACE")

                prev_state = state

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Security Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
