
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~25MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

FINGER_TIPS = [3, 7, 11, 15, 19]
FINGER_PIPS = [3, 6, 10, 14, 20]


def count_fingers(hand_landmarks, handedness):
    """Count raised fingers from hand landmarks."""
    lm = hand_landmarks
    fingers_up = 0

    is_right = (handedness == "Right")
    if is_right:
        if lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x:
            fingers_up += 1
    else:
        if lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x:
            fingers_up += 1

    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip_id].y < lm[pip_id].y:
            fingers_up += 1

    return fingers_up


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Finger Counter started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        finger_count = 0
        hand_found   = False

        if result.hand_landmarks:
            hand_found = True
            lm         = result.hand_landmarks[0]
            handedness = result.handedness[0][0].display_name
            finger_count = count_fingers(lm, handedness)

        # ── Draw big number ──────────────────────────────────────────────────
        cv2.putText(frame, str(finger_count),
                    (w // 2 - 60, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8, (0, 0, 0), 20, cv2.LINE_AA)
        cv2.putText(frame, str(finger_count),
                    (w // 2 - 60, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8, (255, 255, 255), 12, cv2.LINE_AA)

        # ── Status bar ───────────────────────────────────────────────────────
        status = "Hand detected" if hand_found else "Show your hand..."
        cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 100) if hand_found else (100, 100, 255),
                    2, cv2.LINE_AA)

        cv2.putText(frame, "Press Q to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Finger Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Finger Counter stopped.")


if __name__ == "__main__":
    main()