import cv2
import mediapipe as mp
import pyautogui
import time
import math

# User settings
MIRROR = True
SMOOTHING = 0.45
PINCH_THRESHOLD = 0.05
WEBCAM_INDEX = 0
EDGE_BONUS = 0.1
UPPER_BONUS = 0.2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Quit.")
    exit(1)

screen_w, screen_h = pyautogui.size()
smoothed_x, smoothed_y = None, None
dragging = False
last_pinch_time = 0

def norm_dist(a, b):
    """Euclidean distance between two normalized (x,y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

print("Starting. Make sure Terminal/Python has Accessibility and Screen Recording permissions in macOS Settings.")
print("Press ESC to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        wrist_screen = None
        pinch = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            wrist_norm = (wrist.x, wrist.y)

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            thumb_norm = (thumb_tip.x, thumb_tip.y)
            index_norm = (index_tip.x, index_tip.y)

            d = norm_dist(thumb_norm, index_norm)
            pinch = d < PINCH_THRESHOLD

            # Scale normalized wrist coordinates to screen coordinates
            # Expand slightly to allow reaching edges
            screen_x = min(max((wrist_norm[0] - EDGE_BONUS) / (1-(2*EDGE_BONUS)) * screen_w, 0), screen_w - 1)
            screen_y = min(max((wrist_norm[1] - EDGE_BONUS - UPPER_BONUS) / (1-(2*EDGE_BONUS)-UPPER_BONUS) * screen_h, 0), screen_h - 1)

            # Exponential moving average smoothing
            if smoothed_x is None:
                smoothed_x, smoothed_y = screen_x, screen_y
            else:
                smoothed_x = smoothed_x + (screen_x - smoothed_x) * SMOOTHING
                smoothed_y = smoothed_y + (screen_y - smoothed_y) * SMOOTHING

            wrist_screen = (int(smoothed_x), int(smoothed_y))

            cx = int(wrist_norm[0] * w)
            cy = int(wrist_norm[1] * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0) if not pinch else (0, 0, 255), -1)
            cv2.putText(frame, f"pinch_dist={d:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if wrist_screen is not None:
            x, y = wrist_screen
            try:
                pyautogui.moveTo(x, y, duration=0)
            except Exception as e:
                cv2.putText(frame, f"pyautogui error: {e}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            if pinch and not dragging:
                try:
                    pyautogui.mouseDown()
                    dragging = True
                    last_pinch_time = time.time()
                except Exception as e:
                    print("mouseDown error:", e)
            elif not pinch and dragging:
                try:
                    pyautogui.mouseUp()
                    dragging = False
                except Exception as e:
                    print("mouseUp error:", e)

        status = "DRAGGING" if dragging else "IDLE"
        cv2.putText(frame, status, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if not dragging else (0,0,255), 2)

        cv2.imshow("Hand Mouse (ESC to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    try:
        if dragging:
            pyautogui.mouseUp()
    except Exception:
        pass
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
