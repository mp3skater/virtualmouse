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
UPPER_BONUS = 0

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
prev_pinch = False

calib_points = []
calibrated = False

def norm_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def map_from_calibration(wrist_norm, w, h):
    if calibrated and len(calib_points) == 2:
        (x1, y1), (x2, y2) = calib_points
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        range_x = max(max_x - min_x, 1e-4)
        range_y = max(max_y - min_y, 1e-4)

        nx = (wrist_norm[0] - min_x) / range_x
        ny = (wrist_norm[1] - min_y) / range_y
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        screen_x = min(max((nx - EDGE_BONUS) / (1 - (2 * EDGE_BONUS)) * screen_w, 0), screen_w - 1)
        screen_y = min(max((ny - EDGE_BONUS - UPPER_BONUS) / (1 - (2 * EDGE_BONUS) - UPPER_BONUS) * screen_h, 0), screen_h - 1)

        rect_p1 = (int(min_x * w), int(min_y * h))
        rect_p2 = (int(max_x * w), int(max_y * h))
        return int(screen_x), int(screen_y), rect_p1, rect_p2
    else:
        screen_x = min(max((wrist_norm[0] - EDGE_BONUS) / (1 - (2 * EDGE_BONUS)) * screen_w, 0), screen_w - 1)
        screen_y = min(max((wrist_norm[1] - EDGE_BONUS - UPPER_BONUS) / (1 - (2 * EDGE_BONUS) - UPPER_BONUS) * screen_h, 0), screen_h - 1)
        return int(screen_x), int(screen_y), None, None

print("Starting. Press ESC to quit.")
print("Pinch twice to calibrate (outermost corners). After that, pinch to click/drag.")

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
        wrist_norm = None

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

            screen_x, screen_y, rect_p1, rect_p2 = map_from_calibration(wrist_norm, w, h)

            if smoothed_x is None:
                smoothed_x, smoothed_y = screen_x, screen_y
            else:
                smoothed_x += (screen_x - smoothed_x) * SMOOTHING
                smoothed_y += (screen_y - smoothed_y) * SMOOTHING

            wrist_screen = (int(smoothed_x), int(smoothed_y))

            cx = int(wrist_norm[0] * w)
            cy = int(wrist_norm[1] * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0) if not pinch else (0, 0, 255), -1)
            cv2.putText(frame, f"pinch_dist={d:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if rect_p1 and rect_p2:
                cv2.rectangle(frame, rect_p1, rect_p2, (255, 255, 0), 2)
                cv2.putText(frame, "CALIBRATED AREA", (rect_p1[0], max(rect_p1[1]-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        pinch_start = pinch and not prev_pinch
        prev_pinch = pinch

        if pinch_start and wrist_norm is not None and not calibrated:
            if len(calib_points) < 2:
                calib_points.append(wrist_norm)
                print(f"Calibration point {len(calib_points)} recorded.")
                if len(calib_points) == 2:
                    calibrated = True
                    print("Calibration complete.")
        else:
            if wrist_screen is not None:
                x, y = wrist_screen
                try:
                    pyautogui.moveTo(x, y, duration=0)
                except Exception:
                    pass

                if pinch and not dragging and calibrated:
                    try:
                        pyautogui.mouseDown()
                        dragging = True
                    except Exception:
                        pass
                elif not pinch and dragging:
                    try:
                        pyautogui.mouseUp()
                        dragging = False
                    except Exception:
                        pass

        if not calibrated:
            cv2.putText(frame, f"CALIBRATION: {len(calib_points)}/2 (pinch to set)",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            status = "DRAGGING" if dragging else "IDLE"
            cv2.putText(frame, status, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if not dragging else (0,0,255), 2)

        cv2.imshow("Hand Mouse (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
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
