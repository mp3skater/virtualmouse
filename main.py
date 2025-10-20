"""
Hand-based mouse emulator for macOS using OpenCV + MediaPipe + PyAutoGUI

Usage:
 1) Install dependencies:
    pip install opencv-python mediapipe pyautogui

 2) Grant permissions on macOS (very important):
    - System Settings -> Privacy & Security -> Accessibility: add Terminal (or your Python IDE) so the script can move/click the mouse.
    - System Settings -> Privacy & Security -> Camera: allow Terminal/Python to use the webcam (if prompted)

 3) Run:
    python mac_hand_mouse.py

Controls implemented:
 - Move cursor: move your index finger. The tip of the index finger (landmark 8) is mapped to screen coordinates.
 - Click (two modes):
    * PINCH mode (default): bring thumb tip (4) and index tip (8) close together to click.
    * FIST mode: make a closed fist (all finger tips close to the wrist) to click.

Notes:
 - The script includes smoothing for cursor movement.
 - Tweak thresholds and smoothing parameters at the top of the file.

"""

import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ------------ CONFIG ---------------
CLICK_MODE = 'PINCH'  # 'PINCH' or 'FIST'
SMOOTHING = 7         # higher is smoother but more lag
CLICK_COOLDOWN = 0.35 # seconds between clicks
PINCH_THRESHOLD = 40  # pixels (distance between thumb tip and index tip)
FIST_THRESHOLD = 0.25 # normalized distance ratio for fist detection
CAMERA_INDEX = 0      # default webcam
FRAME_REDUCTION = 100 # reduce active area padding (helps control mapping)

SMOOTHING_BASE   = 6        # smaller -> faster / less smooth; larger -> slower / smoother
STEPS_PER_FRAME  = 3        # synthetic intermediate updates per camera frame (1 = no extra steps)
DRAG_ON_PINCH    = True     # True => pinch+hold -> mouseDown (drag). Release -> mouseUp
PINCH_HOLD_TIME  = 0.10     # seconds required before starting drag (0 = immediate)

# tap/click tuning
TAP_MAX_DURATION = 0.20     # max time (s) pinch can be held and still count as a tap/click
TAP_MAX_MOVE     = 8.0      # max pixels cursor may move during tap to still count as click
DOUBLE_CLICK_TIME= 0.40     # max time between clicks to count as a double-click
# -----------------------------------

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
last_click_time = 0

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.6,
                    max_num_hands=1) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)  # mirror image
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # extract tip coordinates
            lm = hand_landmarks.landmark
            # landmark indices: 4=thumb_tip, 8=index_tip, 12=middle_tip
            thumb = lm[4]
            index = lm[8]
            middle = lm[12]

            # convert to pixel coords (frame coords)
            ix, iy = int(index.x * w), int(index.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)

            # draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # map webcam coordinates to screen coordinates
            # apply frame reduction so you don't have to use full webcam frame edges
            rx = FRAME_REDUCTION
            ry = FRAME_REDUCTION
            usable_w = w - 2 * rx
            usable_h = h - 2 * ry
            # clamp
            cx = min(max(ix - rx, 0), usable_w)
            cy = min(max(iy - ry, 0), usable_h)

            # map to screen
            screen_x = int((cx / usable_w) * screen_w)
            screen_y = int((cy / usable_h) * screen_h)

            # put these config vars near your other config values:
            SMOOTHING_BASE = 4  # smaller -> faster / less smooth; larger -> slower / smoother
            STEPS_PER_FRAME = 3  # synthetic intermediate updates per camera frame (1 = no extra steps)
            DRAG_ON_PINCH = True  # True = pinch and hold -> mouseDown (drag). Release -> mouseUp
            PINCH_HOLD_TIME = 0.50  # seconds required before starting drag (0 = immediate)

            # runtime state (add near prev_x/prev_y/last_click_time)
            is_dragging = False
            pinch_start_time = None

            # Replace your smoothing + move + click detection area with this:
            # map to screen (existing code above should compute screen_x, screen_y)
            # (keep curr_x/prev_x variables, or initialize if missing)

            # adaptive smoothing: compute alpha from distance so far-away jumps are faster
            target_x = screen_x
            target_y = screen_y

            dx = target_x - prev_x
            dy = target_y - prev_y
            dist = math.hypot(dx, dy)

            alpha = 1.0 / SMOOTHING_BASE
            alpha_boost = min(1.0, dist / 400.0)
            alpha = alpha + (alpha_boost * (1.0 - alpha))
            alpha = min(alpha, 1.0)

            new_x = prev_x + dx * alpha
            new_y = prev_y + dy * alpha

            if STEPS_PER_FRAME <= 1:
                curr_x, curr_y = new_x, new_y
                try:
                    pyautogui.moveTo(curr_x, curr_y)
                except Exception as e:
                    print("pyautogui error:", e)
            else:
                for s in range(1, STEPS_PER_FRAME + 1):
                    t = s / float(STEPS_PER_FRAME)
                    step_x = prev_x + (new_x - prev_x) * t
                    step_y = prev_y + (new_y - prev_y) * t
                    try:
                        pyautogui.moveTo(step_x, step_y)
                    except Exception as e:
                        print("pyautogui error:", e)
                curr_x, curr_y = new_x, new_y

            prev_x, prev_y = curr_x, curr_y

            try:
                pyautogui.moveTo(curr_x, curr_y)  # corrected: no horizontal inversion
            except Exception as e:
                print("pyautogui error:", e)

            # CLICK DETECTION
            now = time.time()
            perform_click = False

            if CLICK_MODE == 'PINCH':
                # Euclidean distance between thumb tip and index tip
                dist = math.hypot(tx - ix, ty - iy)
                cv2.putText(frame, f'PinchDist:{int(dist)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if dist < PINCH_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                    perform_click = True

            elif CLICK_MODE == 'FIST':
                # measure average normalized distance of tips to wrist (landmark 0)
                wrist = lm[0]
                wrist_x, wrist_y = wrist.x * w, wrist.y * h
                tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
                avg_dist = 0
                for t in tips:
                    txp, typ = t.x * w, t.y * h
                    avg_dist += math.hypot(txp - wrist_x, typ - wrist_y)
                avg_dist /= len(tips)
                # normalize by frame diagonal
                diag = math.hypot(w, h)
                norm = avg_dist / diag
                cv2.putText(frame, f'FistNorm:{norm:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if norm < FIST_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                    perform_click = True

            if perform_click:
                last_click_time = now
                # left click
                pyautogui.click()
                cv2.putText(frame, 'CLICK', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        # show frame with helpful overlays
        cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION), (w - FRAME_REDUCTION, h - FRAME_REDUCTION), (255,0,0), 2)
        cv2.imshow('Hand Mouse - press q to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            CLICK_MODE = 'FIST' if CLICK_MODE == 'PINCH' else 'PINCH'
            print('Switched click mode to', CLICK_MODE)

cap.release()
cv2.destroyAllWindows()
