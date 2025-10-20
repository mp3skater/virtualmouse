"""
Hand-to-mouse controller for macOS
- Move cursor by moving your hand in front of the webcam.
- Click: pinch index + thumb (touch tips).
- Drag: close ALL fingers (mouseDown) and move; release when fingers open.

Requirements:
pip install opencv-python mediapipe pyautogui
(mac permissions: Camera, Accessibility)

Author: ChatGPT (adapted for mp3skater)
"""
import cv2
import mediapipe as mp
import time
import math
import pyautogui

# ---------- Settings ----------
SMOOTHING = 0.2         # 0 - no smoothing, 1 - very smooth (lag)
CLICK_DISTANCE_THRESH = 0.04   # normalized (tweak if necessary)
FINGER_FOLDED_THRESH = 0.06    # normalized threshold to decide if finger is folded
CLICK_COOLDOWN = 0.25    # seconds between click registrations
MAX_NUM_HANDS = 1
# ------------------------------

pyautogui.FAILSAFE = False  # optional: disable top-left corner failsafe

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

prev_x, prev_y = None, None
last_click_time = 0
dragging = False

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def normalized_distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

print("Starting. Make sure Terminal/Python has Camera and Accessibility permissions on macOS.")
print("Move your hand to move the cursor. Pinch index+thumb to click. Close all fingers to drag.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally so motion is intuitive (like mirror)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # draw landmarks for debugging
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark

        # landmarks indices of interest
        idx_tip = lm[8]
        thumb_tip = lm[4]
        wrist = lm[0]

        # compute a rough hand size to scale thresholds (distance wrist -> middle_mcp(9))
        hand_size = normalized_distance(wrist, lm[9])  # normalized (0..1) approx

        # map index tip to screen coordinates
        # x: landmark.x is left->right from camera; after flip, it's intuitive to map as is
        target_x = idx_tip.x * screen_w
        target_y = idx_tip.y * screen_h

        # smoothing
        if prev_x is None:
            cur_x, cur_y = target_x, target_y
        else:
            cur_x = prev_x + (target_x - prev_x) * (1 - SMOOTHING)
            cur_y = prev_y + (target_y - prev_y) * (1 - SMOOTHING)

        # move mouse
        try:
            pyautogui.moveTo(cur_x, cur_y, duration=0)  # immediate
        except Exception as e:
            # on mac sometimes accessibility not allowed
            cv2.putText(frame, "Mouse control failed: grant Accessibility permission.", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        prev_x, prev_y = cur_x, cur_y

        # ==== click detection (pinch index + thumb) ====
        pinch_dist = normalized_distance(idx_tip, thumb_tip)
        # scale thresholds with hand_size
        pinch_thresh = CLICK_DISTANCE_THRESH * (1.0)  # relative to normalized coords; tweak if needed

        now = time.time()
        is_pinching = pinch_dist < pinch_thresh

        # ==== finger fold detection (for drag) ====
        # We'll check index, middle, ring, pinky: tip vs pip distance relative to hand size
        # tip ids: 8,12,16,20  pip ids: 6,10,14,18
        folded_count = 0
        finger_pairs = [(8,6),(12,10),(16,14),(20,18)]
        for tip_id, pip_id in finger_pairs:
            tip = lm[tip_id]
            pip = lm[pip_id]
            d = normalized_distance(tip, pip)
            # if tip is very close to pip (folded), count as folded
            if d < FINGER_FOLDED_THRESH * 1.0:
                folded_count += 1

        all_fingers_folded = (folded_count >= 4)  # all closed

        # handle drag (all fingers closed -> mouseDown, maintain while closed)
        if all_fingers_folded and not dragging:
            # start drag
            try:
                pyautogui.mouseDown()
                dragging = True
                # small visual
                cv2.putText(frame, "DRAG START", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            except Exception:
                pass
        elif not all_fingers_folded and dragging:
            # release drag
            try:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(frame, "DRAG END", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            except Exception:
                pass

        # handle pinch click (single click)
        if is_pinching and (now - last_click_time) > CLICK_COOLDOWN:
            # if currently dragging, treat pinch as click? We'll do a normal click only if not dragging.
            if not dragging:
                try:
                    pyautogui.click()
                    last_click_time = now
                    cv2.putText(frame, "CLICK", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                except Exception:
                    pass
            else:
                # if dragging, optionally toggle? We ignore pinch while dragging to avoid conflicts
                pass

        # Debug overlay: show pinch distance and folded count
        cv2.putText(frame, f"PinchDist:{pinch_dist:.3f}", (10,h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Folded:{folded_count}/4", (10,h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    else:
        prev_x, prev_y = None, None  # reset smoothing when no hand detected

    cv2.imshow("Hand Mouse - press ESC to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
try:
    if dragging:
        pyautogui.mouseUp()
except Exception:
    pass
print("Exiting.")
