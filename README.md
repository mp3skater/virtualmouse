# AI Virtual Mouse & Gesture Controller

**Control your computer mouse using hand gestures and a webcam. No extra hardware required.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-teal.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg)

## 📖 Overview

This tool converts your hand movements into mouse cursor actions in real-time. Using **MediaPipe** for hand tracking and **OpenCV** for image processing, it allows you to control your PC touch-free.

Unlike basic implementations, this script features a **Dynamic Calibration System**. You don't need to stretch your arm across the whole camera frame; simply define a comfortable "active zone" in the air, and it maps to your full screen resolution.

## ⚡ Key Features

*   **👆 Touch-Free Navigation:** Move the cursor by moving your wrist.
*   **🤏 Pinch Interactions:** Pinch thumb and index finger to **Click**, **Hold**, and **Drag**.
*   **🎯 Custom Calibration:** Define your own workspace boundaries in 3 seconds.
*   **🌊 Motion Smoothing:** Integrated algorithm to remove jitter and shaky hand movements.
*   **🖥️ Multi-Platform:** Works on Windows, macOS, and Linux.

## Showcase

![Demo](https://www.dropbox.com/scl/fi/80pv3cwealz8fxhp7rldj/temp.gif?rlkey=xfzqvnl8qke61bvvmqxndhkbb&st=sqdvb9zp&dl=0)

---

## 🛠️ Prerequisites

### 1. Python & Libraries
Ensure you have Python installed, then install the dependencies:

```bash
pip install opencv-python mediapipe pyautogui
```

### 2. Permissions (MacOS/Linux)
*   **MacOS:** You must grant **Accessibility** and **Screen Recording** permissions to your Terminal/IDE for `pyautogui` to control the mouse.
*   **Linux:** May require `sudo apt-get install python3-tk python3-dev` depending on your distro.

---

## 🚀 Usage Guide

Run the script:

```bash
python main.py
```

### Phase 1: Calibration (One-time setup)
When the script starts, the mouse won't move yet. You need to define your "virtual mousepad".
1.  Raise your hand in front of the camera.
2.  Move your hand to the **Top-Left** of your desired comfortable area and **Pinch** (Thumb + Index).
3.  Move your hand to the **Bottom-Right** of the area and **Pinch** again.
4.  *The system will draw a yellow box. Your hand movements inside this box now map to the entire screen.*

### Phase 2: Control
*   **Move Cursor:** Move your hand freely. The Green dot (Wrist) controls the position.
*   **Left Click / Drag:** Pinch your Thumb and Index finger together.
    *   *Quick Pinch:* Click.
    *   *Hold Pinch:* Drag files or select text.
*   **Quit:** Press `ESC` on your keyboard.

---

## ⚙️ Configuration

You can tweak the constants at the top of the script to fit your environment:

```python
# MIRROR: Set to True if it feels like looking in a mirror (intuitive). 
# Set False if you want direct mapping.
MIRROR = True 

# SMOOTHING: Controls cursor lag vs. stability. 
# 0.1 = Very fast but jittery. 0.9 = Very smooth but slow.
SMOOTHING = 0.45 

# PINCH_THRESHOLD: How close fingers must be to trigger a click.
PINCH_THRESHOLD = 0.05 
```

---

## 🧩 How It Works

1.  **Detection:** MediaPipe extracts 21 hand landmarks (knuckles, fingertips, wrist).
2.  **Tracking:** The script tracks landmark `0` (Wrist) for absolute positioning.
3.  **Interaction:** It calculates the Euclidean distance between landmark `4` (Thumb Tip) and `8` (Index Tip). If distance < `threshold`, it triggers a mouse down event.
4.  **Mapping:** The math function `map_from_calibration` converts the coordinates from the webcam resolution (e.g., 640x480) to your monitor resolution (e.g., 1920x1080) based on your calibration box.

---

## ⚠️ Troubleshooting

**Cursor is shaking too much:**
*   Increase `SMOOTHING` in the code (e.g., to `0.6`).
*   Ensure your room has good lighting.

**Clicks are not registering:**
*   Adjust `PINCH_THRESHOLD`. If your hand is far from the camera, increase it (e.g., `0.08`).

**PyAutoGUI FailSafeException:**
*   The script has `pyautogui.FAILSAFE = False` to prevent crashes when touching screen corners, but always keep your keyboard ready to press `ESC` or `Ctrl+C` in the terminal if the mouse gets stuck.

---

## ⚖️ License

The License is the [MIT License](LICENSE). Enjoy the future of HCI (Human-Computer Interaction)!
