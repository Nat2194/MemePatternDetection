# Meme Overlay: Gesture & Expression Engine

A modular, computer-vision-powered meme overlay. This application uses your webcam to track 3D facial expressions and hand gestures in real-time, automatically triggering custom meme images based on rules you define.

### **Features**

- **3D Geometry Tracking:** Uses true 3D Euclidean math, making it highly resistant to head tilting and varying distances from the camera.
- **Auto-Calibration:** Automatically measures your unique resting face structure upon startup to ensure highly accurate eyebrow and smile tracking.
- **Smart Fallbacks:** Implements body-pose fallback logic to detect fists even when your hands are tucked near your chest or when fingers are obscured (the "Fleshy Blob" solution).
- **Live Debug Overlay:** Displays real-time boolean states and raw mathematical ratios on-screen for easy threshold tuning.
- **Modular JSON Logic:** Easily create complex expression combos (e.g., furious = open mouth + two fists) without touching the Python code.

---

## 🛠️ Prerequisites

- **OS:** Windows (uses a `.bat` launch script).
- **Python:** Python 3.9, 3.10, or 3.11 installed and added to your system PATH. _(Note: MediaPipe can sometimes have compatibility issues with the absolute newest versions of Python)._
- **Webcam:** Any standard USB webcam or virtual camera.

---

## 🚀 Installation & Setup

**1. Prepare Your Directory**
Ensure your project folder contains the following structure:

```text
your-project-folder/
├── main.py         # The core Python script
├── run.bat         # The Windows launch script
├── rules.json      # Your gesture configuration file
└── memes/          # Directory for your image assets
    ├── happy.jpg
    ├── angry.jpg
    └── default.jpg
    └── ...
```
