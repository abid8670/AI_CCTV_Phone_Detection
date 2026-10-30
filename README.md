# AI CCTV Phone Detection

Python project that uses YOLOv8 to detect phone usage in CCTV footage via RTSP stream.

## Features
- Real-time RTSP camera stream capture (multi-threaded)
- YOLOv8 person + cell phone detection
- Alert after 3 seconds of simultaneous person + phone detection
- CPU-friendly (detection every 3 frames, resized to 640x360)

## Requirements
```
pip install ultralytics opencv-python
```

## Usage
1. Edit `RTSP_URL` in `main.py` to your camera stream.
2. Place `yolov8n.pt` in the same folder (auto-downloads on first run).
3. Run: `python main.py`
4. Press ESC to exit.

## Note
Recovered from VS Code local history after original file deletion.
