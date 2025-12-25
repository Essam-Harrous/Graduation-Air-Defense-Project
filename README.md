# 🎯 Air Defense - Enemy Detection System

A real-time object detection system with servo-controlled tracking, ultrasonic radar scanning, and a web dashboard for monitoring.

## 📁 Project Structure

```
my-yolo/
├── arduino_servo_control/
│   └── arduino_servo_control.ino   # Arduino firmware for servos & ultrasonic
│
├── static/
│   ├── templates/
│   │   └── index.html              # Dashboard HTML
│   ├── app.js                      # Dashboard JavaScript (radar, controls)
│   └── styles.css                  # Dashboard styling
│
├── detect_mac.py                   # Main detection script (Mac/webcam)
├── detect_picamera.py              # Detection script (Raspberry Pi)
├── dashboard.py                    # Flask dashboard server
│
├── best.onnx                       # Trained YOLOv8 model
├── yolov8n.pt                      # Base YOLOv8 nano weights
│
├── dataset/                        # Training images
├── dataset_annotated/              # Annotated training data
├── prepare_dataset.py              # Dataset preparation script
├── train_yolov8_colab.ipynb        # Colab training notebook
│
└── runs/                           # Training outputs
```

## 🚀 Quick Start

### Mac
```bash
python3 detect_mac.py
```

### Raspberry Pi
```bash
python3 detect_picamera.py
```

### Dashboard
Open http://localhost:8080 in your browser.

## 🔧 Hardware

| Component | Pin |
|-----------|-----|
| Pan Servo | 9 |
| Tilt Servo | 10 |
| Scan Servo | 11 |
| Ultrasonic TRIG | 3 |
| Ultrasonic ECHO | 2 |

## 📡 Features

- **YOLO Detection**: Real-time enemy detection with trained model
- **Servo Tracking**: Pan/tilt follows detected targets
- **Radar Scanning**: 180° ultrasonic sweep with distance detection
- **Dashboard**: Live video, radar display, component status
- **Radar Handoff**: Automatic camera pointing to radar-detected objects
