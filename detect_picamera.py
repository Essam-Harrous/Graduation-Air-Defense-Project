#!/usr/bin/env python3
"""
Enemy Detection - Works on both Raspberry Pi and Mac (Dashboard Integrated).

Usage:
    python detect_picamera.py          # Run on Raspberry Pi
    python detect_picamera.py --mac    # Run on Mac with webcam
    
Requirements:
    Pi:  pip install ultralytics opencv-python picamera2 numpy flask
    Mac: pip install ultralytics opencv-python numpy flask
"""

import cv2
import time
import argparse
import sys
import threading
from ultralytics import YOLO
import numpy as np
import serial
import serial.tools.list_ports
from dashboard import DashboardServer

# Configuration
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
INFERENCE_SIZE = 192  # Reduced from 320 for faster Pi inference

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

arduino_global = None

def find_arduino():
    """Auto-detect Arduino serial port."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'usbmodem' in port.device or 'usbserial' in port.device or 'ttyUSB' in port.device or 'ttyACM' in port.device:
            return port.device
    return None

class SerialReader:
    def __init__(self, port, baudrate=115200):
        self.arduino = None
        self.running = False
        self.latest_data = {"dist": 0, "angle": 0, "scan": False, "raw_dist": 0}
        self.thread = None
        
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"Connected to Arduino on {port} at {baudrate} baud")
            self.start()
        except serial.SerialException as e:
            print(f"Warning: Could not connect to Arduino: {e}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.arduino:
            self.arduino.close()

    def write(self, data):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(data)
                return True
            except Exception as e:
                print(f"Serial write error: {e}")
        return False

    def _read_loop(self):
        while self.running and self.arduino and self.arduino.is_open:
            try:
                if self.arduino.in_waiting > 0:
                    line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    self._parse_line(line)
            except Exception as e:
                time.sleep(1)

    def _parse_line(self, line):
        if line.startswith("D"):
            try:
                parts = line[1:].split(',')
                distance = int(parts[0])
                angle = int(parts[1]) if len(parts) > 1 else 0
                
                # Filter: Ignore > 50cm
                filtered_dist = 400 if distance > 50 else distance
                
                self.latest_data["dist"] = filtered_dist
                self.latest_data["angle"] = angle
                self.latest_data["scan"] = True
                self.latest_data["raw_dist"] = distance
            except ValueError:
                pass
        elif "Pan:" in line:
            # Debug Format: sErr:x,y Pan:p Tilt:t
            try:
                parts = line.split(" ")
                for p in parts:
                    if p.startswith("Pan:"):
                        self.latest_data["pan"] = int(p.split(":")[1])
                    elif p.startswith("Tilt:"):
                        self.latest_data["tilt"] = int(p.split(":")[1])
            except:
                pass

# Initialize global
serial_reader = None

def init_arduino(port=None):
    """Initialize Arduino serial connection."""
    global serial_reader
    if not port:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "usb" in p.device.lower() or "acm" in p.device.lower():
                port = p.device
                break
        if not port and len(ports) > 0:
            port = ports[0].device
            
    if not port:
        print("Warning: Arduino not found. Servo control disabled.")
        return None
    serial_reader = SerialReader(port)
    return serial_reader

def send_error_to_arduino(arduino_reader, error_x, error_y):
    """Send error values to Arduino for PID control. Format: 'E<errorX>,<errorY>\n'"""
    if arduino_reader is None: return False
    command = f"E{int(error_x)},{int(error_y)}\n"
    return arduino_reader.write(command.encode())

def calculate_error(center_x, center_y, frame_width, frame_height):
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    error_x = center_x - frame_center_x
    error_y = center_y - frame_center_y
    return error_x, error_y

def main():
    global arduino_global
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mac', action='store_true', help='Use Mac webcam instead of Pi Camera')
    parser.add_argument('--port', type=str, default=None, help='Arduino serial port')
    parser.add_argument('--no-servo', action='store_true', help='Disable servo control')
    parser.add_argument('--no-stream', action='store_true', help='Disable video streaming to dashboard (for FPS testing)')
    parser.add_argument('--debug', action='store_true', help='Print timing debug info for each step')
    args = parser.parse_args()
    
    use_mac = args.mac
    platform = "Mac" if use_mac else "Pi"
    print(f"🚀 Starting Enemy Detection on {platform}")

    # Initialize Dashboard
    dashboard = DashboardServer()
    
    def on_center():
        if arduino_global:
            arduino_global.write(b'C\n')
            print("[Command] Centering Servos")
            
    def on_fire():
        # Placeholder for fire
        print("[Command] FIRE!")

    dashboard.on_center = on_center
    dashboard.on_fire = on_fire
    dashboard.start(port=8080)

    # Load YOLOv8 model
    print(f"📦 Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded! Warming up inference engine...")
        # Warmup: Run one dummy inference to trigger ONNX Runtime initialization
        dummy_frame = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        model.predict(source=dummy_frame, imgsz=INFERENCE_SIZE, verbose=False)
        print("✅ engine ready!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize Camera
    picam2 = None
    cap = None
    
    if use_mac:
        print("📷 Initializing Mac Webcam...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            print("❌ Failed to open webcam!")
            return
    else:
        from picamera2 import Picamera2
        print("📷 Initializing Pi Camera V2...")
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
            )
            picam2.configure(config)
            picam2.start()
            print("Camera started!")
        except Exception as e:
            print(f"Camera Init Failed: {e}")
            return
    
    # Initialize Arduino
    if not args.no_servo:
        arduino_global = init_arduino(args.port)
        if arduino_global:
            send_error_to_arduino(arduino_global, 0, 0)
            print("Servos centered (PID mode)")
            dashboard.update_state({"state": "ONLINE - SERIAL ACTIVE"})
        else:
            dashboard.update_state({"state": "ONLINE - NO SERIAL"})
    else:
        dashboard.update_state({"state": "ONLINE - SERVO DISABLED"})
    
    print("\n" + "="*50)
    print("Detection running. Press 'q' to quit.")
    print("Dashboard available at http://localhost:8080")
    print("="*50 + "\n")
    
    prev_time = time.time()
    fps = 0
    frame_count = 0  # For frame skipping
    DASHBOARD_SKIP = 3  # Update dashboard every N frames (1=every frame, 2=every other, etc)
    
    try:
        while True:
            # Capture
            t_start = time.time()
            if use_mac:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
            else:
                frame_bgr = picam2.capture_array()
            t_capture = time.time()
            
            # Predict
            results = model.predict(
                source=frame_bgr,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )
            t_inference = time.time()
            
            # Process Arduino Data (threaded)
            if arduino_global and arduino_global.running:
                data = arduino_global.latest_data
                dashboard.update_state({
                    "radar": {
                        "dist": data["dist"],
                        "angle": data["angle"],
                        "scan": data["scan"]
                    }
                })
                dashboard.state["components"]["ultrasonic"]["status"] = "ONLINE"
                dashboard.state["components"]["ultrasonic"]["val"] = data["raw_dist"]
                dashboard.state["components"]["scan_servo"]["status"] = "ONLINE"
                dashboard.state["components"]["scan_servo"]["val"] = data["angle"]
                # Update Pan/Tilt servos if data available
                if "pan" in data:
                    dashboard.state["components"]["pan_servo"]["status"] = "ONLINE"
                    dashboard.state["components"]["pan_servo"]["val"] = data["pan"]
                else:
                    dashboard.state["components"]["pan_servo"]["status"] = "ONLINE"
                if "tilt" in data:
                    dashboard.state["components"]["tilt_servo"]["status"] = "ONLINE"
                    dashboard.state["components"]["tilt_servo"]["val"] = data["tilt"]
                else:
                    dashboard.state["components"]["tilt_servo"]["status"] = "ONLINE"
                
                dashboard.latest_distance = data["dist"]
            
            # Default status
            real_dist = getattr(dashboard, 'latest_distance', 0)
            
            # Reset status defaults
            current_status = {
                "radar": {"angle": 0, "dist": real_dist, "hit": False},
                "lastDetection": {"name": "-", "conf": 0}
            }
            
            # Track if we found an enemy this frame
            enemy_detected_this_frame = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) 
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    display_name = "enemy" if class_name.lower() == "pepsi" else class_name
                    label = f"{display_name} ({confidence:.2f})"
                    
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - 40), (x1 + max(label_w, 100), y1), (0, 255, 0), -1)
                    
                    cv2.putText(frame_bgr, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    error_x, error_y = calculate_error(center_x, center_y, FRAME_WIDTH, FRAME_HEIGHT)
                    if not args.no_servo and arduino_global:
                        send_error_to_arduino(arduino_global, error_x, error_y)
                    
                    cv2.line(frame_bgr, (FRAME_WIDTH//2 - 10, FRAME_HEIGHT//2), (FRAME_WIDTH//2 + 10, FRAME_HEIGHT//2), (255, 255, 0), 1)
                    cv2.line(frame_bgr, (FRAME_WIDTH//2, FRAME_HEIGHT//2 - 10), (FRAME_WIDTH//2, FRAME_HEIGHT//2 + 10), (255, 255, 0), 1)
                    
                    error_info = f"Error X:{error_x} Y:{error_y}"
                    cv2.putText(frame_bgr, error_info, (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Update Status
                    current_status["radar"] = {
                        "angle": int(error_x),
                        "dist": real_dist,
                        "hit": True
                    }
                    current_status["lastDetection"] = {
                        "name": display_name,
                        "conf": confidence
                    }
                    current_status["pan"] = int(error_x)
                    current_status["tilt"] = int(error_y)
                    
                    enemy_detected_this_frame = True
                    
                    # Update hit flag on radar
                    dashboard.state["radar"]["hit"] = True
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        dashboard.log_event(f"Detected {display_name} (Conf: {confidence:.2f})")

                    break
            
            # --- Radar-to-Camera Handoff ---
            # If no enemy detected by camera, but radar sees something, point camera there
            if not enemy_detected_this_frame and not args.no_servo and arduino_global and arduino_global.running:
                radar_data = arduino_global.latest_data
                if radar_data["dist"] > 0 and radar_data["dist"] <= 50:
                    # Use direct Pan Position command (P<angle>)
                    # Radar scan goes 0-180. Pan servo also 0-180.
                    target_angle = radar_data["angle"]
                    
                    # Send direct position command
                    command = f"P{int(target_angle)}\n"
                    arduino_global.write(command.encode())
                    
                    dashboard.log_event(f"Radar: Investigating target at {radar_data['angle']}° ({radar_data['dist']}cm)")
            
            # FPS calculation (every frame)
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            frame_count += 1
            
            # Dashboard Update (skip frames to improve detection FPS)
            t_process = time.time()
            if frame_count % DASHBOARD_SKIP == 0 and not args.no_stream:
                cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Note: Picamera2 BGR888 returns RGB on some Pi configs, so we convert
                if not use_mac:
                    frame_for_dashboard = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                else:
                    frame_for_dashboard = frame_bgr.copy()
                dashboard.update_frame(frame_for_dashboard)
            
            dashboard.update_state(current_status)
            t_dashboard = time.time()
            
            # Debug timing output
            if args.debug and frame_count % 10 == 0:  # Print every 10 frames
                print(f"[DEBUG] Capture: {(t_capture-t_start)*1000:.1f}ms | "
                      f"Inference: {(t_inference-t_capture)*1000:.1f}ms | "
                      f"Process: {(t_process-t_inference)*1000:.1f}ms | "
                      f"Dashboard: {(t_dashboard-t_process)*1000:.1f}ms | "
                      f"Total: {(t_dashboard-t_start)*1000:.1f}ms | FPS: {fps:.1f}")
            
            # Local Display (Needs RGB on Pi, BGR on Mac)
            # if use_mac:
            #     cv2.imshow("Enemy Detection", frame_bgr)
            # else:
            #     frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            #     cv2.imshow("Enemy Detection", frame_rgb_display)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
    except KeyboardInterrupt:
        print("\n👋 System stopped")
    
    finally:
        if arduino_global:
            send_error_to_arduino(arduino_global, 0, 0)
            arduino_global.stop()
        if use_mac and cap:
            cap.release()
        elif picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        print("Goodbye!")

if __name__ == "__main__":
    main()