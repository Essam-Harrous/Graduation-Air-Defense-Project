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
INFERENCE_SIZE = 192  # Re-exported ONNX model at this size for speed

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

class RadarOverlay:
    """Draws a radar overlay on the camera frame using OpenCV."""
    
    def __init__(self, size=150, max_range=50):
        self.size = size  # Diameter of radar display
        self.max_range = max_range  # Max distance in cm
        self.blip_history = []  # Store blips with timestamps for fade effect
        self.blip_lifetime = 2.0  # Seconds for blips to fade
        self.displayed_angle = 90  # For smooth animation
        
    def draw(self, frame, angle, dist, hit=False):
        """Draw radar overlay on bottom-right of frame."""
        h, w = frame.shape[:2]
        
        # Radar center position (bottom-right corner)
        cx = w - self.size // 2 - 10
        cy = h - 10
        radius = self.size // 2 - 5
        
        # Create overlay for alpha blending
        overlay = frame.copy()
        
        # --- Background circle (semi-transparent) ---
        cv2.ellipse(overlay, (cx, cy), (radius, radius), 0, 180, 360, (0, 0, 0), -1)
        
        # --- Grid: Concentric arcs ---
        green = (85, 221, 136)  # BGR for radar green
        dark_green = (40, 100, 60)
        
        for scale in [0.25, 0.5, 0.75, 1.0]:
            r = int(radius * scale)
            cv2.ellipse(overlay, (cx, cy), (r, r), 0, 180, 360, dark_green, 1)
        
        # --- Grid: Spoke lines at 0°, 45°, 90°, 135°, 180° ---
        for deg in [0, 45, 90, 135, 180]:
            rad = np.radians(180 + deg)  # Map to semi-circle (180=left, 0=right)
            x_end = int(cx + np.cos(rad) * radius)
            y_end = int(cy + np.sin(rad) * radius)
            cv2.line(overlay, (cx, cy), (x_end, y_end), dark_green, 1)
        
        # --- Smooth sweep line animation ---
        diff = angle - self.displayed_angle
        self.displayed_angle += diff * 0.15
        if abs(diff) < 0.5:
            self.displayed_angle = angle
        
        # Draw sweep line
        sweep_rad = np.radians(180 + (180 - self.displayed_angle))
        sweep_x = int(cx + np.cos(sweep_rad) * radius)
        sweep_y = int(cy + np.sin(sweep_rad) * radius)
        cv2.line(overlay, (cx, cy), (sweep_x, sweep_y), (0, 255, 0), 2)
        
        # Draw sweep wedge (glow effect)
        wedge_pts = [(cx, cy)]
        for a in range(int(self.displayed_angle) - 5, int(self.displayed_angle) + 6):
            rad = np.radians(180 + (180 - a))
            wedge_pts.append((int(cx + np.cos(rad) * radius), int(cy + np.sin(rad) * radius)))
        wedge_pts.append((cx, cy))
        if len(wedge_pts) > 2:
            cv2.fillPoly(overlay, [np.array(wedge_pts)], (0, 180, 0))
        
        # --- Add new blip if object detected ---
        now = time.time()
        if 0 < dist <= self.max_range:
            # Avoid duplicate blips at similar angles
            existing = any(abs(b['angle'] - angle) < 5 and now - b['time'] < 0.2 for b in self.blip_history)
            if not existing:
                self.blip_history.append({'angle': angle, 'dist': dist, 'hit': hit, 'time': now})
        
        # --- Remove old blips ---
        self.blip_history = [b for b in self.blip_history if now - b['time'] < self.blip_lifetime]
        
        # --- Draw blips with fade ---
        for blip in self.blip_history:
            age = now - blip['time']
            alpha = max(0, 1 - age / self.blip_lifetime)
            
            # Calculate blip position
            pix_dist = (blip['dist'] / self.max_range) * radius
            blip_rad = np.radians(180 + (180 - blip['angle']))
            bx = int(cx + np.cos(blip_rad) * pix_dist)
            by = int(cy + np.sin(blip_rad) * pix_dist)
            
            # Color: red for hit, yellow for normal
            if blip['hit']:
                color = (0, 0, int(255 * alpha))  # Red BGR
            else:
                color = (0, int(255 * alpha), int(255 * alpha))  # Yellow BGR
            
            blip_size = int(4 + alpha * 3)
            cv2.circle(overlay, (bx, by), blip_size, color, -1)
            
            # Distance label for fresh blips
            if alpha > 0.7:
                cv2.putText(overlay, f"{blip['dist']}cm", (bx + 8, by + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # --- Labels ---
        cv2.putText(overlay, "RADAR", (cx - 20, cy - radius - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, green, 1)
        cv2.putText(overlay, f"{self.max_range}cm", (cx + radius - 25, cy - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, dark_green, 1)
        
        # --- Blend overlay onto frame ---
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame

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
    parser.add_argument('--no-preview', action='store_true', help='Disable native Pi camera preview window')
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
            print("✅ Camera started!")
            if not args.no_preview:
                print("📺 Local preview window will open (OpenCV)")
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
    DASHBOARD_SKIP = 3  # Update dashboard every N frames
    INFERENCE_SKIP = 1  # Run YOLO every frame (set to 2+ to skip frames)
    
    # Cache for last detection (used when skipping inference)
    cached_detection = None  # (x1, y1, x2, y2, confidence, class_name, center_x, center_y)
    
    # Initialize radar overlay for native preview
    radar_overlay = RadarOverlay(size=160, max_range=50)
    
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
            
            # Run YOLO inference (skip frames for speed, reuse cached detection)
            run_inference = (frame_count % INFERENCE_SKIP == 0)
            
            if run_inference:
                results = model.predict(
                    source=frame_bgr,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )
                # Extract detection from results
                cached_detection = None
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        box = boxes[0]  # First detection
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        cached_detection = (x1, y1, x2, y2, confidence, class_name, center_x, center_y)
                        break
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
            
            # Use cached detection (from YOLO or previous frame)
            if cached_detection is not None:
                x1, y1, x2, y2, confidence, class_name, center_x, center_y = cached_detection
                
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
                
                if confidence > CONFIDENCE_THRESHOLD and run_inference:
                    dashboard.log_event(f"Detected {display_name} (Conf: {confidence:.2f})")
            
            # --- Radar-to-Camera Handoff ---
            # If no enemy detected by camera, but radar sees something, point camera there
            if not enemy_detected_this_frame and not args.no_servo and arduino_global and arduino_global.running:
                radar_data = arduino_global.latest_data
                if radar_data["dist"] > 0 and radar_data["dist"] <= 50:
                    # Cooldown: Only send handoff command once per second
                    now = time.time()
                    last_handoff = getattr(main, '_last_handoff_time', 0)
                    if now - last_handoff > 1.0:
                        main._last_handoff_time = now
                        
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
            
            # Local Display Window (on Pi's HDMI via OpenCV)
            if not args.no_preview:
                # Get radar data for overlay
                radar_angle = 90
                radar_dist = 0
                radar_hit = False
                if arduino_global and arduino_global.running:
                    radar_data = arduino_global.latest_data
                    radar_angle = radar_data.get("angle", 90)
                    radar_dist = radar_data.get("dist", 0)
                    radar_hit = cached_detection is not None  # Hit = camera sees target
                
                # Draw radar overlay on frame
                radar_overlay.draw(frame_bgr, radar_angle, radar_dist, radar_hit)
                
                # Display frame (Mac uses BGR directly, Pi needs RGB conversion)
                if use_mac:
                    cv2.imshow("Enemy Detection", frame_bgr)
                else:
                    frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    cv2.imshow("Enemy Detection", frame_rgb_display)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 Quitting...")
                    break
                
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