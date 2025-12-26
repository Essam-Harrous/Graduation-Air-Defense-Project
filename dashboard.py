from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import threading
import cv2
import time
import webbrowser
import subprocess
import os


def open_browser_pi(url):
    """Open browser on Raspberry Pi - tries Chromium first, then fallback methods."""
    # Try Chromium (default on Raspberry Pi OS)
    try:
        subprocess.Popen(['chromium-browser', url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        pass
    
    # Try chromium without -browser suffix
    try:
        subprocess.Popen(['chromium', url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        pass
    
    # Try Firefox
    try:
        subprocess.Popen(['firefox', url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        pass
    
    # Fallback to webbrowser module
    try:
        webbrowser.open(url)
        return True
    except Exception:
        return False

class DashboardServer:
    def __init__(self, template_folder='static/templates', static_folder='static'):
        self.app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Application State
        self.state = {
            "state": "IDLE",
            "radar": {"angle": 0, "dist": 0, "hit": False, "scan": False},
            "components": {
                "pan_servo": {"pin": 9, "status": "OFFLINE", "val": 90},
                "tilt_servo": {"pin": 10, "status": "OFFLINE", "val": 90},
                "scan_servo": {"pin": 11, "status": "OFFLINE", "val": 0},
                "ultrasonic": {"pin": "2,3", "status": "OFFLINE", "val": 0},
                "camera": {"pin": "USB", "status": "ONLINE", "val": "ACTIVE"}
            },
            "pan": 0,
            "tilt": 0,
            "lastDetection": {"name": "-", "conf": 0},
            "autoAlarmRemaining": 0.0,
            "manualArmed": False,
            "log": []
        }
        
        # Callbacks
        self.on_center = None
        self.on_arm = None
        self.on_fire = None
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/static/<path:path>')
        def send_static(path):
            return send_from_directory('static', path)

        @self.app.route('/video')
        def video_feed():
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.state)

        @self.app.route('/api/center', methods=['POST'])
        def center_camera():
            if self.on_center:
                self.on_center()
            self.log_event("Camera centering requested")
            return jsonify({"status": "ok"})

        @self.app.route('/api/arm', methods=['POST'])
        def arm_system():
            data = request.json
            self.state['manualArmed'] = data.get('armed', False)
            if self.on_arm:
                self.on_arm(self.state['manualArmed'])
            self.log_event(f"System {'ARMED' if self.state['manualArmed'] else 'DISARMED'}")
            return jsonify({"status": "ok", "armed": self.state['manualArmed']})

        @self.app.route('/api/fire', methods=['POST'])
        def fire_weapon():
            if not self.state['manualArmed']:
                return jsonify({"status": "error", "message": "System not armed"}), 400
            
            if self.on_fire:
                self.on_fire()
            self.log_event("MANUAL FIRE INITIATED")
            return jsonify({"status": "ok"})

    def _generate_frames(self):
        while self.running:
            with self.lock:
                if self.frame is None:
                    continue
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.frame)
                frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05) # Limit to ~20fps for streaming

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def update_state(self, updates):
        self.state.update(updates)

    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.state['log'].insert(0, {"t": timestamp, "msg": message})
        if len(self.state['log']) > 50:
            self.state['log'].pop()

    def start(self, port=8080):
        def run_flask():
            # Disable Werkzeug logging to keep console clean
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

        thread = threading.Thread(target=run_flask, daemon=True)
        thread.start()
        
        print(f"✅ Dashboard starting at http://localhost:{port}")
        time.sleep(1)  # Give Flask a moment to start
        open_browser_pi(f"http://localhost:{port}")
