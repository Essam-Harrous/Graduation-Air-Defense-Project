/*
 * Arduino Servo Control with Smoothing for Target Tracking
 * 
 * Receives error values from Python: E<errorX>,<errorY>\n
 * Example: E50,-30\n (target is 50px right, 30px above center)
 * 
 * Uses exponential smoothing to filter noisy detections
 * and proportional control for movement.
 * 
 * Wiring:
 *   Pan Servo  -> Pin 9  (horizontal movement)
 *   Tilt Servo -> Pin 10 (vertical movement)
 *   Servo VCC  -> 5V (or external power for high-torque servos)
 *   Servo GND  -> GND
 *   
 *   Buzzer     -> Pin 4
 *   Laser      -> Pin 5
 */

#include <Servo.h>

// Servo pins
#define PAN_PIN 9
#define TILT_PIN 10
#define SCAN_PIN 11  // New Servo for Radar Scan

// Output pins
#define BUZZER_PIN 4
#define LASER_PIN 5
#define ENEMY_LED_PIN 6

// Ultrasonic pins
#define TRIG_PIN 3
#define ECHO_PIN 2

// Servo limits
#define SERVO_MIN 0
#define SERVO_MAX 180
#define SERVO_CENTER 90

// Control parameters (tune these)
#define SMOOTHING 0.3      // Error smoothing (0.1=very smooth, 0.5=responsive)
#define GAIN 0.02          // How much to move per pixel of error
#define DEADZONE 30        // Pixels from center to ignore (prevents jitter)
#define MAX_STEP 3         // Maximum degrees to move per update

#define TARGET_TIMEOUT 500 // ms to keep "Enemy" status after last tracking command

// Servo objects
Servo panServo;
Servo tiltServo;
Servo scanServo;

// Current servo positions
float panAngle = SERVO_CENTER;
float tiltAngle = SERVO_CENTER;

// Smoothed error values
float smoothErrorX = 0;
float smoothErrorY = 0;

// Tracking state
unsigned long lastTargetTime = 0;

// Serial buffer
char inputBuffer[32];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200); // Faster communication
  
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);
  pinMode(ENEMY_LED_PIN, OUTPUT);
  
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  scanServo.attach(SCAN_PIN);
  
  // Center servos on startup
  panServo.write((int)panAngle);
  tiltServo.write((int)tiltAngle);
  scanServo.write(90);
  
  Serial.println("Servo Control Ready");
}

void loop() {
  // Read serial data
  while (Serial.available() > 0) {
    char inChar = Serial.read();
    
    if (inChar == '\n') {
      inputBuffer[bufferIndex] = '\0';
      parseCommand(inputBuffer);
      bufferIndex = 0;
    } else if (bufferIndex < 31) {
      inputBuffer[bufferIndex] = inChar;
      bufferIndex++;
    }
  }
  
  // Update Buzzer and Laser based on tracking state
  updateBuzzerAndLaser();
  
  // Radar scan variables (declared early, used later)
  static int scanAngle = 0;
  static int scanDir = 2; // 2 degrees per step
  
  // Measure Distance FIRST (this can block up to 25ms)
  static unsigned long lastMeasureTime = 0;
  static int lastDistanceCm = 400;
  static int lastScanAngleForDist = 0;
  
  if (millis() - lastMeasureTime > 30) {
     lastMeasureTime = millis();
     
     digitalWrite(TRIG_PIN, LOW);
     delayMicroseconds(2);
     digitalWrite(TRIG_PIN, HIGH);
     delayMicroseconds(10);
     digitalWrite(TRIG_PIN, LOW);
     
     // Timeout after 25ms (approx 4 meters max distance)
     long duration = pulseIn(ECHO_PIN, HIGH, 25000); 
     
     int distanceCm = 0;
     if (duration == 0) {
         distanceCm = 400; // No echo = max range
     } else {
         distanceCm = duration * 0.034 / 2;
     }
     
     // Cap at 400cm
     if (distanceCm > 400) distanceCm = 400;
     lastDistanceCm = distanceCm;
     lastScanAngleForDist = scanAngle;
     
     // Send D<dist>,<angle>\n
     Serial.print("D");
     Serial.print(distanceCm);
     Serial.print(",");
     Serial.println(lastScanAngleForDist);
  }
  
  // --- Radar Scan Logic (AFTER blocking code) ---
  static unsigned long lastScanTime = 0;
  
  if (millis() - lastScanTime > 25) { // 25ms for smoother motion
      lastScanTime = millis();
      scanAngle += scanDir;
      
      // Clamp and reverse at limits
      if (scanAngle >= 180) {
          scanAngle = 180;
          scanDir = -abs(scanDir);
      } else if (scanAngle <= 0) {
          scanAngle = 0;
          scanDir = abs(scanDir);
      }
      
      scanServo.write(scanAngle);
  }
}

void updateBuzzerAndLaser() {
  bool isEnemy = (millis() - lastTargetTime < TARGET_TIMEOUT);
  
  if (isEnemy) {
    // Enemy Detected Mode
    digitalWrite(LASER_PIN, HIGH); // Laser ON
    
    // Alarm Sound: Fast Beep (100ms ON, 100ms OFF)
    int cycle = millis() % 200;
    if (cycle < 100) {
      digitalWrite(BUZZER_PIN, HIGH);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
    }
    
    // Quick Blink LED (50ms ON, 50ms OFF)
    int ledCycle = millis() % 100;
    if (ledCycle < 50) {
      digitalWrite(ENEMY_LED_PIN, HIGH);
    } else {
      digitalWrite(ENEMY_LED_PIN, LOW);
    }
    
  } else {
    // Scanning Mode
    digitalWrite(LASER_PIN, LOW); // Laser OFF
    
    // Radar Sound: Short Blip (50ms ON, 950ms OFF)
    int cycle = millis() % 1000;
    if (cycle < 50) {
      digitalWrite(BUZZER_PIN, HIGH);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
    }
    
    digitalWrite(ENEMY_LED_PIN, LOW);
  }
}

// Parse command format: E<errorX>,<errorY> or C (Center) or P<angle> (Pan to angle)
void parseCommand(char* cmd) {
  // Center Command
  if (cmd[0] == 'C') {
    panAngle = SERVO_CENTER;
    tiltAngle = SERVO_CENTER;
    panServo.write((int)panAngle);
    tiltServo.write((int)tiltAngle);
    Serial.println("Centered");
    return;
  }
  
  // Direct Pan Position Command (from radar handoff)
  if (cmd[0] == 'P') {
    int targetAngle = atoi(cmd + 1);
    targetAngle = constrain(targetAngle, SERVO_MIN, SERVO_MAX);
    panAngle = targetAngle;
    panServo.write((int)panAngle);
    Serial.print("Pan direct: ");
    Serial.println((int)panAngle);
    return;
  }

  if (cmd[0] == 'E') {
    // Only 'E' commands count as active tracking
    lastTargetTime = millis();
    
    char* commaPos = strchr(cmd, ',');
    if (commaPos == NULL) return;
    
    // Extract raw error values
    int rawErrorX = atoi(cmd + 1);
    int rawErrorY = atoi(commaPos + 1);
    
    // Apply exponential smoothing to filter noise
    smoothErrorX = SMOOTHING * rawErrorX + (1 - SMOOTHING) * smoothErrorX;
    smoothErrorY = SMOOTHING * rawErrorY + (1 - SMOOTHING) * smoothErrorY;
    
    // Calculate movement (only if outside deadzone)
    float moveX = 0;
    float moveY = 0;
    
    if (abs(smoothErrorX) > DEADZONE) {
      moveX = smoothErrorX * GAIN;
      moveX = constrain(moveX, -MAX_STEP, MAX_STEP);
    }
    
    if (abs(smoothErrorY) > DEADZONE) {
      moveY = smoothErrorY * GAIN;
      moveY = constrain(moveY, -MAX_STEP, MAX_STEP);
    }
    
    // Apply movement (negative to move opposite to error)
    panAngle -= moveX;
    tiltAngle -= moveY;
    
    // Constrain to servo limits
    panAngle = constrain(panAngle, SERVO_MIN, SERVO_MAX);
    tiltAngle = constrain(tiltAngle, SERVO_MIN, SERVO_MAX);
    
    // Write to servos
    panServo.write((int)panAngle);
    tiltServo.write((int)tiltAngle);
    
    // Debug output
    Serial.print("sErr:");
    Serial.print((int)smoothErrorX);
    Serial.print(",");
    Serial.print((int)smoothErrorY);
    Serial.print(" Pan:");
    Serial.print((int)panAngle);
    Serial.print(" Tilt:");
    Serial.println((int)tiltAngle);
  }
}

