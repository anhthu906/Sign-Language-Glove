#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

// ----- Hardware Objects -----
Adafruit_ADS1115 ads;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29);

#define I2C_SDA   D1
#define I2C_SCL   D2
#define PINKY_PIN 0

const char* SIGN_LABEL = "baonhieu";

const float ALPHA       = 0.3;
const int   MED_SIZE    = 5;

float fThumb = 0, fIndex = 0, fMiddle = 0, fRing = 0, fPinky = 0;
int offsetThumb = 0, offsetIndex = 0, offsetMiddle = 0, offsetRing = 0, offsetPinky = 0;
bool bnoAvailable = false;

// ========================= MEDIAN FILTER =========================
// One circular buffer per finger
struct MedianFilter {
  int   buf[MED_SIZE];
  int   idx = 0;
  bool  full = false;

  void push(int val) {
    buf[idx] = val;
    idx = (idx + 1) % MED_SIZE;
    if (idx == 0) full = true;
  }

  int median() {
    // Copy active values into a temp array and sort
    int tmp[MED_SIZE];
    int count = full ? MED_SIZE : idx;
    if (count == 0) return 0;

    for (int i = 0; i < count; i++) tmp[i] = buf[i];

    // Insertion sort (fast enough for MED_SIZE ≤ 7)
    for (int i = 1; i < count; i++) {
      int key = tmp[i], j = i - 1;
      while (j >= 0 && tmp[j] > key) { tmp[j+1] = tmp[j]; j--; }
      tmp[j+1] = key;
    }
    return tmp[count / 2];
  }
};

MedianFilter medThumb, medIndex, medMiddle, medRing, medPinky;

// ========================= CALIBRATION ===========================
void calibrateSensors() {
  Serial.println("CALIBRATING...");
  long sumT = 0, sumI = 0, sumM = 0, sumR = 0, sumP = 0;
  const int samples = 60;

  for (int i = 0; i < samples; i++) {
    sumT += ads.readADC_SingleEnded(0);
    sumI += ads.readADC_SingleEnded(1);
    sumM += ads.readADC_SingleEnded(2);
    sumR += ads.readADC_SingleEnded(3);
    sumP += analogRead(PINKY_PIN);
    delay(30);
  }

  offsetThumb  = sumT / samples;
  offsetIndex  = sumI / samples;
  offsetMiddle = sumM / samples;
  offsetRing   = sumR / samples;
  offsetPinky  = sumP / samples;

  Serial.println("CALIBRATION DONE!");
  Serial.print("Offsets -> T:"); Serial.print(offsetThumb);
  Serial.print(" I:"); Serial.print(offsetIndex);
  Serial.print(" M:"); Serial.print(offsetMiddle);
  Serial.print(" R:"); Serial.print(offsetRing);
  Serial.print(" P:"); Serial.println(offsetPinky);
}

// ========================= SETUP =================================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("=== GLOVE BOOT ===");

  Serial.println("[1/4] Starting I2C...");
  Wire.begin(I2C_SDA, I2C_SCL);
  Serial.println("      I2C OK");

  Serial.println("[2/4] Initializing ADS1115...");
  if (!ads.begin()) {
    Serial.println("      FAILED: ADS1115 not found!");
    while (1) { delay(1000); }
  }
  Serial.println("      ADS1115 OK");

  Serial.println("[3/4] Initializing BNO055...");
  if (!bno.begin()) {
    Serial.println("      WARNING: BNO055 not found — IMU will output 0.00");
    bnoAvailable = false;
  } else {
    Serial.println("      BNO055 OK");
    bnoAvailable = true;
  }

  Serial.println("[4/4] Calibrating...");
  calibrateSensors();

  // Seed EMA with calibrated offsets
  fThumb = offsetThumb; fIndex  = offsetIndex;
  fMiddle= offsetMiddle; fRing  = offsetRing; fPinky = offsetPinky;

  Serial.println("=== STREAMING STARTED ===");
  Serial.println("Filter: Median(5) → EMA(0.3)");
  Serial.println("Send 'c' to recalibrate.");
}

// ========================= LOOP ==================================
void loop() {
  if (Serial.available() > 0) {
    if (Serial.read() == 'c') {
      Serial.println("Recalibrating...");
      calibrateSensors();
      fThumb = offsetThumb; fIndex  = offsetIndex;
      fMiddle= offsetMiddle; fRing  = offsetRing; fPinky = offsetPinky;
    }
  }

  // --- 1. RAW READ ---
  int rawT = ads.readADC_SingleEnded(0);
  int rawI = ads.readADC_SingleEnded(1);
  int rawM = ads.readADC_SingleEnded(2);
  int rawR = ads.readADC_SingleEnded(3);
  int rawP = analogRead(PINKY_PIN);

  // --- 2. MEDIAN FILTER (removes spikes) ---
  medThumb.push(rawT);  int mT = medThumb.median();
  medIndex.push(rawI);  int mI = medIndex.median();
  medMiddle.push(rawM); int mM = medMiddle.median();
  medRing.push(rawR);   int mR = medRing.median();
  medPinky.push(rawP);  int mP = medPinky.median();

  // --- 3. EMA FILTER (smooths remaining noise) ---
  fThumb  = (ALPHA * mT) + (1.0 - ALPHA) * fThumb;
  fIndex  = (ALPHA * mI) + (1.0 - ALPHA) * fIndex;
  fMiddle = (ALPHA * mM) + (1.0 - ALPHA) * fMiddle;
  fRing   = (ALPHA * mR) + (1.0 - ALPHA) * fRing;
  fPinky  = (ALPHA * mP) + (1.0 - ALPHA) * fPinky;

  // --- 4. IMU READ ---
  float imuX = 0.0, imuY = 0.0, imuZ = 0.0;
  if (bnoAvailable) {
    sensors_event_t event;
    bno.getEvent(&event, Adafruit_BNO055::VECTOR_EULER);
    imuX = event.orientation.x;
    imuY = event.orientation.y;
    imuZ = event.orientation.z;
  }

  // --- 5. OUTPUT ---
  Serial.print("FLX:");
  Serial.print(abs((int)fThumb  - offsetThumb));  Serial.print(",");
  Serial.print(abs((int)fIndex  - offsetIndex));  Serial.print(",");
  Serial.print(abs((int)fMiddle - offsetMiddle)); Serial.print(",");
  Serial.print(abs((int)fRing   - offsetRing));   Serial.print(",");
  Serial.print(abs((int)fPinky  - offsetPinky));  Serial.print(",");
  Serial.print(imuX, 2); Serial.print(",");
  Serial.print(imuY, 2); Serial.print(",");
  Serial.print(imuZ, 2);
  Serial.println();

  delay(60);
}