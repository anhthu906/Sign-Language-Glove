#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_ADS1X15.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

// ======================================================
// TensorFlow Lite Micro
// ======================================================


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "glove_model2.h"

// ======================================================
// Hardware
// ======================================================

Adafruit_ADS1115 ads;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29);

#define I2C_SDA    D1
#define I2C_SCL    D2
#define PINKY_PIN  0

bool bnoAvailable = false;

// ======================================================
// Filters
// ======================================================

const float ALPHA = 0.3;

#define MED_SIZE 5


struct MedianFilter {

    int buf[MED_SIZE];

    int idx = 0;

    bool full = false;

    void push(int val) {

        buf[idx] = val;

        idx = (idx + 1) % MED_SIZE;

        if (idx == 0) {
            full = true;
        }
    }

    int median() {

        int tmp[MED_SIZE];

        int count = full ? MED_SIZE : idx;

        if (count == 0) {
            return 0;
        }

        for (int i = 0; i < count; i++) {
            tmp[i] = buf[i];
        }

        for (int i = 1; i < count; i++) {

            int key = tmp[i];

            int j = i - 1;

            while (j >= 0 && tmp[j] > key) {

                tmp[j + 1] = tmp[j];

                j--;
            }

            tmp[j + 1] = key;
        }

        return tmp[count / 2];
    }
};

MedianFilter medThumb;
MedianFilter medIndex;
MedianFilter medMiddle;
MedianFilter medRing;
MedianFilter medPinky;

// ======================================================
// Flex Calibration
// ======================================================

float fThumb  = 0;
float fIndex  = 0;
float fMiddle = 0;
float fRing   = 0;
float fPinky  = 0;

int minThumb  = 65535;
int maxThumb  = 0;

int minIndex  = 65535;
int maxIndex  = 0;

int minMiddle = 65535;
int maxMiddle = 0;

int minRing   = 65535;
int maxRing   = 0;

int minPinky  = 4095;
int maxPinky  = 0;

// ======================================================
// Sliding Window
// ======================================================

float inputMatrix[WINDOW_SIZE][NUM_FEATURES];

float flatFeatures[
    WINDOW_SIZE * NUM_FEATURES
];
int bufferIndex = 0;

bool bufferIsFull = false;
// ======================================================
// Inference Timing
// ======================================================

unsigned long lastInference = 0;
int lastPrediction = -1;
// String lastPrediction = "";



// ======================================================
// TensorFlow Lite Micro
// ======================================================

const tflite::Model* model_tflite = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input  = nullptr;
TfLiteTensor* model_output = nullptr;

// IMPORTANT
constexpr int kTensorArenaSize = 12 * 1024;

alignas(16)
uint8_t tensor_arena[kTensorArenaSize];

// ======================================================
// Push Window
// ======================================================

void pushToWindow(
    float t,
    float i,
    float m,
    float r,
    float p,
    float imuX,
    float imuY,
    float imuZ
) {

    inputMatrix[bufferIndex][0] = t;
    inputMatrix[bufferIndex][1] = i;
    inputMatrix[bufferIndex][2] = m;
    inputMatrix[bufferIndex][3] = r;
    inputMatrix[bufferIndex][4] = p;
    inputMatrix[bufferIndex][5] = imuX;
    inputMatrix[bufferIndex][6] = imuY;
    inputMatrix[bufferIndex][7] = imuZ;

    bufferIndex =
        (bufferIndex + 1) % WINDOW_SIZE;

    if (bufferIndex == 0) {
        bufferIsFull = true;
    }
}

// ======================================================
// Flatten Window
// ======================================================

void flattenWindow(float* outputArray) {

    int idx = 0;

    for (int i = 0; i < WINDOW_SIZE; i++) {

        int actualRow =
            (bufferIndex + i) % WINDOW_SIZE;

        for (int j = 0; j < NUM_FEATURES; j++) {

            outputArray[idx++] =
                inputMatrix[actualRow][j];
        }
    }
}

// ======================================================
// Live Calibration
// ======================================================

void runLiveCalibration() {

    Serial.println(
        ">>> CALIBRATING FLEX SENSORS..."
    );

    unsigned long start = millis();

    while (millis() - start < 4000) {

        int t = ads.readADC_SingleEnded(0);
        int i = ads.readADC_SingleEnded(1);
        int m = ads.readADC_SingleEnded(2);
        int r = ads.readADC_SingleEnded(3);

        int p = analogRead(PINKY_PIN);

        if (t < minThumb)  minThumb = t;
        if (t > maxThumb)  maxThumb = t;

        if (i < minIndex)  minIndex = i;
        if (i > maxIndex)  maxIndex = i;

        if (m < minMiddle) minMiddle = m;
        if (m > maxMiddle) maxMiddle = m;

        if (r < minRing)   minRing = r;
        if (r > maxRing)   maxRing = r;

        if (p < minPinky)  minPinky = p;
        if (p > maxPinky)  maxPinky = p;

        delay(20);
    }

    Serial.println(
        ">>> CALIBRATION COMPLETE!"
    );
}

// ======================================================
// Setup
// ======================================================

void setup() {

    Serial.begin(115200);

    delay(2000);

    Serial.println();
    Serial.println("BOOT START");

    Serial.println(
        "=== GLOVE INITIALIZING ==="
    );

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(100000);

    // ADS1115
    if (!ads.begin()) {

        Serial.println(
            "ADS1115 NOT FOUND!"
        );

        while (1);
    }

    // BNO055
    if (!bno.begin()) {

        Serial.println(
            "BNO055 NOT FOUND!"
        );

        bnoAvailable = false;
    }
    else {

        bnoAvailable = true;
    }

    // Calibration
    runLiveCalibration();

    // ==================================================
    // TensorFlow Lite
    // ==================================================

    Serial.println(
        "Loading TFLite model..."
    );

    model_tflite =
        tflite::GetModel(glove_model_data);

    if (model_tflite->version()
        != TFLITE_SCHEMA_VERSION) {

        Serial.println(
            "Schema mismatch!"
        );

        while (1);
    }

    // Resolver
    
    static tflite::AllOpsResolver resolver;

    // Interpreter
    static tflite::MicroInterpreter
    static_interpreter(

        model_tflite,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    Serial.println(
        "Allocating tensors..."
    );

    if (interpreter->AllocateTensors()
        != kTfLiteOk) {

        Serial.println(
            "Tensor allocation FAILED!"
        );

        while (1);
    }

    Serial.println(
        "Tensor allocation OK!"
    );

    model_input  = interpreter->input(0);
    model_output = interpreter->output(0);

    // Debug shapes
    Serial.println("INPUT SHAPE:");

    for (int i = 0;
         i < model_input->dims->size;
         i++) {

        Serial.print(
            model_input->dims->data[i]
        );

        Serial.print(" ");
    }

    Serial.println();

    Serial.println("OUTPUT SHAPE:");

    for (int i = 0;
         i < model_output->dims->size;
         i++) {

        Serial.print(
            model_output->dims->data[i]
        );

        Serial.print(" ");
    }

    Serial.println();

    Serial.println(
        "=== READY FOR INFERENCE ==="
    );
}

// ======================================================
// Loop
// ======================================================

void loop() {

    // ==================================================
    // Read Sensors
    // ==================================================

    int rawT =
        ads.readADC_SingleEnded(0);

    int rawI =
        ads.readADC_SingleEnded(1);

    int rawM =
        ads.readADC_SingleEnded(2);

    int rawR =
        ads.readADC_SingleEnded(3);

    int rawP =
        analogRead(PINKY_PIN);

    // ==================================================
    // Median Filters
    // ==================================================

    medThumb.push(rawT);
    medIndex.push(rawI);
    medMiddle.push(rawM);
    medRing.push(rawR);
    medPinky.push(rawP);

    int mT = medThumb.median();
    int mI = medIndex.median();
    int mM = medMiddle.median();
    int mR = medRing.median();
    int mP = medPinky.median();

    // ==================================================
    // Low-pass filter
    // ==================================================

    fThumb  =
        ALPHA * mT +
        (1.0 - ALPHA) * fThumb;

    fIndex  =
        ALPHA * mI +
        (1.0 - ALPHA) * fIndex;

    fMiddle =
        ALPHA * mM +
        (1.0 - ALPHA) * fMiddle;

    fRing   =
        ALPHA * mR +
        (1.0 - ALPHA) * fRing;

    fPinky  =
        ALPHA * mP +
        (1.0 - ALPHA) * fPinky;

    // ==================================================
    // Normalize Flex
    // ==================================================

    float normT =
        (maxThumb > minThumb)
        ? (fThumb - minThumb)
            / (float)(maxThumb - minThumb)
        : 0.5;

    float normI =
        (maxIndex > minIndex)
        ? (fIndex - minIndex)
            / (float)(maxIndex - minIndex)
        : 0.5;

    float normM =
        (maxMiddle > minMiddle)
        ? (fMiddle - minMiddle)
            / (float)(maxMiddle - minMiddle)
        : 0.5;

    float normR =
        (maxRing > minRing)
        ? (fRing - minRing)
            / (float)(maxRing - minRing)
        : 0.5;

    float normP =
        (maxPinky > minPinky)
        ? (fPinky - minPinky)
            / (float)(maxPinky - minPinky)
        : 0.5;

    // ==================================================
    // IMU
    // ==================================================

    float rawX = 0;
    float rawY = 0;
    float rawZ = 0;

    if (bnoAvailable) {

        sensors_event_t event;

        bno.getEvent(
            &event,
            Adafruit_BNO055::VECTOR_EULER
        );

        rawX = event.orientation.x;
        rawY = event.orientation.y;
        rawZ = event.orientation.z;
    }

    // Normalize IMU
    float normX =
        (rawX - IMU_MEAN)
        / (IMU_STD + 1e-8);

    float normY =
        (rawY - IMU_MEAN)
        / (IMU_STD + 1e-8);

    float normZ =
        (rawZ - IMU_MEAN)
        / (IMU_STD + 1e-8);

    // ==================================================
    // Push Window
    // ==================================================

    pushToWindow(
        normT,
        normI,
        normM,
        normR,
        normP,
        normX,
        normY,
        normZ
    );
    Serial.println("START INFERENCE");

    // ==================================================
    // Inference
    // ==================================================

    if (
      bufferIsFull &&
      millis() - lastInference > 300
      ) {

        float flatFeatures[
            WINDOW_SIZE * NUM_FEATURES
        ];

        flattenWindow(flatFeatures);

        // Fill tensor
        for (int i = 0;
             i < WINDOW_SIZE * NUM_FEATURES;
             i++) {

            model_input->data.f[i] =
                flatFeatures[i];
        }
        Serial.println("START INFERENCE");

        // Run model
        if (interpreter->Invoke()
            == kTfLiteOk) { 

            int highestClassIdx = 0;

            float maxConfidence = 0;

            for (int i = 0;
                 i < NUM_CLASSES;
                 i++) {

                float conf =
                    model_output->data.f[i];

                if (conf > maxConfidence) {

                    maxConfidence = conf;

                    highestClassIdx = i;
                }
            }
            

            // Threshold
            if (maxConfidence > 0.85f) {

                Serial.print(
                    "--> SIGN: "
                );

                Serial.print(
                    SIGN_LABELS[
                        highestClassIdx
                    ]
                );

                Serial.print(" | ");

                Serial.print(
                    maxConfidence * 100.0f,
                    1
                );

                Serial.println("%");
            }
        }
        else {

            Serial.println(
                "Inference FAILED!"
            );
        }
    }
    else {

        Serial.print("Buffering: ");

        Serial.print(bufferIndex);

        Serial.print("/");

        Serial.println(WINDOW_SIZE);
    }

    // 30Hz
    delay(33);
}