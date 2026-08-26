/*
  ==================================================================================================
  SMART HYDROPONIC CONTROLLER - EMBEDDED FIRMWARE (ESP32 DevKit V1)
  ==================================================================================================
  
  1. ARCHITECTURAL MIGRATION SUMMARY (Arduino Mega 2560 + ESP8266 -> ESP32 Standalone):
  --------------------------------------------------------------------------------------------------
  - PREVIOUS ARCHITECTURE:
    * Arduino Mega 2560 as the main controller handling sensors/ML, communicating with an external
      ESP8266 AT co-processor over hardware Serial1 (Pins 18/19 at 9600 baud) via the 'WiFiEsp' library.
    * Problems solved: High latency over UART AT-commands, extra wiring complexity, dual power supplies,
      and 'avr/pgmspace.h' compilation failures on modern non-AVR toolchains.
  
  - CURRENT ESP32 NATIVE ARCHITECTURE:
    * Single dual-core ESP32 micro-controller operating at 240 MHz.
    * Replaced 'WiFiEsp' with ESP32 native '<WiFi.h>', '<WiFiServer.h>', and '<WiFiClient.h>' for direct,
      high-speed SoftAP hosting and instant REST JSON API responses.
    * Upgraded ADC from Mega 2560's 10-bit (0-1023 at 5.0V) to ESP32's 12-bit (0-4095 at 3.3V reference).
    * Sensor inputs mapped strictly to ADC1 channels (GPIO 32, 34, 35) because ESP32 ADC2 channels 
      are unavailable when Wi-Fi is actively transmitting.
    * Custom I2C Bus remapping: Routed SSD1306 OLED display to GPIO 26 (SDA) and GPIO 33 (SCL) via
      'Wire.begin(26, 33)' since default I2C pins (GPIO 21/22) are assigned to 12V dosing relays.

  2. TDS SENSOR TO ELECTRICAL CONDUCTIVITY (EC) CONVERSION FOR MACHINE LEARNING:
  --------------------------------------------------------------------------------------------------
  - CONVERSION PRINCIPLE:
    * TDS (Total Dissolved Solids) probes measure electrical conductivity via AC excitation voltage.
    * The analog output voltage is first sampled with 10-sample moving average on GPIO 34 (12-bit ADC).
    * Temperature Compensation: Solution conductivity rises ~2% per °C above 25.0 °C. The voltage is 
      normalized using: CompVoltage = RawVoltage / (1.0 + 0.02 * (Temp - 25.0)).
    * DFRobot Polynomial Calibration: Compensated voltage is converted to TDS (ppm on 500-scale):
        TDS_ppm = (133.42 * Vc^3 - 255.86 * Vc^2 + 857.39 * Vc) * 0.5
    * Hydroponic Scale Conversion: Standard hydroponics uses the 500-scale (1.0 mS/cm EC = 500 ppm TDS).
      Converting TDS to EC: EC (mS/cm) = TDS_ppm / 500.0.
  - MACHINE LEARNING INTEGRATION:
    * The embedded RandomForest model ('RandomForestEC' in 'EC.h') expects input feature x[0] in mS/cm.
    * Model thresholds: Low <= 1.995 mS/cm (Class 1), Optimal = 2.0 to 3.5 mS/cm (Class 2), High > 3.505 mS/cm (Class 0).
    * This conversion supplies the exact unit scale required by the ML model with zero discrepancies.
  - NaN SHIELDING:
    * Voltage and TDS values are clamped >= 0.0, division-by-zero is blocked, plausibility ranges (0-10 mS/cm)
      are enforced, and 'isnan()' checks fall back to 'lastGoodEC' to guarantee no NaN outputs.

  3. ACTUATOR CONTROL & TIMING SCHEDULES:
  --------------------------------------------------------------------------------------------------
  - Grow Light Schedule: 8 hours ON, followed by 16 hours OFF, repeating endlessly (Active-LOW).
  - Main Water Circulation Pump: 15 minutes ON, 45 minutes OFF (repeating 1-hour cycle).
  - Nutrient & pH Dosing Pumps: Evaluated every 16 hours via ML prediction ('timer.every(SIXTEEN_HR, ...)').
  - ML-to-Relay Class Correction in 'setPump()':
    * Class 1 (Low / Below Target)  -> Activates UP Pump (Nutrient/pH Up) with LOW for 5 seconds.
    * Class 0 (High / Above Target) -> Activates DOWN Pump (Dilution/pH Down) with LOW for 5 seconds.
    * Class 2 (Optimal)             -> Both pumps kept safely OFF (HIGH).
  ==================================================================================================
*/

#include <WiFi.h>              // Native ESP32 WiFi core library
#include <WiFiServer.h>        // Native ESP32 HTTP Server
#include <WiFiClient.h>        // Native ESP32 HTTP Client
#include <Wire.h>              // I2C communication for OLED display
#include <Adafruit_GFX.h>      // Core graphics library
#include <Adafruit_SSD1306.h>  // SSD1306 OLED driver
#include <DHT.h>               // DHT11 temperature/humidity sensor library
#include <arduino-timer.h>     // Non-blocking timer scheduler
#include <math.h>              // isnan(), powf()

// ================= ML MODELS (RandomForest Embedded Classifiers) =================
#include "EC.h"
#include "pH.h"
#include "Humidity.h"
#include "Temperature.h"

Eloquent::ML::Port::RandomForestEC ForestEC;
Eloquent::ML::Port::RandomForestpH ForestPH;
Eloquent::ML::Port::RandomForestHumidity ForestHumidity;
Eloquent::ML::Port::RandomForestTemperature ForestTemperature;

// ================= WIFI NETWORK SETTINGS =================
char ssid[] = "SmartHydro1";
char password[] = "Password123";

String message = "";

// ================= OLED DISPLAY CONFIGURATION =================
#define SCREEN_WIDTH 128       // OLED display width in pixels
#define SCREEN_HEIGHT 64       // OLED display height in pixels
#define OLED_RESET    -1       // Reset pin (-1 sharing micro-controller reset)
#define SCREEN_ADDRESS 0x3C    // Default I2C address for SSD1306

// Custom I2C Pin Remapping for ESP32
// Assigned to free, non-strapping GPIOs 26 & 33 to leave GPIO 21 & 22 free for 12V relays
#define OLED_SDA_PIN  26       // ESP32 SDA Data Pin (GPIO 26)
#define OLED_SCL_PIN  33       // ESP32 SCL Clock Pin (GPIO 33)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool oledAvailable = false;
bool apCreated = false;
char macAddressStr[18] = "00:00:00:00:00:00";

// ================= HTTP REST SERVER =================
WiFiServer server(80);

// Lightweight RingBuffer for HTTP request header extraction
struct RequestBuffer {
  char data[17]; // 16 characters + 1 null terminator
  void init() { 
    memset(data, 0, sizeof(data)); 
  }
  void push(char c) {
    for (int i = 0; i < 15; i++) {
      data[i] = data[i + 1];
    }
    data[15] = c;
    data[16] = '\0'; // Guarantee null-termination for strcmp
  }
  bool endsWith(const char* str) {
    int len = strlen(str);
    if (len > 16) return false;
    return (strcmp(&data[16 - len], str) == 0);
  }
} buf;

// ================= PIN ASSIGNMENTS (ESP32 DevKit V1) =================
// Analog Inputs (Dedicated ADC1 channels - unaffected by Wi-Fi activity)
#define LIGHT_PIN     32       // ADC1_CH4 (LDR Ambient Light Sensor)
#define EC_PIN        34       // ADC1_CH6 (Analog TDS Probe - Input Only)
#define PH_PIN        35       // ADC1_CH7 (Analog pH Probe - Input Only)

// Digital Sensor Pins
#define DHTTYPE DHT11
#define DHT_PIN       27       // Digital GPIO 27 for DHT11 data bus

// 230V Appliance Relays (Active LOW: LOW = Relay Energized / ON, HIGH = OFF)
#define LED_PIN       16       // 230V Relay IN1 — Grow Light
#define FAN_PIN       17       // 230V Relay IN2 — Circulation Fan
#define PUMP_PIN      18       // 230V Relay IN3 — Main Water Pump
#define EXTRACTOR_PIN 19       // 230V Relay IN4 — Exhaust / Extractor Fan

// 12V Peristaltic Dosing Pumps (Active LOW: LOW = Pump ON, HIGH = Pump OFF)
#define PH_UP_PIN     21       // 12V Relay IN1 — pH Up Peristaltic Pump
#define PH_DOWN_PIN   22       // 12V Relay IN2 — pH Down Peristaltic Pump
#define EC_UP_PIN     23       // 12V Relay IN3 — EC Up Nutrient Concentrate Pump
#define EC_DOWN_PIN   25       // 12V Relay IN4 — EC Down Dilution Water Pump

// ================= SENSOR OBJECTS =================
DHT dht(DHT_PIN, DHTTYPE);

// ================= TIMING CONSTANTS (Milliseconds) =================
const unsigned long SIXTEEN_HR            = 57600000UL;// 16 hours (nutrient & pH dosing interval)
const unsigned long PUMP_INTERVAL         = 7000UL;    // 7-second peristaltic dosing pulse duration (5-10s safe window)
const unsigned long QUARTER_HR            = 900000UL;  // 15 minutes water pump ON
const unsigned long FORTY_FIVE_MIN        = 2700000UL; // 45 minutes water pump OFF

// Grow light schedule: 8 h ON then 16 h OFF (24h photoperiod)
const unsigned long EIGHT_HR              = 28800000UL; // 8 hours (grow light ON period)
const unsigned long SIXTEEN_HR_LIGHT      = 57600000UL; // 16 hours (grow light OFF period)

const unsigned long SENSOR_INTERVAL       = 3000UL;    // 3-second cadence for telemetry acquisition
const unsigned long DHT_MIN_INTERVAL      = 2000UL;    // 2-second minimum DHT11 sensor read spacing

auto timer = timer_create_default();

// ================= SENSOR READINGS & LAST-KNOWN-GOOD FALLBACK BUFFERS =================
// Pre-initialized with valid hydroponic baseline values to guarantee zero startup NaNs
float temperature  = 24.0f;       
float humidity     = 60.0f;          
float ecLevel      = 1.5f;           // EC in mS/cm (derived from TDS sensor)
float phLevel      = 6.5f;           
float lightLevel   = 500.0f;        

// Cached fallback buffers in case of an intermittent sensor glitch or wiring disconnect
float lastGoodTemp  = 24.0f;
float lastGoodHum   = 60.0f;
float lastGoodEC    = 1.5f;
float lastGoodPH    = 6.5f;
float lastGoodLight = 500.0f;

unsigned long lastSensorReadMs = 0;
unsigned long lastDhtReadMs = 0;

// ================= FUNCTION DECLARATIONS =================
void togglePin(int pin);
void togglePin(int pin, int toggleValue);
void sendHttpResponse(WiFiClient client, String message);
int analogReadAvg(int pin, uint8_t samples);
float getLightLevel();
float getEC();         // Converts TDS sensor reading on GPIO 34 to EC in mS/cm
float getPH();         // Converts analog pH reading on GPIO 35 to pH units
void readDHTSensors();
void setComponent(int result, int pin, int status);
void setPump(int result, int pinUp, int pinDown, int statusUp, int statusDown);
bool estimateTemperature(void *argument = nullptr);
bool estimateHumidity(void *argument = nullptr);
bool estimatePH(void *argument = nullptr);
bool estimateEC(void *argument = nullptr);
void estimateFactors();
bool disablePH(void *argument = nullptr);
bool disableEC(void *argument = nullptr);
void updateOLEDDisplay();
void retrieveMacAddress();
void initOLED();
bool initAccessPoint();
bool growLightOn(void *argument = nullptr);   // Turns grow light ON (LOW), schedules turn-off after 8 hours
bool growLightOff(void *argument = nullptr);  // Turns grow light OFF (HIGH), schedules turn-on after 16 hours
bool togglePumpOn(void *argument = nullptr);  // Turns water pump ON (LOW), schedules turn-off after 15 minutes
bool togglePumpOff(void *argument = nullptr); // Turns water pump OFF (HIGH), schedules turn-on after 45 minutes

// -------------------------------------------------------------------------------------------------
// Utility: Averaged 12-bit ADC reading on ESP32 (oversampled for noise suppression)
// -------------------------------------------------------------------------------------------------
int analogReadAvg(int pin, uint8_t samples = 10) {
  long sum = 0;
  for (uint8_t i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delayMicroseconds(250); 
  }
  return (int)(sum / samples);
}

// ================= SETUP (System Initialization) =================
void setup() {
  // Initialize Serial Monitor at standard ESP32 baud rate (115200)
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n=========================================================="));
  Serial.println(F("  Smart Hydroponics Controller - Native ESP32 Firmware   "));
  Serial.println(F("=========================================================="));

// -------------------------------------------------------------------------------------------------
// Initializes OLED Display with Automatic I2C Address Detection (0x3C / 0x3D)
// -------------------------------------------------------------------------------------------------
void initOLED() {
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  delay(100);

  Serial.println(F("\n[I2C] Scanning I2C bus on SDA (GPIO 26) & SCL (GPIO 33)..."));
  uint8_t detectedAddress = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print(F("[I2C] Found device at address 0x"));
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
      if (addr == 0x3C || addr == 0x3D) {
        detectedAddress = addr;
      }
    }
  }

  // Try detected address first, or fallback to trying 0x3C then 0x3D
  uint8_t targetAddresses[] = { (detectedAddress != 0) ? detectedAddress : (uint8_t)0x3C, (uint8_t)0x3D };
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t addr = targetAddresses[i];
    if (display.begin(SSD1306_SWITCHCAPVCC, addr)) {
      oledAvailable = true;
      display.dim(false); // Maximum brightness
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(10, 15);
      display.println(F("SmartHydro System"));
      display.setCursor(10, 30);
      display.println(F("Booting Controller..."));
      display.display();
      Serial.print(F("[OLED] SUCCESS: SSD1306 OLED initialized at address 0x"));
      Serial.println(addr, HEX);
      return;
    }
  }

  Serial.println(F("[OLED] Warning: SSD1306 OLED not responding. Check:"));
  Serial.println(F("  1. VCC connected to 3.3V (or 5V) and GND connected to common GND."));
  Serial.println(F("  2. SDA connected to GPIO 26, SCL connected to GPIO 33 (try swapping them if unsure)."));
}

// ================= SETUP (System Initialization) =================
void setup() {
  // Initialize Serial Monitor at standard ESP32 baud rate (115200)
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n=========================================================="));
  Serial.println(F("  Smart Hydroponics Controller - Native ESP32 Firmware   "));
  Serial.println(F("=========================================================="));

  // Initialize OLED display with auto address detection
  initOLED();

  // Initialize WiFi SoftAP Access Point with verification retry loop
  initAccessPoint();

  // Initialize DHT11 sensor
  dht.begin();

  // Configure ESP32 ADC for 12-bit resolution (0 to 4095)
  analogReadResolution(12);

  // Safe Relay Initialization (Active-LOW: Set HIGH before OUTPUT to prevent startup relay clicks)
  const int relayPins[] = { LED_PIN, FAN_PIN, PUMP_PIN, EXTRACTOR_PIN, PH_UP_PIN, PH_DOWN_PIN, EC_UP_PIN, EC_DOWN_PIN };
  for (int i = 0; i < 8; i++) {
    digitalWrite(relayPins[i], HIGH); // Set HIGH (OFF)
    pinMode(relayPins[i], OUTPUT);    // Set as output
  }

  // Force all dosing pumps OFF at startup (Active-LOW: HIGH = OFF)
  disablePH();
  disableEC();

  // Turn ON default active environmental appliances (Active LOW: LOW = ON)
  togglePin(FAN_PIN, LOW);
  togglePin(EXTRACTOR_PIN, LOW);
  togglePin(PUMP_PIN, LOW); 

  // Initial sensor warm-up read
  readDHTSensors();
  lightLevel = getLightLevel();
  ecLevel = getEC();
  phLevel = getPH();

  // Initial OLED dashboard refresh
  updateOLEDDisplay();

  // Scheduled Timers via non-blocking arduino-timer
  timer.every(5000, estimateTemperature); 
  timer.every(5000, estimateHumidity);    
  timer.every(SIXTEEN_HR, estimateEC);    // ML evaluation & Nutrient dosing every 16 hours
  timer.every(SIXTEEN_HR, estimatePH);    // ML evaluation & pH dosing every 16 hours

  // Start Grow Light Cycle: Starts ON immediately at boot, turns OFF after 8 hours
  digitalWrite(LED_PIN, LOW);  // Active LOW: LOW = ON
  Serial.println(F("[GrowLight] 8h ON / 16h OFF cycle started -> Grow light ON"));
  timer.in(EIGHT_HR, growLightOff);

  // Start Water Pump Cycle: Starts ON for 15 minutes, turns OFF for 45 minutes
  timer.in(QUARTER_HR, togglePumpOff); 

  Serial.println(F("[Setup] System Setup Complete. System running.\n"));
}

// ================= MAIN LOOP =================
void loop() {
  WiFiClient client = server.available();
  unsigned long now = millis();

  // Non-blocking paced sensor acquisition
  if (now - lastSensorReadMs >= SENSOR_INTERVAL) {
    lastSensorReadMs = now;

    // Read DHT sensor if minimum interval has elapsed (prevents sensor self-heating)
    if (now - lastDhtReadMs >= DHT_MIN_INTERVAL) {
      lastDhtReadMs = now;
      readDHTSensors();
    }

    lightLevel = getLightLevel();
    ecLevel    = getEC();       // TDS sensor → mS/cm conversion
    phLevel    = getPH();

    // Print real-time sensor diagnostics to Serial Monitor
    Serial.print(F("[Telemetry] Temp: ")); Serial.print(temperature, 1); Serial.print(F(" C | Hum: ")); Serial.print(humidity, 0);
    Serial.print(F(" % | Light ADC: ")); Serial.print((int)lightLevel);
    Serial.print(F(" | EC: ")); Serial.print(ecLevel, 2); Serial.print(F(" mS/cm | pH: ")); Serial.println(phLevel, 2);

    // Refresh live OLED dashboard
    updateOLEDDisplay();
  }

  // Process non-blocking scheduled timers
  timer.tick();

  // Handle incoming HTTP client requests for REST JSON telemetry & overrides
  if (client) {
    buf.init();
    message = "";

    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        buf.push(c);

        // Check for end of HTTP Request Headers
        if (buf.endsWith("\r\n\r\n")) {
          // Serve JSON telemetry containing all calibrated sensor values
          // EC reported in mS/cm (derived from TDS sensor via 500-scale conversion)
          message =
            "{\n  \"PH\": \"" + String(phLevel, 2) +
            "\",\n  \"Light\": \"" + String((int)lightLevel) +
            "\",\n  \"EC\": \"" + String(ecLevel, 2) +
            "\",\n  \"Humidity\": \"" + String(humidity, 1) +
            "\",\n  \"Temperature\": \"" + String(temperature, 1) +
            "\"\n}";
          sendHttpResponse(client, message);
          break;
        }

        // HTTP Endpoint Routing for Manual Actuator Overrides
        if (buf.endsWith("/light")) togglePin(LED_PIN);
        if (buf.endsWith("/fan")) togglePin(FAN_PIN);
        if (buf.endsWith("/extract")) togglePin(EXTRACTOR_PIN);
        if (buf.endsWith("/pump")) togglePin(PUMP_PIN);

        // Peristaltic Dosing Pump Manual Triggers (5-second safety pulse)
        if (buf.endsWith("/phUp")) {
          togglePin(PH_DOWN_PIN, HIGH);
          togglePin(PH_UP_PIN, LOW);
          timer.in(PUMP_INTERVAL, disablePH);
        }

        if (buf.endsWith("/phDown")) {
          togglePin(PH_UP_PIN, HIGH);
          togglePin(PH_DOWN_PIN, LOW);
          timer.in(PUMP_INTERVAL, disablePH);
        }

        if (buf.endsWith("/ecUp")) {
          togglePin(EC_DOWN_PIN, HIGH);
          togglePin(EC_UP_PIN, LOW);
          timer.in(PUMP_INTERVAL, disableEC);
        }

        if (buf.endsWith("/ecDown")) {
          togglePin(EC_UP_PIN, HIGH);
          togglePin(EC_DOWN_PIN, LOW);
          timer.in(PUMP_INTERVAL, disableEC);
        }

        if (buf.endsWith("/ph")) disablePH();
        if (buf.endsWith("/ec")) disableEC();

        if (buf.endsWith("/components")) {
          // digitalRead returns HIGH (1 = Relay OFF) or LOW (0 = Relay ON) for Active-LOW relays.
          // Inverted with '!' so 1 = ON, 0 = OFF in the JSON API response for intuitive client parsing.
          message =
            "{\n  \"PHPump\": \"" + String(!digitalRead(PH_UP_PIN)) +
            "\",\n  \"Light\": \"" + String(!digitalRead(LED_PIN)) +
            "\",\n  \"ECPump\": \"" + String(!digitalRead(EC_UP_PIN)) +
            "\",\n  \"WaterPump\": \"" + String(!digitalRead(PUMP_PIN)) +
            "\",\n  \"Exctractor\": \"" + String(!digitalRead(EXTRACTOR_PIN)) +
            "\",\n  \"Fan\": \"" + String(!digitalRead(FAN_PIN)) +
            "\"\n}";
          sendHttpResponse(client, message);
          break;
        }
      }
    }

    client.stop();
  }
}

// ================= SENSOR ACQUISITION FUNCTIONS (WITH NaN SHIELDING) =================

// -------------------------------------------------------------------------------------------------
// DHT11 Temperature & Humidity Acquisition with Range-Clamping & NaN Fallback
// -------------------------------------------------------------------------------------------------
void readDHTSensors() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();

  // Validate temperature: must not be NaN and within plausible hydroponic range (-10 to 80 °C)
  if (!isnan(t) && t >= -10.0f && t <= 80.0f) {
    temperature = t;
    lastGoodTemp = t;
  } else {
    // Fallback to last known good temperature (guarantees NO NaN propagation)
    temperature = lastGoodTemp;
  }

  // Validate humidity: must not be NaN and within 0-100%
  if (!isnan(h) && h >= 0.0f && h <= 100.0f) {
    humidity = h;
    lastGoodHum = h;
  } else {
    // Fallback to last known good humidity
    humidity = lastGoodHum;
  }
}

// -------------------------------------------------------------------------------------------------
// LDR Ambient Light Sensor on GPIO 32 (12-bit ADC: 0 - 4095)
// -------------------------------------------------------------------------------------------------
float getLightLevel() {
  int raw = analogReadAvg(LIGHT_PIN, 10);
  float val = (float)raw;
  if (!isnan(val) && val >= 0.0f && val <= 4095.0f) {
    lastGoodLight = val;
    return val;
  }
  return lastGoodLight;
}

// -------------------------------------------------------------------------------------------------
// Analog pH Probe on GPIO 35 (12-bit ADC mapped to 0-3300 mV)
// -------------------------------------------------------------------------------------------------
float getPH() {
  // Use current temperature or standard 25.0 °C reference for compensation
  float temp = (!isnan(temperature) && temperature > 0.0f) ? temperature : 25.0f;
  int raw = analogReadAvg(PH_PIN, 10);
  
  // ESP32 12-bit ADC (0 to 4095) mapped to 0 to 3300 mV
  float phVoltage = ((float)raw / 4095.0f) * 3300.0f;
  
  // Standard pH calibration formula (neutral pH 7.0 centered at 1650 mV on 3.3V reference)
  float calculatedPH = 7.0f + ((1650.0f - phVoltage) / 1000.0f) * 3.5f;
  
  // Sanity check: valid pH range is 0.0 to 14.0
  if (!isnan(calculatedPH) && calculatedPH >= 0.0f && calculatedPH <= 14.0f) {
    lastGoodPH = calculatedPH;
    return calculatedPH;
  }
  return lastGoodPH; // Fallback to last-known-good pH
}

// -------------------------------------------------------------------------------------------------
// Analog TDS Probe on GPIO 34 -> Calibrated EC (mS/cm) Conversion for Machine Learning
// -------------------------------------------------------------------------------------------------
float getEC() {
  /*
    TDS TO EC CONVERSION MATHEMATICS:
    ---------------------------------
    1. Read 12-bit ADC (0-4095) at 3.3V reference to calculate raw analog voltage.
    2. Temperature Compensation: Conductivity rises ~2% per °C above 25.0 °C.
       CompVoltage = Voltage / (1.0 + 0.02 * (Temp - 25.0))
    3. DFRobot 500-scale cubic calibration polynomial converts CompVoltage -> TDS (ppm):
       TDS_ppm = (133.42 * Vc^3 - 255.86 * Vc^2 + 857.39 * Vc) * 0.5
    4. Hydroponic Scale Conversion:
       1.0 mS/cm EC = 500 ppm TDS  ==>  EC (mS/cm) = TDS_ppm / 500.0
    5. Machine Learning Output:
       Returns EC in mS/cm matching the feature vector expected by 'ForestEC.predict()'.
  */

  float temp = (!isnan(temperature) && temperature > 0.0f) ? temperature : 25.0f;
  int raw = analogReadAvg(EC_PIN, 10);

  // Step 1: 12-bit ADC (0-4095) at 3.3 V reference -> Voltage in Volts
  float voltage = ((float)raw / 4095.0f) * 3.3f;
  if (voltage < 0.0f) voltage = 0.0f;

  // Step 2: Temperature compensation coefficient
  float compCoeff = 1.0f + 0.02f * (temp - 25.0f);
  if (compCoeff <= 0.001f) compCoeff = 1.0f; // Guard against division by zero
  float compVoltage = voltage / compCoeff;

  // Step 3: Cubic polynomial to convert voltage -> TDS in ppm (500 scale)
  float tds = (133.42f * powf(compVoltage, 3.0f)
             - 255.86f * powf(compVoltage, 2.0f)
             + 857.39f * compVoltage) * 0.5f;
  if (tds < 0.0f) tds = 0.0f;

  // Step 4: TDS (ppm) -> EC (mS/cm) using standard hydroponic 500-scale factor
  float ec_mS = tds / 500.0f;

  // Sanity check: 0.0 - 10.0 mS/cm covers the entire conceivable hydroponic range
  if (!isnan(ec_mS) && ec_mS >= 0.0f && ec_mS <= 10.0f) {
    lastGoodEC = ec_mS;
    return ec_mS;
  }
  return lastGoodEC; // Fallback — never returns NaN
}

// ================= OLED DASHBOARD & ACCESS POINT MANAGEMENT =================

// -------------------------------------------------------------------------------------------------
// SoftAP Access Point Initialization with Verification & Retry Loop
// -------------------------------------------------------------------------------------------------
bool initAccessPoint() {
  Serial.println(F("\n[WiFi] Initializing ESP32 SoftAP..."));

  WiFi.mode(WIFI_AP);

  // Configure custom Static IP for SoftAP (192.168.8.14)
  IPAddress localIp(192, 168, 8, 14);
  IPAddress gateway(192, 168, 8, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.softAPConfig(localIp, gateway, subnet);

  // SoftAP creation retry loop (up to 5 attempts)
  const uint8_t maxRetries = 5;
  for (uint8_t attempt = 1; attempt <= maxRetries; attempt++) {
    Serial.print(F("[WiFi] Creating SoftAP '"));
    Serial.print(ssid);
    Serial.print(F("' (Attempt "));
    Serial.print(attempt);
    Serial.print(F("/"));
    Serial.print(maxRetries);
    Serial.println(F(")..."));

    // Display attempt on OLED if available
    if (oledAvailable) {
      display.clearDisplay();
      display.setTextSize(1);
      display.setCursor(0, 5);
      display.println(F("SMART HYDRO SYSTEM"));
      display.drawLine(0, 15, 127, 15, SSD1306_WHITE);
      display.setCursor(0, 24);
      display.print(F("Starting SoftAP..."));
      display.setCursor(0, 38);
      display.print(F("Attempt "));
      display.print(attempt);
      display.print(F(" of "));
      display.print(maxRetries);
      display.display();
    }

    bool success = WiFi.softAP(ssid, password);
    retrieveMacAddress();

    if (success) {
      apCreated = true;
      Serial.println(F("[WiFi] SUCCESS: Access Point created successfully!"));
      Serial.print(F("[WiFi] SSID: ")); Serial.println(ssid);
      Serial.print(F("[WiFi] Password: ")); Serial.println(password);
      Serial.print(F("[WiFi] AP IP Address: ")); Serial.println(localIp);
      Serial.print(F("[WiFi] MAC Address: ")); Serial.println(macAddressStr);

      server.begin();
      Serial.println(F("[WiFi] HTTP REST Web Server listening on port 80."));

      // Display Success Splash on OLED
      if (oledAvailable) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setCursor(0, 2);
        display.println(F("AP CREATED OK!"));
        display.drawLine(0, 12, 127, 12, SSD1306_WHITE);
        display.setCursor(0, 16);
        display.print(F("SSID: ")); display.println(ssid);
        display.setCursor(0, 28);
        display.print(F("IP:   192.168.8.14"));
        display.setCursor(0, 40);
        display.print(F("MAC: ")); display.println(macAddressStr);
        display.setCursor(0, 52);
        display.println(F("Server: Port 80 OK"));
        display.display();
        delay(2000); // 2-second success splash display
      }
      return true;
    }

    Serial.println(F("[WiFi] SoftAP creation attempt failed. Retrying in 1.5s..."));
    delay(1500);
  }

  Serial.println(F("[WiFi] ERROR: Failed to create SoftAP after multiple attempts."));
  apCreated = false;
  return false;
}

// -------------------------------------------------------------------------------------------------
// Formats ESP32 SoftAP MAC Address into standard hexadecimal string (XX:XX:XX:XX:XX:XX)
// -------------------------------------------------------------------------------------------------
void retrieveMacAddress() {
  String mac = WiFi.softAPmacAddress();
  mac.toCharArray(macAddressStr, sizeof(macAddressStr));
  Serial.print(F("[WiFi] ESP32 MAC Address: "));
  Serial.println(macAddressStr);
}

// -------------------------------------------------------------------------------------------------
// Renders Real-time Telemetry Dashboard & MAC Address onto 0.96" I2C OLED Display
// -------------------------------------------------------------------------------------------------
void updateOLEDDisplay() {
  if (!oledAvailable) return;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Top Header Banner
  display.setTextSize(1);
  display.setCursor(4, 0);
  display.print(F("SMART HYDRO DASHBOARD"));
  display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

  // Line 1: MAC Address Display
  display.setCursor(0, 12);
  display.print(F("MAC: "));
  display.print(macAddressStr);

  // Line 2: WiFi SoftAP & Status Indicator
  display.setCursor(0, 22);
  if (apCreated) {
    display.print(F("AP: "));
    display.print(ssid);
    display.print(F(" [OK]"));
  } else {
    display.print(F("AP: "));
    display.print(ssid);
    display.print(F(" [FAIL]"));
  }

  display.drawLine(0, 32, 127, 32, SSD1306_WHITE);

  // Line 3: Live EC (mS/cm) and pH
  display.setCursor(0, 35);
  display.print(F("EC: "));
  display.print(ecLevel, 2);
  display.print(F("mS"));
  display.setCursor(68, 35);
  display.print(F("pH: "));
  display.print(phLevel, 2);

  // Line 4: Temperature (°C) and Humidity (%)
  display.setCursor(0, 45);
  display.print(F("Temp: "));
  display.print(temperature, 1);
  display.print((char)247); // Degree symbol
  display.print(F("C"));
  display.setCursor(68, 45);
  display.print(F("Hum: "));
  display.print((int)humidity);
  display.print(F("%"));

  // Line 5: Actuator Status (Grow Light & Water Pump)
  display.setCursor(0, 55);
  display.print(F("Lgt:"));
  display.print(digitalRead(LED_PIN) == LOW ? F("ON ") : F("OFF"));
  display.setCursor(68, 55);
  display.print(F("Pmp:"));
  display.print(digitalRead(PUMP_PIN) == LOW ? F("RUN") : F("IDL"));

  display.display();
}

// ================= GROW LIGHT & PUMP TIMERS =================

// -------------------------------------------------------------------------------------------------
// Grow Light: 8 Hours ON -> 16 Hours OFF Non-blocking Cycle
// -------------------------------------------------------------------------------------------------
bool growLightOff(void *argument) {
  digitalWrite(LED_PIN, HIGH); // Active LOW: HIGH = Relay De-energized (OFF)
  Serial.println(F("[GrowLight] 8h ON phase complete -> Grow light OFF (16h OFF phase)"));
  timer.in(SIXTEEN_HR_LIGHT, growLightOn);
  return false; // Complete this one-shot event
}

bool growLightOn(void *argument) {
  digitalWrite(LED_PIN, LOW);  // Active LOW: LOW = Relay Energized (ON)
  Serial.println(F("[GrowLight] 16h OFF phase complete -> Grow light ON (8h ON phase)"));
  timer.in(EIGHT_HR, growLightOff);
  return false; // Complete this one-shot event
}

// -------------------------------------------------------------------------------------------------
// Main Water Pump: 15 Minutes ON -> 45 Minutes OFF Repeating Hydroponic Cycle
// -------------------------------------------------------------------------------------------------
bool togglePumpOn(void *argument) {
  digitalWrite(PUMP_PIN, LOW); // Active LOW: LOW = Pump Energized (ON)
  Serial.println(F("[WaterPump] 45m OFF cycle ended -> Main water pump ON (15 min ON cycle)"));
  timer.in(QUARTER_HR, togglePumpOff);
  return false; // Complete this one-shot event
}

bool togglePumpOff(void *argument) {
  digitalWrite(PUMP_PIN, HIGH); // Active LOW: HIGH = Pump De-energized (OFF)
  Serial.println(F("[WaterPump] 15m ON cycle ended -> Main water pump OFF (45 min OFF cycle)"));
  timer.in(FORTY_FIVE_MIN, togglePumpOn);
  return false; // Complete this one-shot event
}

// ================= ACTUATOR & RELAY HELPERS =================
void togglePin(int pin) {
  digitalWrite(pin, !digitalRead(pin));
}

void togglePin(int pin, int toggleValue) {
  digitalWrite(pin, toggleValue);
}

void sendHttpResponse(WiFiClient client, String message) {
  client.print(
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/json\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Connection: close\r\n"
  );
  if (message.length() > 0) {
    client.print("Content-Length: " + String(message.length()) + "\r\n\r\n");
    client.print(message);
  }
}

void setComponent(int result, int pin, int status) {
  // ML Model Class: 0 = High / Above target, 1 = Low / Below target, 2 = Optimal
  if (result == 0) {
    if (status == HIGH) digitalWrite(pin, LOW);  // Turn ON Fan / Extractor for cooling/venting
  } else if (result == 1) {
    if (status == LOW) digitalWrite(pin, HIGH);  // Turn OFF Fan / Extractor
  }
}

// -------------------------------------------------------------------------------------------------
// Peristaltic Dosing Pump Actuation (Corrected Machine Learning Class Mapping)
// -------------------------------------------------------------------------------------------------
void setPump(int result, int pinUp, int pinDown, int statusUp, int statusDown) {
  /*
    RANDOM FOREST CLASS MAPPING EXPLANATION:
    ----------------------------------------
    * Class 1: Reading is LOW (EC <= 1.995 mS/cm or pH <= 5.80) -> Below Target.
               Action: Activate UP Pump (Nutrient concentrate or pH Up) with LOW (Active-LOW).
    * Class 0: Reading is HIGH (EC > 3.505 mS/cm or pH > 6.30) -> Above Target.
               Action: Activate DOWN Pump (Dilution water or pH Down) with LOW (Active-LOW).
    * Class 2: Reading is in OPTIMAL TARGET RANGE (EC 2.0-3.5 mS/cm, pH 5.8-6.3).
               Action: Keep both pumps de-energized (HIGH).
  */

  if (result == 1) { 
    // Below target -> Activate UP pump (Nutrient Up or pH Up)
    if (statusUp == HIGH || statusDown == LOW) {
      digitalWrite(pinUp, LOW);   // Turn ON UP pump (Active LOW)
      digitalWrite(pinDown, HIGH);// Ensure DOWN pump is OFF
    }
  } else if (result == 0) { 
    // Above target -> Activate DOWN pump (Dilution or pH Down)
    if (statusUp == LOW || statusDown == HIGH) {
      digitalWrite(pinUp, HIGH);  // Ensure UP pump is OFF
      digitalWrite(pinDown, LOW); // Turn ON DOWN pump (Active LOW)
    }
  } else { 
    // Optimal (result == 2) -> Both pumps OFF
    digitalWrite(pinUp, HIGH);
    digitalWrite(pinDown, HIGH);
  }
}

// ================= ML ESTIMATION TASKS (Executed on Scheduled Timers) =================
bool estimateTemperature(void *argument) {
  if (!isnan(temperature)) {
    int result = ForestTemperature.predict(&temperature);
    int fanStatus = digitalRead(FAN_PIN);
    setComponent(result, FAN_PIN, fanStatus);
  }
  return true; // Repeat timer
}

bool estimateHumidity(void *argument) {
  if (!isnan(humidity)) {
    int result = ForestHumidity.predict(&humidity);
    int extractorStatus = digitalRead(EXTRACTOR_PIN);
    setComponent(result, EXTRACTOR_PIN, extractorStatus);
  }
  return true; // Repeat timer
}

bool estimatePH(void *argument) {
  if (!isnan(phLevel)) {
    int result = ForestPH.predict(&phLevel);
    int phUpStatus = digitalRead(PH_UP_PIN);
    int phDownStatus = digitalRead(PH_DOWN_PIN);
    setPump(result, PH_UP_PIN, PH_DOWN_PIN, phUpStatus, phDownStatus);
    timer.in(PUMP_INTERVAL, disablePH); // Automatically turn off dosing pump after 5-second pulse
  }
  return true; // Repeat timer
}

bool estimateEC(void *argument) {
  if (!isnan(ecLevel)) {
    int result = ForestEC.predict(&ecLevel);
    int ecUpStatus = digitalRead(EC_UP_PIN);
    int ecDownStatus = digitalRead(EC_DOWN_PIN);
    setPump(result, EC_UP_PIN, EC_DOWN_PIN, ecUpStatus, ecDownStatus);
    timer.in(PUMP_INTERVAL, disableEC); // Automatically turn off dosing pump after 5-second pulse
  }
  return true; // Repeat timer
}

void estimateFactors() {
  estimatePH();
  estimateTemperature();
  estimateHumidity();
  estimateEC();
}

bool disablePH(void *argument) {
  digitalWrite(PH_UP_PIN, HIGH);
  digitalWrite(PH_DOWN_PIN, HIGH);
  Serial.println(F("[Dosing] pH Pump pulse finished -> pH Pumps OFF (Relays HIGH)"));
  return false; // Complete one-shot
}

bool disableEC(void *argument) {
  digitalWrite(EC_UP_PIN, HIGH);
  digitalWrite(EC_DOWN_PIN, HIGH);
  Serial.println(F("[Dosing] Nutrient Pump pulse finished -> EC Pumps OFF (Relays HIGH)"));
  return false; // Complete one-shot
}