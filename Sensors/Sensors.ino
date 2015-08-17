//
// Sensors.ino
// Firefly [v1]
//
// Copyright (c) 2015 Mihir Garimella.
//

#include <Adafruit_Sensor.h>
#include <TMP006.h>
#include <i2c_t3.h>

bool enabled = true;
bool first = true;

// Configure the TMP006 sensors.
TMP006_ temperatureFront(0x44);
TMP006 temperatureBottom;

int temperatureFrontReading_previous;
int temperatureBottomReading_previous;

int concentrationLeftReading_previous;
int concentrationRightReading_previous;

void setup() {
	// Configure the ADC.
	analogReadResolution(12);
	analogReadAveraging(16);
	
	// Start serial communication with the main processor.
	Serial.begin(115200);
	Serial3.begin(115200);
	
	// Start i2c communication with the TMP006 sensors.
	temperatureFront.begin(TMP006_CFG_1SAMPLE);
	temperatureBottom.begin(TMP006_CFG_1SAMPLE);
	
	// Seed a random number generator that we're going to use later on.
	randomSeed(analogRead(0));
}

void loop() {
	unsigned long loopStartTime = millis();
	
	// Read enable/disable commands over serial.
	while (Serial.available() > 1) { }
	if (Serial.available() == 1) {
		char received = Serial.read();
		if (received == 'e') {
			enabled = true;
			first = true;
			temperatureFront.wake();
			temperatureBottom.wake();
		} else if (received == 'd') {
			enabled = false;
			temperatureFront.sleep();
			temperatureBottom.sleep();
		}
	}
	
	if (enabled) {
		// Read each sensor, sending the processed readings to the main processor over serial.
		if (first) {
			temperatureFrontReading_previous = round(10 * temperatureFront.readObjTempC());
			temperatureBottomReading_previous = round(10 * temperatureBottom.readObjTempC());
			first = false;
		} else {
			int d_temperatureFrontReading = round(10 * temperatureFront.readObjTempC()) - temperatureFrontReading_previous;
			int d_temperatureBottomReading = round(10 * temperatureBottom.readObjTempC()) - temperatureBottomReading_previous;
			
			temperatureFrontReading_previous += d_temperatureFrontReading;
			temperatureBottomReading_previous += d_temperatureBottomReading;
			
			Serial3.write(upperByte(temperatureFrontReading_previous));
			Serial3.write(lowerByte(temperatureFrontReading_previous));
			
			Serial3.write(upperByte(temperatureBottomReading_previous));
			Serial3.write(lowerByte(temperatureBottomReading_previous));
			
			if (abs(d_temperatureFrontReading) < 10) d_temperatureFrontReading = 0;
			if (abs(d_temperatureBottomReading) < 10) d_temperatureBottomReading = 0;
			
			Serial3.write(sign(2 * d_temperatureFrontReading + d_temperatureBottomReading));
			
			int d_concentrationLeftReading = analogRead(6) - concentrationLeftReading_previous;
			int d_concentrationRightReading = analogRead(7) - concentrationRightReading_previous;
			
			concentrationLeftReading_previous += d_concentrationLeftReading;
			concentrationRightReading_previous += d_concentrationRightReading;
			
			Serial3.write(upperByte(concentrationLeftReading_previous));
			Serial3.write(lowerByte(concentrationLeftReading_previous));
			
			Serial3.write(upperByte(concentrationRightReading_previous));
			Serial3.write(lowerByte(concentrationRightReading_previous));
			
			if (abs(d_concentrationLeftReading) < 6) d_concentrationLeftReading = 0;
			if (abs(d_concentrationRightReading) < 6) d_concentrationRightReading = 0;
			
			Serial3.write(sign(-d_concentrationLeftReading - d_concentrationRightReading));
			
			if (concentrationLeftReading_previous - concentrationRightReading_previous > 6) {
				Serial3.write(0);
			} else if (concentrationRightReading_previous - concentrationLeftReading_previous > 6) {
				Serial3.write(1);
			} else {
				Serial3.write((uint8_t)(random(2)));
			}
		}
	}
	
	// Run this loop at 4 Hz.
	delay(250 - (int)(millis() - loopStartTime));
}

// Helper functions to convert a 16 bit integer into two 8 bit integers.
byte upperByte(uint16_t x) { return x >> 8; }
byte lowerByte(uint16_t x) { return x & 255 /* 0000000011111111 in binary */; }

// Helper function to return the sign of a number, essentially sgn(x) + 1.
uint8_t sign(int16_t x) { return 2 * (x > 0) + (x == 0); }
