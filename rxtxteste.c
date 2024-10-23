#include <Servo.h>
#include <SoftwareSerial.h>
#include <Arduino.h>

Servo myServo;
int servoPin = 9; // Pino do servo
SoftwareSerial raspy(10, 11);

void setup() {
    Serial.begin(115200); // Inicializa a comunicação serial
    raspy.begin(115200);
    myServo.attach(servoPin); // Anexa o servo ao pino 9
    Serial.println("Conexão serial estabelecida.");
}

void loop() {
    if (raspy.available() > 0) {
        char command = raspy.read(); // Lê o comando recebido

        if (command == 'V') {
            Serial.println("Fita detectada. Movendo o servo.");
            myServo.write(0); // Retorna o servo para a posição inicial (0 graus)
        }

        if (command == 'S') {
            myServo.write(90); // Ajusta o servo para a posição desejada (ex: 90 graus)
            Serial.println("Fita não detectada");
        }
  }
}
