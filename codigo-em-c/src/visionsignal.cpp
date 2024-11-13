#include <Arduino.h>
#include <SoftwareSerial.h>

// Inicialize o SoftwareSerial nos pinos desejados
SoftwareSerial raspy(10, 11); // RX, TX

String receivedData = ""; // Variável para armazenar os dados recebidos

void setup() {
    Serial.begin(9600);       // Inicializa a comunicação serial com o monitor serial
    raspy.begin(9600);        // Inicializa a comunicação serial com a Raspberry Pi
}

void loop() {
    // Verifica se há dados disponíveis para leitura
    if (raspy.available() > 0) {
        char incomingChar = raspy.read(); // Lê o caractere recebido

        // Armazena o caractere na string até encontrar '\n'
        if (incomingChar != '\n') {
            receivedData += incomingChar;
        } else {
            // Imprime a mensagem completa recebida e limpa a variável
            Serial.println(receivedData);
            receivedData = ""; // Limpa a string para receber a próxima mensagem
        }
    }
}