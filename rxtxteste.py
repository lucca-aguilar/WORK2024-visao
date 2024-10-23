import cv2
import numpy as np
import serial
import time

# Parâmetros da conexão serial
SERIAL_PORT = '/dev/ttyS0'  # Altere para a porta correta
BAUD_RATE = 250000

# Inicia a comunicação serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Aguarda a inicialização da conexão
    print("Conexão serial estabelecida.")
except serial.SerialException as e:
    print(f"Falha ao estabelecer conexão serial: {e}")
    exit(1)

# Inicia a visão
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro ao acessar a câmera.")
    exit(1)

# Ajusta a resolução
camera.set(3, 640)  # Largura
camera.set(4, 480)  # Altura

# Definir área mínima para componentes
MIN_AREA = 1000  

# Kernel para operações morfológicas
kernel = np.ones((5, 5), np.uint8)

def create_red_mask(hsv_frame):
    """Cria máscara para detecção de vermelho."""
    lower_red_1 = np.array([0, 130, 70])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)

    lower_red_2 = np.array([170, 130, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)

    return cv2.add(mask1, mask2)

frame_count = 0  # Contador de quadros

while True:
    frame_count += 1
    status, frame = camera.read()

    if not status:
        print("Erro ao capturar imagem.")
        break

    # Processa a cada 3 quadros
    if frame_count % 3 == 0:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = create_red_mask(hsv_frame)
        white_mask = cv2.inRange(hsv_frame, np.array([0, 0, 180]), np.array([180, 50, 255]))
        
        # Operações morfológicas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Encontra contornos do vermelho
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_components = [contour for contour in contours_red if cv2.contourArea(contour) > MIN_AREA]

        # Verifica se a máscara branca foi detectada
        white_detected = np.sum(white_mask) > 0

        if len(red_components) >= 3 and white_detected:
            print("Fita zebra detectada.")
            try:
                ser.write(b'V')
                print("Mensagem 'V' enviada para o Arduino.")
            except serial.SerialException:
                print("Falha ao enviar mensagem para o Arduino.")
        else:
            print("Fita zebra não detectada ou incompleta.")
            ser.write(b'S')
            print("Mensagem 'S' enviada para o Arduino.")

        # Ler resposta do Arduino
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            print(f"Mensagem recebida do Arduino: {response}")

    if cv2.waitKey(1) == 27:  # Finaliza ao pressionar ESC
        break

# Fechar a conexão com a câmera e a comunicação serial
camera.release()
cv2.destroyAllWindows()
ser.close()
