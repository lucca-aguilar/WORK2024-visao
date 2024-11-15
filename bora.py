import cv2
import time
import serial
import apriltag
import numpy as np
import gc  # Importa a biblioteca de coleta de lixo
from collections import Counter

# Configuração da comunicação serial
arduino = serial.Serial("/dev/ttyS0", 115200)  # Verifique se a porta está correta
time.sleep(2)  # Aguarda a estabilização da conexão

# Definindo limites de cor para detecção de vermelho e azul no espaço HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Inicializa a câmera
cap = cv2.VideoCapture(0)  # Ajuste o índice da câmera conforme necessário
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Largura
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura

# Inicializa o detector de AprilTags uma vez
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)

def detectar_tag():
    while True:
        fim = time.time() + 1  # Define o tempo de detecção (3 segundos)
        right_X = 0
        right_tag_id = None

        while time.time() < fim:
            time.sleep(1)
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar o frame.")
                break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        # Verifica a tag mais próxima -> Verificar tag mais à direita AAAAAAAAAAAAAAAAAAAAAAAAAAAA
        if tags:
            for tag in tags:
                if len(tag.corners) > 0:
                    cX = int(tag.center[0])
                else:
                    cX = 0
                if cX > right_X:
                    right_X = cX
                    right_tag_id = tag.tag_id
                    right_tag = tag
            print("Tag detectada:")
            print(right_tag_id)
            print("\n")
            arduino.write(f"{right_tag_id}\n".encode())
        else:
            print("Nenhuma tag detectada\n")
  
# Função para alinhar com a AprilTag mais próxima em tempo real, fornecendo feedback contínuo
def alinhar_com_tag():
    while True:
        fim = time.time() + 3  # Define o tempo de detecção (3 segundos)
        right_X = 0
        right_tag_id = None

        while time.time() < fim:
            time.sleep(1)
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar o frame.")
                break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        # Verifica a tag mais próxima -> Verificar tag mais à direita AAAAAAAAAAAAAAAAAAAAAAAAAAAA
        if tags:
            for tag in tags:
                if len(tag.corners) > 0:
                    cX = int(tag.center[0])
                else:
                    cX = 0
                if cX > right_X:
                    right_X = cX
                    right_tag_id = tag.tag_id
                    right_tag = tag
            print("Tag detectada:")
            print(right_tag_id)
            print("\n")
            arduino.write(f"{right_tag_id}\n".encode())
        else:
            print("Nenhuma tag detectada\n")

        if right_tag_id:
            # Encontra o centro da tag
            cX = int(right_tag.center[0])

            # Verifica se a tag está centralizada
            if cX < 300:
                arduino.write('L'.encode())
                print("Tag à esquerda\n")
            elif cX > 340:
                arduino.write('R'.encode())
                print("Tag à direita\n")
            else:
                arduino.write('S'.encode())
                print("Tag centralizada\n")
                return right_tag_id if right_tag_id is not None else 1000
        else:
            print("Nenhuma tag detectada")
            arduino.write(f"{-1}\n".encode())
            break

    gc.collect()  # Força a coleta de lixo

# Função para detectar a cor mais frequente em um intervalo de 3 segundos
def detectar_cor_mais_frequente():
    fim = time.time() + 3  # Define o tempo de detecção (3 segundos)
    cores_detectadas = []  # Lista para armazenar as cores detectadas

    while time.time() < fim:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Converte o frame para HSV e aplica as máscaras de cor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Conta os pixels de cada máscara
        red_area = cv2.countNonZero(mask_red)
        blue_area = cv2.countNonZero(mask_blue)

        # Adiciona a cor detectada à lista
        if red_area > blue_area and red_area > 5000:
            cores_detectadas.append('R')
        elif blue_area > red_area and blue_area > 5000:
            cores_detectadas.append('B') # Otimiza a leitura dos frames

    # Retorna a cor mais frequente
    if cores_detectadas:
        cor_mais_frequente = Counter(cores_detectadas).most_common(1)[0][0]
    else:
        cor_mais_frequente = 'N'  # 'N' para nenhum resultado

    gc.collect()  # Força a coleta de lixo
    return cor_mais_frequente

try:
    while True:
        # Verifica se há um comando do Arduino
        if arduino.in_waiting > 0:
            comando = arduino.read().decode().strip()  # Lê o comando do Arduino
            if comando == 'A': # Verifica apenas o ID da tag
                detectar_tag()
            elif comando == 'C':  # Inicia a detecção ao receber 'C'
                cor = detectar_cor_mais_frequente()
                arduino.write(cor.encode())  # Envia a cor detectada para o Arduino
            elif comando == 'L':  # Inicia alinhamento com April Tags
                alinhar_com_tag()

except KeyboardInterrupt:
    print("Interrompido pelo usuário.")
finally:
    # Libera a câmera e fecha a conexão serial
    cap.release()
    arduino.close()
    gc.collect()  # Coleta de lixo final após liberar recursos
