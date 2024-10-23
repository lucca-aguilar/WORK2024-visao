import cv2
import numpy as np
import serial  
import time

vision = True
camera = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not camera.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Definir área mínima para considerar um componente de "tamanho considerável"
MIN_AREA = 1000  

# Kernel para operações morfológicas
kernel = np.ones((5, 5), np.uint8)

while vision:
    # lê as imagens da webcam
    status, frame = camera.read()

    if not status:
        print("Erro ao capturar imagem.")
        break

    # converte para o espaço de cor HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # criando a máscara para reconhecer o vermelho
    # lower mask (0-10)
    lower_red = np.array([0, 130, 70])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsvFrame, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 130, 70])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsvFrame, lower_red, upper_red)

    redMask = mask0 + mask1

    # ajustando a máscara para reconhecer o branco (faixa mais ampla)
    lower_white = np.array([0, 0, 180])  # Intervalo baixo para branco
    upper_white = np.array([180, 50, 255])  # Intervalo alto para branco
    whiteMask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # Aplicar operações morfológicas para melhorar a detecção
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, kernel)
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos para vermelho
    contours_red, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Listas para armazenar os componentes detectados
    red_components = []
    white_detected = False

    # Filtrar contornos vermelhos com base na área e guardar as coordenadas
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:  # Ignorar áreas pequenas
            x, y, w, h = cv2.boundingRect(contour)
            red_components.append((x, y, w, h))

            # Calcular a centróide do contorno e desenhar no frame
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

    # Verifica se há qualquer presença de branco
    if np.any(whiteMask > 0):  # Se houver algum pixel branco na máscara
        white_detected = True

    # Verificar se temos pelo menos 3 componentes vermelhos com área considerável
    if len(red_components) >= 3 and white_detected:
        # Ordenar os componentes vermelhos com base na posição horizontal (eixo X)
        red_components_sorted = sorted(red_components, key=lambda x: x[0])

        # Obter o vermelho mais à esquerda e o mais à direita para definir a ROI
        leftmost_red = red_components_sorted[0]
        rightmost_red = red_components_sorted[-1]

        # Criar uma ROI que engloba todos os componentes vermelhos
        roi_x = leftmost_red[0]
        roi_width = rightmost_red[0] + rightmost_red[2] - roi_x
        roi_y = min([y for _, y, _, _ in red_components])
        roi_height = max([y + h for _, y, _, h in red_components]) - roi_y


        # Mostrar mensagem de fita zebra detectada
        print("Fita zebra detectada.")
        # Enviar o caractere 'V' para o Arduino
        serial.write(b'V')
        
    else:
        print("Fita zebra não detectada ou incompleta.")

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

# Fechar a conexão com a câmera e a comunicação serial
camera.release()
cv2.destroyAllWindows()
serial.close()
