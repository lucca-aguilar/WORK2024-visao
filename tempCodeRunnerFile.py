import cv2
import numpy as np

vision = True
camera = cv2.VideoCapture(0)

# Definir área mínima para considerar um componente de "tamanho considerável"
MIN_AREA = 1000  # Ajuste conforme necessário

# Kernel para operações morfológicas
kernel = np.ones((5, 5), np.uint8)

while vision:
    # lê as imagens da webcam
    status, frame = camera.read()

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
    lower_white = np.array([0, 0, 200])  # Intervalo baixo para branco
    upper_white = np.array([180, 25, 255])  # Intervalo alto para branco
    whiteMask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # Aplicar operações morfológicas para melhorar a detecção
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, kernel)
    whiteMask = cv2.morphologyEx(whiteMask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos para vermelho e branco
    contours_red, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(whiteMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar as faixas detectadas
    red_components = []
    white_components = []

    # Filtrar contornos vermelhos com base na área e guardar as coordenadas
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:  # Ignorar áreas pequenas
            x, y, w, h = cv2.boundingRect(contour)
            red_components.append((x, y, w, h))
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)  # Desenhar contornos vermelhos

    # Filtrar contornos brancos com base na área e guardar as coordenadas
    for contour in contours_white:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:  # Ignorar áreas pequenas
            x, y, w, h = cv2.boundingRect(contour)
            white_components.append((x, y, w, h))
            cv2.drawContours(frame, [contour], -1, (255, 255, 255), 2)  # Desenhar contornos brancos

    # Verificar se temos componentes vermelhos e brancos para processar
    if len(red_components) > 0 and len(white_components) > 0:
        # Ordenar os componentes com base na posição horizontal (eixo X)
        red_components = sorted(red_components, key=lambda x: x[0])
        white_components = sorted(white_components, key=lambda x: x[0])

        # Iniciar a contagem alternada de vermelho e branco
        zebra_pattern = []
        last_color = None

        for component in red_components + white_components:
            x, y, w, h = component
            center_x = x + w // 2

            if component in red_components:
                if last_color != "red":
                    zebra_pattern.append("red")
                    last_color = "red"
            elif component in white_components:
                if last_color != "white":
                    zebra_pattern.append("white")
                    last_color = "white"

        # Verificar se o padrão é alternado (fita zebra)
        if len(zebra_pattern) > 1 and zebra_pattern[0] == "red" and zebra_pattern[-1] == "white":
            # Mostrar mensagem na tela
            cv2.putText(frame, "Fita Zebra Detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Padrão de fita zebra detectado: {zebra_pattern}")
        else:
            print("Fita zebra não detectada ou incompleta.")

    # Mostrar a imagem original e a detecção de vermelho e branco
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', cv2.bitwise_and(frame, frame, mask=redMask))
    cv2.imshow('Detecção de Branco', cv2.bitwise_and(frame, frame, mask=whiteMask))

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()