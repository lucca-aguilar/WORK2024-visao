import cv2 
import numpy as np

vision = True
camera = cv2.VideoCapture(0)

# Definir área mínima para considerar um componente de "tamanho considerável"
MIN_AREA = 100  # Ajuste conforme necessário

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
    lower_white = np.array([0, 0, 190])  # Intervalo baixo para branco
    upper_white = np.array([180, 40, 255])  # Intervalo alto para branco
    whiteMask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # aplicando as máscaras
    redDetection = cv2.bitwise_and(frame, frame, mask=redMask)
    whiteDetection = cv2.bitwise_and(frame, frame, mask=whiteMask)

    # Encontrar componentes conectados e suas estatísticas
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(redMask)
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(whiteMask)

    # Armazenar componentes vermelhos e brancos
    components_red = []
    components_white = []

    for label in range(1, num_labels_red):
        area = stats_red[label, cv2.CC_STAT_AREA]
        if area > MIN_AREA:  # Ignorar áreas muito pequenas
            components_red.append((label, area, centroids_red[label]))

    for label in range(1, num_labels_white):
        area = stats_white[label, cv2.CC_STAT_AREA]
        if area > MIN_AREA:  # Ignorar áreas muito pequenas
            components_white.append((label, area, centroids_white[label]))

    # Se houver componentes detectados
    if len(components_red) > 1:
        # Ordenar componentes com base na posição X do centróide (mais à esquerda e à direita)
        components_red = sorted(components_red, key=lambda x: x[2][0])

        # Pegar o componente mais à esquerda e o mais à direita
        left_component = components_red[0]
        right_component = components_red[-1]

        # Desenhar o bounding box e o centroide do componente mais à esquerda
        x_left = stats_red[left_component[0], cv2.CC_STAT_LEFT]
        y_left = stats_red[left_component[0], cv2.CC_STAT_TOP]
        w_left = stats_red[left_component[0], cv2.CC_STAT_WIDTH]
        h_left = stats_red[left_component[0], cv2.CC_STAT_HEIGHT]
        centroid_left = centroids_red[left_component[0]]

        cv2.rectangle(frame, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid_left[0]), int(centroid_left[1])), 5, (255, 0, 0), -1)

        # Desenhar o bounding box e o centroide do componente mais à direita
        x_right = stats_red[right_component[0], cv2.CC_STAT_LEFT]
        y_right = stats_red[right_component[0], cv2.CC_STAT_TOP]
        w_right = stats_red[right_component[0], cv2.CC_STAT_WIDTH]
        h_right = stats_red[right_component[0], cv2.CC_STAT_HEIGHT]
        centroid_right = centroids_red[right_component[0]]

        cv2.rectangle(frame, (x_right, y_right), (x_right + w_right, y_right + h_right), (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid_right[0]), int(centroid_right[1])), 5, (255, 0, 0), -1)

        # Definir a região de interesse (ROI)
        x_min = min(x_left, x_right)
        x_max = max(x_left + w_left, x_right + w_right)
        y_min = min(y_left, y_right)
        y_max = max(y_left + h_left, y_right + h_right)

        # Desenhar a ROI na imagem original
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        # Extrair as máscaras da ROI
        redMask_roi = redMask[y_min:y_max, x_min:x_max]
        whiteMask_roi = whiteMask[y_min:y_max, x_min:x_max]

        # Contar componentes vermelhos dentro da ROI
        red_count = 0
        for label in range(1, num_labels_red):
            area = stats_red[label, cv2.CC_STAT_AREA]
            if area > MIN_AREA:  # Considerar apenas os componentes com área considerável
                # Desenhar bounding box do componente
                x = stats_red[label, cv2.CC_STAT_LEFT]
                y = stats_red[label, cv2.CC_STAT_TOP]
                w = stats_red[label, cv2.CC_STAT_WIDTH]
                h = stats_red[label, cv2.CC_STAT_HEIGHT]
                if x_min <= x <= x_max and y_min <= y <= y_max:  # Checar se está na ROI
                    red_count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vermelho para bounding box

        # Contar componentes brancos dentro da ROI e entre os dois componentes vermelhos
        white_count = 0
        for label in range(1, num_labels_white):
            area = stats_white[label, cv2.CC_STAT_AREA]
            if area > MIN_AREA:  # Considerar apenas os componentes com área considerável
                # Desenhar bounding box do componente
                x = stats_white[label, cv2.CC_STAT_LEFT]
                y = stats_white[label, cv2.CC_STAT_TOP]
                w = stats_white[label, cv2.CC_STAT_WIDTH]
                h = stats_white[label, cv2.CC_STAT_HEIGHT]
                if x_min <= x <= x_max and y_min <= y <= y_max:  # Checar se está na ROI
                    # Verificar se está entre os dois componentes vermelhos
                    if (x >= x_left and (x + w) <= (x_right + w_right)):
                        white_count += 1
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Branco para bounding box

        # Exibir contagens no console
        print(f"Componentes vermelhos consideráveis na ROI: {red_count}")
        print(f"Componentes brancos consideráveis entre os vermelhos: {white_count}")

    # Mostrar a imagem original e a detecção de vermelho e branco
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)
    cv2.imshow('Detecção de Branco', whiteDetection)

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows()


