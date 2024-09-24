import cv2 
import numpy as np
from matplotlib import pyplot as plt

vision = True
camera = cv2.VideoCapture(0)

while vision:
    # Lê as imagens da webcam
    status, frame = camera.read()

    # Converte para o espaço de cor HSV e para cinza
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Criando a máscara para reconhecer o vermelho
    # Lower mask (0-10)
    lowerRed = np.array([0, 130, 70])
    upperRed = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsvFrame, lowerRed, upperRed)

    # Upper mask (170-180)
    lowerRed = np.array([170, 130, 70])
    upperRed = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsvFrame, lowerRed, upperRed)

    redMask = mask0 + mask1

    # Aplicando a máscara
    redDetection = cv2.bitwise_and(frame, frame, mask=redMask)

    # Encontrar componentes conectados e suas estatísticas
    red_numLabels, red_labelsIm, red_stats, red_centroids = cv2.connectedComponentsWithStats(redMask)

    # Armazenar as estatísticas (exceto o fundo, que é o label 0)
    redComponents = []
    for red_label in range(1, red_numLabels):
        red_area = red_stats[red_label, cv2.CC_STAT_AREA]
        if red_area > 100:  # Ignorar áreas muito pequenas
            redComponents.append((red_label, red_area))

    # Ordenar os componentes por área (do maior para o menor)
    redComponents = sorted(redComponents, key=lambda x: x[1], reverse=True)

    # Considerar apenas os dois maiores componentes
    if len(redComponents) >= 2:
        redComponent1 = redComponents[0]
        redComponent2 = redComponents[1]

        # Obter as coordenadas dos bounding boxes dos dois maiores componentes
        x1 = red_stats[redComponent1[0], cv2.CC_STAT_LEFT]
        y1 = red_stats[redComponent1[0], cv2.CC_STAT_TOP]
        width1 = red_stats[redComponent1[0], cv2.CC_STAT_WIDTH]
        height1 = red_stats[redComponent1[0], cv2.CC_STAT_HEIGHT]

        x2 = red_stats[redComponent2[0], cv2.CC_STAT_LEFT]
        y2 = red_stats[redComponent2[0], cv2.CC_STAT_TOP]
        width2 = red_stats[redComponent2[0], cv2.CC_STAT_WIDTH]
        height2 = red_stats[redComponent2[0], cv2.CC_STAT_HEIGHT]

        # Determinar a região de interesse (ROI) entre os dois objetos
        x_roi = min(x1, x2)
        y_roi = min(y1, y2)
        width_roi = max(x1 + width1, x2 + width2) - x_roi
        height_roi = max(y1 + height1, y2 + height2) - y_roi

        # Extrair a área entre os dois componentes (nossa ROI)
        roi = frame[y_roi:y_roi + height_roi, x_roi:x_roi + width_roi]

        # Desenhar retângulos delimitadores ao redor dos dois maiores componentes
        cv2.rectangle(frame, (x1, y1), (x1 + width1, y1 + height1), (0, 255, 0), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + width2, y2 + height2), (0, 255, 0), 2)

        # Desenhar o retângulo delimitador da ROI
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + width_roi, y_roi + height_roi), (255, 0, 0), 2)

        # Mostrar a ROI separadamente
        cv2.imshow('Região de Interesse', roi)

        # Espaço de cor HSV na ROI
        hsvFrame2 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) 

        # Criando a máscara para reconhecer o branco
        lower_white = np.array([0, 0, 200])  # Limites para branco
        upper_white = np.array([180, 25, 255])

        whiteMask = cv2.inRange(hsvFrame2, lower_white, upper_white)

        # Aplicando a máscara
        whiteDetection = cv2.bitwise_and(roi, roi, mask=whiteMask)

        # Mostrar a detecção de branco na ROI
        cv2.imshow('Detecção de Branco na ROI', whiteDetection)

        # Encontrar componentes conectados e suas estatísticas na ROI
        white_numLabels, white_labelsIm, white_stats, white_centroids = cv2.connectedComponentsWithStats(whiteMask)

         # Armazenar as estatísticas (exceto o fundo, que é o label 0)
        whiteComponents = []
        for white_label in range(1, white_numLabels):
            white_area = white_stats[white_label, cv2.CC_STAT_AREA]
            if white_area > 100:  # Ignorar áreas muito pequenas
                whiteComponents.append((white_label, white_area))

        # Ordenar os componentes por área (do maior para o menor) na ROI
        whiteComponents = sorted(whiteComponents, key=lambda x: x[1], reverse=True)

        # Considerar apenas o maior componente na ROI
        if len(whiteComponents) >= 1:
            whiteComponent1 = whiteComponents[0]

            # Obter as coordenadas dos bounding boxes do maior componente na ROI
            x1 = white_stats[whiteComponent1[0], cv2.CC_STAT_LEFT]
            y1 = white_stats[whiteComponent1[0], cv2.CC_STAT_TOP]
            width1 = white_stats[whiteComponent1[0], cv2.CC_STAT_WIDTH]
            height1 = white_stats[whiteComponent1[0], cv2.CC_STAT_HEIGHT]
        
        # Compara a área do maior componente branco na ROI com a soma das duas áreas vermelhas previamente encontradas
        ## if white_area[whiteComponents[0]] < red_area[redComponents[0]] + red_area[redComponents[1]]:

    # Mostrar a imagem original e a detecção de vermelho
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)

    # Finaliza o programa
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows()
