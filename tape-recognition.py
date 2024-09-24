import cv2 
import numpy as np

vision = True
camera = cv2.VideoCapture(0)

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

    # aplicando a máscara
    redDetection = cv2.bitwise_and(frame, frame, mask=redMask)

    # Encontrar componentes conectados e suas estatísticas
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(redMask)

    # Armazenar as estatísticas (exceto o fundo, que é o label 0)
    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 100:  # Ignorar áreas muito pequenas
            components.append((label, area))

    # Ordenar os componentes por área (do maior para o menor)
    components = sorted(components, key=lambda x: x[1], reverse=True)

    # Considerar apenas os dois maiores componentes
    components = components[:2]

    bounding_boxes = []  # Lista para armazenar as bounding boxes

    # Exibir e desenhar apenas as duas maiores regiões vermelhas
    for (label, area) in components:
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        
        print(f"Componente {label}:")
        print(f"  - Área: {area} pixels")
        print(f"  - Bounding Box: x={x}, y={y}, largura={width}, altura={height}")

        # Desenhar o retângulo delimitador ao redor do componente na imagem original
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Adicionar as coordenadas dos vértices da bounding box à lista
        top_right = (x + width, y)
        bottom_right = (x + width, y + height)
        top_left = (x, y)
        bottom_left = (x, y + height)

        bounding_boxes.append((top_left, top_right, bottom_left, bottom_right))

        # Se houver duas bounding boxes, desenhar as linhas entre os vértices
        if len(bounding_boxes) == 2:

            # Pegar as coordenadas das duas bounding boxes
            box1_top_left, box1_top_right, box1_bottom_left, box1_bottom_right = bounding_boxes[0]
            box2_top_left, box2_top_right, box2_bottom_left, box2_bottom_right = bounding_boxes[1]

            # Desenhar linha entre o vértice superior direito da box 1 e o vértice superior esquerdo da box 2
            cv2.line(frame, box1_top_right, box2_top_left, (0, 255, 255), 2)  # Linha amarela

            # Desenhar linha entre o vértice inferior direito da box 1 e o vértice inferior esquerdo da box 2
            cv2.line(frame, box1_bottom_right, box2_bottom_left, (0, 255, 255), 2)  # Linha amarela

            # Determinar a região de interesse (ROI) entre os dois objetos
            x_roi = min(stats[components[0][0], cv2.CC_STAT_LEFT], stats[components[1][0], cv2.CC_STAT_LEFT])
            y_roi = min(stats[components[0][0], cv2.CC_STAT_TOP], stats[components[1][0], cv2.CC_STAT_TOP])
            width_roi = max(stats[components[0][0], cv2.CC_STAT_LEFT] + stats[components[0][0], cv2.CC_STAT_WIDTH], 
                        stats[components[1][0], cv2.CC_STAT_LEFT] + stats[components[1][0], cv2.CC_STAT_WIDTH]) - x_roi
            height_roi = max(stats[components[0][0], cv2.CC_STAT_TOP] + stats[components[0][0], cv2.CC_STAT_HEIGHT], 
                            stats[components[1][0], cv2.CC_STAT_TOP] + stats[components[1][0], cv2.CC_STAT_HEIGHT]) - y_roi

            # Extrair a área entre os dois componentes (nossa ROI)
            roi = frame[y_roi:y_roi + height_roi, x_roi:x_roi + width_roi]

            # Desenhar retângulos delimitadores ao redor dos dois maiores componentes
            cv2.rectangle(frame, 
                        (stats[components[0][0], cv2.CC_STAT_LEFT], stats[components[0][0], cv2.CC_STAT_TOP]), 
                        (stats[components[0][0], cv2.CC_STAT_LEFT] + stats[components[0][0], cv2.CC_STAT_WIDTH], 
                        stats[components[0][0], cv2.CC_STAT_TOP] + stats[components[0][0], cv2.CC_STAT_HEIGHT]), 
                        (0, 255, 0), 2)

            cv2.rectangle(frame, 
                        (stats[components[1][0], cv2.CC_STAT_LEFT], stats[components[1][0], cv2.CC_STAT_TOP]), 
                        (stats[components[1][0], cv2.CC_STAT_LEFT] + stats[components[1][0], cv2.CC_STAT_WIDTH], 
                        stats[components[1][0], cv2.CC_STAT_TOP] + stats[components[1][0], cv2.CC_STAT_HEIGHT]), 
                        (0, 255, 0), 2)


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

    # Mostrar a imagem original e a detecção de vermelho
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows() 
