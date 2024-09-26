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

    total_red_area = sum([area for _, area in components])  # Soma das áreas vermelhas

    bounding_boxes = []  # Lista para armazenar as bounding boxes

    # Exibir e desenhar apenas as duas maiores regiões vermelhas
    for (label, area) in components:
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

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

            pts = np.array([[box1_top_right, box2_top_left],
                            [box2_top_left, box2_bottom_left],
                            [box2_bottom_left, box1_bottom_right],
                            [box1_bottom_right, box1_top_right]], np.int32)

            # Ajuste no uso da função cv2.fillPoly
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], color=(150, 150, 0))

            # Criando a máscara para reconhecer o branco no polígono
            lower_white = np.array([0, 0, 200])  # Limites para branco
            upper_white = np.array([180, 25, 255])

            # Aplicar a máscara no polígono da imagem original (frame)
            mask_white = cv2.inRange(hsvFrame, lower_white, upper_white)

            # Criar uma máscara para o polígono
            poly_mask = np.zeros_like(mask_white)
            cv2.fillPoly(poly_mask, [pts], 255)

            # Aplicar a máscara do polígono à detecção de branco
            white_detection_mask = cv2.bitwise_and(mask_white, poly_mask)
            white_detection = cv2.bitwise_and(frame, frame, mask=white_detection_mask)

            # Mostrar a detecção de branco no polígono
            cv2.imshow('Detecção de Branco no Polígono', white_detection)

            # Encontrar componentes conectados e suas estatísticas para a área branca
            white_num_labels, white_labels_im, white_stats, white_centroids = cv2.connectedComponentsWithStats(white_detection_mask)

            total_white_area = 0  # Inicializar a soma das áreas brancas

            # Exibir as estatísticas de cada componente branco (exceto o fundo)
            for white_label in range(1, white_num_labels):
                white_area = white_stats[white_label, cv2.CC_STAT_AREA]
                if white_area > 100:  # Ignorar áreas muito pequenas
                    total_white_area += white_area  # Soma das áreas brancas
                    x = white_stats[white_label, cv2.CC_STAT_LEFT]
                    y = white_stats[white_label, cv2.CC_STAT_TOP]
                    width = white_stats[white_label, cv2.CC_STAT_WIDTH]
                    height = white_stats[white_label, cv2.CC_STAT_HEIGHT]
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            # Verificar se a soma das áreas vermelhas é maior ou igual à área branca
            if total_red_area >= total_white_area:
                cv2.putText(frame, "Objeto: Fita detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                '''# Detecção de bordas usando Canny

                edges = cv2.Canny(white_detection_mask, 50, 150)

                # desenhar linhas usando hough lines

                lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

                # Verificar se detectou alguma linha

                if lines is not None:

                    # Desenhar as linhas detectadas

                    for rho, theta in lines[:, 0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        # Desenhar a linha
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)'''

    # Mostrar a imagem original e a detecção de vermelho
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
