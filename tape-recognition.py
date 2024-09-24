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

    bounding_boxes = []  # Lista para armazenar os vértices das bounding boxes

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

        # Armazenar os vértices superior esquerdo e inferior direito
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        bounding_boxes.append((top_left, bottom_right))

    # Se houver dois bounding boxes, desenhar as linhas entre os vértices correspondentes
    if len(bounding_boxes) == 2:
        # Pegar os vértices das duas bounding boxes
        box1_top_left, box1_bottom_right = bounding_boxes[0]
        box2_top_left, box2_bottom_right = bounding_boxes[1]

        # Desenhar a linha entre os vértices superiores esquerdos
        cv2.line(frame, box1_top_left, box2_top_left, (255, 0, 0), 2)  # Linha azul entre os vértices superiores esquerdos

        # Desenhar a linha entre os vértices inferiores direitos
        cv2.line(frame, box1_bottom_right, box2_bottom_right, (0, 255, 255), 2)  # Linha amarela entre os vértices inferiores direitos

    # Mostrar a imagem original e a detecção de vermelho
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)

    # Finaliza o programa ao pressionar ESC (27)
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows()
