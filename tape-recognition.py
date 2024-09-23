import cv2 
import numpy as np

vision = True
camera = cv2.VideoCapture(0)

while vision:
    # le as imagens da webcam
    status, frame = camera.read()

    # espaço de cor
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # criando a mascara para reconhecer o vermelho
    lowerRed = np.array([0, 50, 50], dtype = "uint8")
    upperRed= np.array([10, 255, 255], dtype = "uint8")
    redMask = cv2.inRange(hsvFrame, lowerRed, upperRed)

    # aplicando a mascara
    redDetection = cv2.bitwise_and(frame, frame, mask = redMask)

    # Encontrar componentes conectados e suas estatísticas
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(redMask)

    # Exibir as estatísticas dos componentes conectados (exceto o fundo, que é o label 0)
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        centroid = centroids[label]

        print(f"Componente {label}:")
        print(f"  - Área: {area} pixels")
        print(f"  - Bounding Box: x={x}, y={y}, largura={width}, altura={height}")
        print(f"  - Centroide: {centroid}")

        # Desenhar o retângulo delimitador ao redor do componente na imagem original
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # Desenhar o centroide
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

    # Mostrar a imagem original e a detecção de vermelho
    cv2.imshow('Original', frame)
    cv2.imshow('Detecção de Vermelho', redDetection)

    # finaliza o programa
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows() 
