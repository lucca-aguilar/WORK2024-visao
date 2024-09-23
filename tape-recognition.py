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

    # mostra o que foi reconhecido como vermelho aplicando a mascara
    cv2.imshow('Original', frame)
    cv2.imshow('Detcção de Vermelho', redDetection)

    # finaliza o programa
    if cv2.waitKey(1) == 27:
        break

camera.release() 
cv2.destroyAllWindows() 
