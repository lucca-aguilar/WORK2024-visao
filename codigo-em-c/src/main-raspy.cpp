#include <opencv2/opencv.hpp>
#include <libserialport.h>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    bool vision = true; // controla o loop de visão
    VideoCapture camera(0); // abre a câmera padrão

    // define a área mínima para considerar um componente de "tamanho considerável"
    const int MIN_AREA = 1000; // ajuste conforme necessário

    // kernel para operações morfológicas
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    // Configurar a porta serial para se comunicar com o Arduino
    struct sp_port *port;
    sp_return result = sp_get_port_by_name("/dev/tty1", &port);
    if (result != SP_OK) {
        cerr << "Erro ao encontrar a porta serial" << endl;
        return 1;
    }

    if (sp_open(port, SP_MODE_WRITE) != SP_OK) {
        cerr << "Erro ao abrir a porta serial" << endl;
        return 1;
    }

    sp_set_baudrate(port, 9600);  // Configurar a taxa de transmissão

    while (vision) {
        Mat frame;
        camera >> frame; // lê as imagens da webcam
       
        // converte para o espaço de cor hsv
        Mat hsvFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

        // criando a máscara para reconhecer o vermelho
        Mat mask0, mask1, redMask;
        inRange(hsvFrame, Scalar(0, 130, 70), Scalar(10, 255, 255), mask0);
        inRange(hsvFrame, Scalar(170, 130, 70), Scalar(180, 255, 255), mask1);
        redMask = mask0 + mask1;

        // ajustando a máscara para reconhecer o branco (faixa mais ampla)
        Mat whiteMask;
        inRange(hsvFrame, Scalar(0, 0, 180), Scalar(180, 50, 255), whiteMask);

        // aplicar operações morfológicas para melhorar a detecção
        morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
        morphologyEx(whiteMask, whiteMask, MORPH_CLOSE, kernel);

        // encontrar contornos para vermelho
        vector<vector<Point>> contours_red;
        findContours(redMask, contours_red, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // listas para armazenar os componentes detectados
        vector<Rect> red_components;
        bool white_detected = false;

        // filtrar contornos vermelhos com base na área e guardar as coordenadas
        for (const auto& contour : contours_red) {
            double area = contourArea(contour);
            if (area > MIN_AREA) { // ignorar áreas pequenas
                Rect bounding_rect = boundingRect(contour);
                red_components.push_back(bounding_rect);

                // calcular a centróide do contorno e desenhar no frame
                Moments M = moments(contour);
                if (M.m00 != 0) {
                    int cX = static_cast<int>(M.m10 / M.m00);
                    int cY = static_cast<int>(M.m01 / M.m00);
                }
            }
        }

        // verifica se há qualquer presença de branco
        if (countNonZero(whiteMask) > 0) { // se houver algum pixel branco na máscara
            white_detected = true;
        }

        // verificar se temos pelo menos 3 componentes vermelhos com área considerável
        if (red_components.size() >= 3 && white_detected) {
            // ordenar os componentes vermelhos com base na posição horizontal (eixo x)
            sort(red_components.begin(), red_components.end(), [](const Rect& a, const Rect& b) {
                return a.x < b.x;
            });

            // obter o vermelho mais à esquerda e o mais à direita para definir a roi
            Rect leftmost_red = red_components.front();
            Rect rightmost_red = red_components.back();

            // criar uma roi que engloba todos os componentes vermelhos
            int roi_x = leftmost_red.x;
            int roi_width = (rightmost_red.x + rightmost_red.width) - roi_x;
            int roi_y = leftmost_red.y;
            int roi_height = rightmost_red.y + rightmost_red.height - roi_y;

            // desenhar a roi no frame
            rectangle(frame, Point(roi_x, roi_y), Point(roi_x + roi_width, roi_y + roi_height), Scalar(0, 255, 0), 2);
            // mostrar mensagem de fita zebra detectada
            printf("fita zebra detectada\n");

            // Enviar o caractere 'V' para o Arduino via serial
            const char command = 'V';
            sp_blocking_write(port, &command, 1, 1000); // enviar o caractere 'V'

        } else {
            printf("fita zebra não detectada ou incompleta\n");
        }

        // mostrar a imagem de detecção de vermelho
        Mat redDetection;
        bitwise_and(frame, frame, redDetection, redMask); // aplica a máscara

        // finaliza o programa ao pressionar esc (27)
        if (waitKey(1) == 27) {
            break;
        }
    }

    // Fechar a porta serial ao sair do programa
    sp_close(port);
    sp_free_port(port);

    camera.release();
    destroyAllWindows();
    return 0;
}