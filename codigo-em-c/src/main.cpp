#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    // abre a webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Erro ao abrir a webcam." << endl;
        return -1;
    }

    Mat frame, frame_HSV, red_detection, red_mask1, red_mask2, red_labels, red_component_mask1;
    Mat white_mask1, white_detection_roi, white_labels_roi, white_component_mask1;

    int reds_in_roi = 0; // Inicializa a contagem de componentes vermelhos
    int whites_in_roi = 0; // Inicializa a contagem de componentes brancos

    Point top_left, bottom_left, top_right, bottom_right; // Variáveis de escopo mais amplo

    while (true) {
        // lê frame por frame da webcam
        cap.read(frame);
        
        // transforma em espaço de cor HSV
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);

        // cria uma máscara de vermelho
        inRange(frame_HSV, Scalar(0, 100, 100), Scalar(10, 255, 255), red_mask1);   
        inRange(frame_HSV, Scalar(160, 100, 100), Scalar(180, 255, 255), red_mask2); 
        bitwise_or(red_mask1, red_mask2, red_detection); // combina as duas máscaras

        // isola os componentes vermelhos
        int num_red_labels = connectedComponents(red_detection, red_labels, 8, CV_32S);
        int leftest_red_label = -1, rightest_red_label = -1;
        int leftest_red_x = numeric_limits<int>::max();
        int rightest_red_x = numeric_limits<int>::min();
        Rect leftest_red_bounding_box, rightest_red_bounding_box;

        for (int label = 1; label < num_red_labels; label++) {
            // cria uma máscara para cada componente
            red_component_mask1 = (red_labels == label);

            // encontra o contorno do componente
            vector<vector<Point>> red_contours;
            findContours(red_component_mask1, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!red_contours.empty()) {
                // encontra a bounding box do componente
                Rect red_bounding_box = boundingRect(red_contours[0]);

                // verifica a área do componente -> ignora componentes com área menor que 500 pixels
                double area = contourArea(red_contours[0]);
                if (area < 500) {
                    continue;
                }

                // componente é o mais à esquerda?
                if (red_bounding_box.x < leftest_red_x) {
                    leftest_red_x = red_bounding_box.x;
                    leftest_red_label = label;
                    leftest_red_bounding_box = red_bounding_box; 
                }

                // componente é o mais à direita?
                if (red_bounding_box.x + red_bounding_box.width > rightest_red_x) {
                    rightest_red_x = red_bounding_box.x + red_bounding_box.width;
                    rightest_red_label = label;
                    rightest_red_bounding_box = red_bounding_box; 
                }
            }
        }

        // define a roi entre os componentes mais extremos
        if (leftest_red_label != -1 && rightest_red_label != -1) {
            // pontos do polígono (ROI)
            top_left = Point(leftest_red_bounding_box.x + leftest_red_bounding_box.width, leftest_red_bounding_box.y);
            bottom_left = Point(leftest_red_bounding_box.x + leftest_red_bounding_box.width, leftest_red_bounding_box.y + leftest_red_bounding_box.height);
            top_right = Point(rightest_red_bounding_box.x, rightest_red_bounding_box.y);
            bottom_right = Point(rightest_red_bounding_box.x, rightest_red_bounding_box.y + rightest_red_bounding_box.height);

            // vetor que define o polígono
            vector<Point> roi_points = {top_left, bottom_left, bottom_right, top_right};

            // desenha o polígono na imagem
            polylines(frame, roi_points, true, Scalar(255, 0, 0), 2);  

            // Cria a região de interesse (ROI) com base nas extremidades internas das bounding boxes
            Rect roi = Rect(top_left, bottom_right);

            // Garante que a ROI seja válida
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= frame.cols && roi.y + roi.height <= frame.rows) {
                Mat frame_roi = frame(roi);  // Aplica a ROI à imagem original

                // Transforma a ROI para o espaço de cor HSV
                Mat frame_HSV_roi;
                cvtColor(frame_roi, frame_HSV_roi, COLOR_BGR2HSV);

                // cria uma máscara de vermelho dentro da ROI
                Mat red_mask1_roi, red_mask2_roi, red_detection_roi;
                inRange(frame_HSV_roi, Scalar(0, 100, 100), Scalar(10, 255, 255), red_mask1_roi);   
                inRange(frame_HSV_roi, Scalar(160, 100, 100), Scalar(180, 255, 255), red_mask2_roi); 
                bitwise_or(red_mask1_roi, red_mask2_roi, red_detection_roi);

                // Detecta componentes vermelhos na ROI
                reds_in_roi = 0; // Reseta a contagem
                int num_red_labels_roi = connectedComponents(red_detection_roi, red_labels, 8, CV_32S);
                for (int red_label_roi = 1; red_label_roi < num_red_labels_roi; red_label_roi++) {
                    red_component_mask1 = (red_labels == red_label_roi);
                    vector<vector<Point>> red_contours_roi;
                    findContours(red_component_mask1, red_contours_roi, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                    if (!red_contours_roi.empty()) {
                        double red_area_roi = contourArea(red_contours_roi[0]);
                        if (red_area_roi < 500) {
                            continue;
                        }
                    }
                    reds_in_roi++;
                }

                // cria uma máscara de branco na ROI
                Mat white_mask1_roi;
                inRange(frame_HSV_roi, Scalar(0, 0, 200), Scalar(180, 30, 255), white_mask1_roi);   
                white_detection_roi = white_mask1_roi;

                // Detecta componentes brancos dentro da ROI
                whites_in_roi = 0;
                int num_white_labels_roi = connectedComponents(white_detection_roi, white_labels_roi, 8, CV_32S);
                for (int white_label_roi = 1; white_label_roi < num_white_labels_roi; white_label_roi++) {
                    white_component_mask1 = (white_labels_roi == white_label_roi);
                    vector<vector<Point>> white_contours_roi;
                    findContours(white_component_mask1, white_contours_roi, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                    if (!white_contours_roi.empty()) {
                        double white_area_roi = contourArea(white_contours_roi[0]);
                        if (white_area_roi < 500) {
                            continue;
                        }
                    }
                    whites_in_roi++;
                }

                // Exibe a ROI
                imshow("ROI", frame_roi);
            }
        }

        // condição para ser uma fita
        if (whites_in_roi == reds_in_roi + 1) {
            putText(frame, "Virtual Wall detected", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            // desenha um polígono e calcula a distância até esse polígono
        }

        // mostra a imagem da webcam
        imshow("Webcam", frame);

        // sai do loop ao pressionar 'q'
        if (waitKey(30) == 'q') {
            break;
        }
    }

    return 0;
}

/*para executar o código pelo terminal com CMake:
cmake --build build
.\build\Debug\Visao.exe 
*/

