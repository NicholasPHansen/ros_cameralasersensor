#include <opencv2/opencv.hpp>
#include "CameraLaserSensor.h"

int main(int argc, char **argv) {

    cv::Mat img = cv::imread(argv[1]);
    CameraLaserSensor *sensor = new CameraLaserSensor(2, img.size().height, img.size().width);

    std::clock_t start_time = std::clock();
    sensor->CalculateDistances(img);
    double total_time = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
    std::cout << "Calculations took: " << total_time << " seconds" << std::endl;
    std::cout << "Frequency: " << 1.0 / total_time << std::endl;
    sensor->ShowImages();

    return 0;
}
