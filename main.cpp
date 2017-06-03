#include <opencv2/opencv.hpp>
#include "CameraLaserSensor.h"

int main(int argc, char **argv) {

    cv::Mat img = cv::imread(argv[1]);
    CameraLaserSensor *sensor = new CameraLaserSensor(20, img.size().height, img.size().width);
    sensor->calculate_distances(img);

    return 0;
}
