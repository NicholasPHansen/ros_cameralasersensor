//
// Created by Nicholas Hansen on 03/06/2017.
//

#ifndef CAMERALASERSENSOR_CAMERALASERSENSOR_H
#define CAMERALASERSENSOR_CAMERALASERSENSOR_H


#include <opencv2/core/mat.hpp>

class CameraLaserSensor {

    int pixel_height, pixel_width; // image size
    int num_of_rois = 2; // Number of regions of interest
    int num_of_laser_lines = 2; // Assume two lasers

    std::vector<float> distances_top, distances_bottom;
    std::vector<cv::Rect> rois_top, rois_bottom;

    void CalculateROIs();
    void InitializeROIs();

public:
    CameraLaserSensor(int num_of_rois, int pixel_height, int pixel_width);
    int CalculateDistances(cv::Mat &image);
    void ShowImages();
};


#endif //CAMERALASERSENSOR_CAMERALASERSENSOR_H
