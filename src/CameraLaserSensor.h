//
// Created by Nicholas Hansen on 03/06/2017.
//

#ifndef CAMERALASERSENSOR_CAMERALASERSENSOR_H
#define CAMERALASERSENSOR_CAMERALASERSENSOR_H


#include <opencv2/core/mat.hpp>

class CameraLaserSensor {

    int pixel_width, pixel_height; // Camera image size
    int roi_width = 20, roi_height = 20; // Region Of Interest (ROI) size
    int num_of_rois = 2; // Number of ROIs
    int num_of_laser_lines = 2; // Assume two lasers

    std::vector<float> distances_top, distances_bottom;
    std::vector<cv::Rect> rois_top, rois_bottom;
    std::vector<cv::Vec4i> lines_top, lines_bottom;

    void CalculateROIs();
    void InitializeROIs();

public:
    CameraLaserSensor(int num_of_rois, int pixel_height, int pixel_width);
    void SetROIParameters(int newNumOfROIs, int newROIWidth, int newROIHeight);
    int CalculateDistances(cv::Mat &image);
    void ShowImages();
};


#endif //CAMERALASERSENSOR_CAMERALASERSENSOR_H
