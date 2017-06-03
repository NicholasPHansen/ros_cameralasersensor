//
// Created by Nicholas Hansen on 03/06/2017.
//

#include "CameraLaserSensor.h"
#include "opencv2/imgproc/imgproc.hpp"

CameraLaserSensor::CameraLaserSensor(int num_of_rois, int pixel_height, int pixel_width)
        : num_of_rois(num_of_rois), pixel_height(pixel_height), pixel_width(pixel_width) {
    // Allocate memory for the distances
    this->distances_top.reserve(num_of_rois / 2);
    this->distances_bottom.reserve(num_of_rois / 2);
}

void CameraLaserSensor::calculate_distances(cv::Mat &image) {

    // Go through each region of interest for each line
    for (int i = 0; i < this->num_of_rois / 2; i++) {
        this->distances_top[i] = 0;
    }
}