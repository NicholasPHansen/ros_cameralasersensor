//
// Created by Nicholas Hansen on 03/06/2017.
//

#ifndef LASERLINES_CAMERALASERSENSOR_H
#define LASERLINES_CAMERALASERSENSOR_H


#include <opencv2/core/mat.hpp>

class CameraLaserSensor {

    int pixel_height, pixel_width; // image size
    int num_of_rois; // Number of regions of interest
    int num_of_laser_lines = 2; // Assume two lasers

    std::vector<float> distances_top, distances_bottom;

public:
    CameraLaserSensor(int num_of_rois, int pixel_height, int pixel_width);


    int calculate_distances(cv::Mat &image);
};


#endif //LASERLINES_CAMERALASERSENSOR_H
