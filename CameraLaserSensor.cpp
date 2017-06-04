//
// Created by Nicholas Hansen on 03/06/2017.
//

#include "CameraLaserSensor.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>


cv::Mat img;        // Camera img
cv::Mat blurImg;    // gray flipped image
cv::Mat invImg;     // Inverted image
cv::Mat bwImg;      // binary image
cv::Mat cdst;       // final image
cv::Mat cannyImg;    // Canny edge image
cv::Mat greenImg;    // Image containing greens
cv::Mat hsvImg;    // HSV color image
cv::Mat errodeImg;
int roi_height;
int height, width;
float x_roi, y_roi, roi_width;
cv::Rect region_of_interest;
cv::Mat top_roi;
cv::Mat bottom_roi;
std::vector<cv::Vec4i> top_lines;
std::vector<cv::Vec4i> bottom_lines;
cv::Point top_center, bottom_center;


void init_images(int width, int height) {
    img = cv::Mat(height, width, IPL_DEPTH_8U, 3);
    hsvImg = cv::Mat(height, width, IPL_DEPTH_8U, 3);
    greenImg = cv::Mat(height, width, IPL_DEPTH_8U, 3);
    invImg = cv::Mat(height, width, IPL_DEPTH_8U, 1);
    blurImg = cv::Mat(height, width, IPL_DEPTH_8U, 1);
    bwImg = cv::Mat(height, width, IPL_DEPTH_8U, 1);
    cannyImg = cv::Mat(height, width, IPL_DEPTH_8U, 1);
    errodeImg = cv::Mat(height, width, IPL_DEPTH_8U, 1);
};

CameraLaserSensor::CameraLaserSensor(int num_of_rois, int pixel_height, int pixel_width)
        : num_of_rois(num_of_rois), pixel_height(pixel_height), pixel_width(pixel_width) {
    // Allocate memory for the distances
    this->distances_top.reserve(num_of_rois / 2);
    this->distances_bottom.reserve(num_of_rois / 2);

    init_images(this->pixel_width, this->pixel_height);
}

// Finds center of line
int find_center(std::vector<cv::Vec4i> lines) {
    if (lines.data()) {
        int center = (lines[0][1] + lines[0][3]) / 2;
        return center;

    } else return -1;
};

int CameraLaserSensor::calculate_distances(cv::Mat &image) {

    std::clock_t start_time = std::clock();
    /*
    // Go through each region of interest for each line
    for (int i = 0; i < this->num_of_rois / 2; i++) {
        this->distances_top[i] = 0;
    }

    int num_of_rois = 20;
    int32_t top_dists[num_of_rois];
    int32_t bottom_dists[num_of_rois];
    int32_t x_center[num_of_rois];
    */
    // Check that image is loaded
    if (image.empty()) { return -1; }
    //img = image;

    // Convert image to HSV
    //cv::cvtColor(img, HSVImg, CV_BGR2HSV);
    //cv::cvtColor(image, hsvImg, CV_BGR2HSV);

    //cv::GaussianBlur(image, HSVImg, cv::Size(3, 3), 2 , 2);

    // Find greens in image
    //cv::inRange(hsvImg, cv::Scalar(80 / 2, 0, 100), cv::Scalar(140 / 2, 255, 255), greenImg);
    cv::inRange(image, cv::Scalar(0, 128, 0), cv::Scalar(128, 255, 128), greenImg);

    // Create Binary image with a threshold value of 128
    cv::threshold(greenImg, bwImg, 1, 255.0, cv::THRESH_BINARY);

    // Erode green lines
    //cv::Mat Kernel(cv::Size(2, 2), CV_8UC1);
    //cv::erode(bwImg, errodeImg, Kernel);
    cv::Mat Kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));;
    cv::dilate(bwImg, errodeImg, Kernel);

    double total_time = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    std::cout << "Calculations took: " << total_time << " seconds" << std::endl;
    std::cout << "Frequency: " << 1.0 / total_time << std::endl;

    cv::Mat downSampeld;
    cv::pyrDown(errodeImg, downSampeld, cv::Size(image.cols / 2, image.rows / 2));
    while (true) {
        cv::imshow("Final Image", downSampeld);
        //cv::imshow("Original Image", image);
        //cv::imshow("Green Image", greenImg);
        //cv::imshow("Green lines Image", bwImg);
        //cv::imshow("Erroded Image", errodeImg);
        char c = cvWaitKey(33); // press escape to quit
        if (c == 27) break;
    }

    /*
    // Invert image
    cv::bitwise_not(bwImg, invImg);

    // Blur image
    cv::GaussianBlur(invImg, blurImg, cv::Size(3, 3), 2, 2);

    // Erode green lines
    cv::Mat Kernel(cv::Size(2, 2), CV_8UC1);
    cv::erode(bwImg, errodeImg, Kernel);

    // Edge detect
    int sobel = 3;
    int lower_thres = 100;
    int upper_thres = 200;
    cv::Canny(errodeImg, cannyImg, lower_thres, upper_thres, sobel);

    cv::cvtColor(cannyImg, cdst, CV_GRAY2BGR);


    cv::Mat bwclone = img.clone();
    cv::cvtColor(img, bwclone, CV_BGR2GRAY);
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp(img.size(), CV_8UC1, cv::Scalar(0)), eroded(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    do {
        cv::erode(bwclone, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(bwclone, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(bwclone);

    } while (countNonZero(bwclone) != 0);
    cv::cvtColor(skel, cdst, CV_GRAY2BGR);


    for (int j = 0; j < num_of_rois; j++) {

        // Set parameters for ROI
        roi_height = height / 2;
        roi_width = width / num_of_rois;
        x_roi = j * roi_width;

        // Set and draw region of interest (TOP)
        region_of_interest = cv::Rect(x_roi, 0, roi_width, roi_height);
        top_roi = skel(region_of_interest);
        cv::rectangle(cdst, region_of_interest, cv::Scalar(0, 0, 255), 1, 8, 0);

        // (BOTTOM)
        region_of_interest = cv::Rect(x_roi, roi_height, roi_width, roi_height);
        bottom_roi = skel(region_of_interest);
        cv::rectangle(cdst, region_of_interest, cv::Scalar(0, 255, 255), 1, 8, 0);

        // Find lines
        cv::HoughLinesP(top_roi, top_lines, 1, CV_PI / 180, 1, width / num_of_rois / 3, 5);
        cv::HoughLinesP(bottom_roi, bottom_lines, 1, CV_PI / 180, 1, width / num_of_rois / 3, 5);

        // Find the center of lines
        try {
            top_center = cv::Point(x_roi + roi_width / 2, find_center(top_lines));
        } catch (cv::Point top_center) {
            std::cout << "Exception Thrown: Top" << std::endl;
        }
        try {
            bottom_center = cv::Point(x_roi + roi_width / 2, find_center(bottom_lines) + roi_height);
        } catch (cv::Point bottom_center) {
            std::cout << "Exception Thrown: Bottom" << std::endl;
        }

        // Calculate the distance
        x_center[j] = -(x_roi + (roi_width - width) / 2);    // -(center_of_roi - center_of_image) : to flip the signage
        top_dists[j] = height / 2 - top_center.y;
        bottom_dists[j] = bottom_center.y - height / 2;

        // Draw Houghlines
        if (top_lines.data()) {
            cv::line(cdst, cv::Point(top_lines[0][0] + x_roi, top_lines[0][1]),
                     cv::Point(top_lines[0][2] + x_roi, top_lines[0][3]), cv::Scalar(0, 0, 255), 3, 8);
        }
        if (bottom_lines.data()) {
            cv::line(cdst, cv::Point(bottom_lines[0][0] + x_roi, bottom_lines[0][1] + roi_height),
                     cv::Point(bottom_lines[0][2] + x_roi, bottom_lines[0][3] + roi_height), cv::Scalar(0, 0, 255), 3,
                     8);
        }

        // Draw Center of lines
        cv::circle(cdst, top_center, 3, cv::Scalar(0, 255, 0), 2);
        cv::circle(cdst, bottom_center, 3, cv::Scalar(0, 255, 0), 2);
        cv::line(cdst, top_center, cv::Point(top_center.x, roi_height), cv::Scalar(0, 255, 255), 1, 8);
        cv::line(cdst, bottom_center, cv::Point(bottom_center.x, roi_height), cv::Scalar(0, 0, 255), 1);

    }

    while (true) {
        cv::imshow("Hough Lines", cdst);
        char c = cvWaitKey(33); // press escape to quit
        if (c == 27) break;
    }
    return 0;
     */


}