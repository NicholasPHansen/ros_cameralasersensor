#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat img;        // Camera img
Mat blurImg;    // gray flipped image
Mat invImg;     // Inverted image
Mat bwImg;      // binary image
Mat cdst;       // final image
Mat cannyImg;    // Canny edge image
Mat greenImg;    // Image containing greens
Mat HSVImg;    // HSV color image
Mat errodeImg;


int roi_height;
int height, width;
float x_roi, y_roi, roi_width;
Rect region_of_interest;
Mat top_roi;
Mat bottom_roi;
vector<Vec4i> top_lines, bottom_lines;
Point top_center, bottom_center;


// Finds center of line
int find_center(vector<Vec4i> lines) {
    if (lines.data()) {
        int center = (lines[0][1] + lines[0][3]) / 2;
        return center;

    } else return -1;
};

void init_images(int width, int height) {
    img = Mat(height, width, IPL_DEPTH_8U, 3);
    HSVImg = Mat(height, width, IPL_DEPTH_8U, 3);
    greenImg = Mat(height, width, IPL_DEPTH_8U, 3);
    invImg = Mat(height, width, IPL_DEPTH_8U, 1);
    blurImg = Mat(height, width, IPL_DEPTH_8U, 1);
    bwImg = Mat(height, width, IPL_DEPTH_8U, 1);
    cannyImg = Mat(height, width, IPL_DEPTH_8U, 1);
    errodeImg = Mat(height, width, IPL_DEPTH_8U, 1);
};

int find_ranges(Mat &img) {

    int num_of_rois = 20;
    int32_t top_dists[num_of_rois];
    int32_t bottom_dists[num_of_rois];
    int32_t x_center[num_of_rois];


    Size s = img.size();
    width = s.width;
    height = s.height;

    // Check that image is loaded
    if (!img.data) { return -1; }

    // Convert image to HSV
    cvtColor(img, HSVImg, CV_BGR2HSV);

    // Find greens in image
    inRange(HSVImg, Scalar(80 / 2, 0, 100), Scalar(140 / 2, 255, 255), greenImg);

    // Create Binary image with a threshold value of 128
    threshold(greenImg, bwImg, 1, 255.0, THRESH_BINARY);

    // Invert image
    bitwise_not(bwImg, invImg);

    // Blur image
    GaussianBlur(invImg, blurImg, Size(3, 3), 2, 2);

    // Erode green lines
    Mat Kernel(Size(2, 2), CV_8UC1);
    erode(bwImg, errodeImg, Kernel);

    // Edge detect
    int sobel = 3;
    int lower_thres = 100;
    int upper_thres = 200;
    Canny(errodeImg, cannyImg, lower_thres, upper_thres, sobel);

    cvtColor(cannyImg, cdst, CV_GRAY2BGR);


    Mat bwclone = img.clone();
    cv::cvtColor(img, bwclone, CV_BGR2GRAY);
    Mat skel(img.size(), CV_8UC1, Scalar(0));
    Mat temp(img.size(), CV_8UC1, Scalar(0)), eroded(img.size(), CV_8UC1, Scalar(0));
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

    do {
        erode(bwclone, eroded, element);
        dilate(eroded, temp, element);
        subtract(bwclone, temp, temp);
        bitwise_or(skel, temp, skel);
        eroded.copyTo(bwclone);

    } while (countNonZero(bwclone) != 0);
    cvtColor(skel, cdst, CV_GRAY2BGR);


    for (int j = 0; j < num_of_rois; j++) {

        // Set parameters for ROI
        roi_height = height / 2;
        roi_width = width / num_of_rois;
        x_roi = j * roi_width;

        // Set and draw region of interest (TOP)
        region_of_interest = Rect(x_roi, 0, roi_width, roi_height);
        top_roi = skel(region_of_interest);
        rectangle(cdst, region_of_interest, Scalar(0, 0, 255), 1, 8, 0);

        // (BOTTOM)
        region_of_interest = Rect(x_roi, roi_height, roi_width, roi_height);
        bottom_roi = skel(region_of_interest);
        rectangle(cdst, region_of_interest, Scalar(0, 255, 255), 1, 8, 0);

        // Find lines
        HoughLinesP(top_roi, top_lines, 1, CV_PI / 180, 1, width / num_of_rois / 3, 5);
        HoughLinesP(bottom_roi, bottom_lines, 1, CV_PI / 180, 1, width / num_of_rois / 3, 5);

        // Find the center of lines
        try {
            top_center = Point(x_roi + roi_width / 2, find_center(top_lines));
        } catch (Point top_center) {
            cout << "Exception Thrown: Top";
        }
        try {
            bottom_center = Point(x_roi + roi_width / 2, find_center(bottom_lines) + roi_height);
        } catch (Point bottom_center) {
            cout << "Exception Thrown: Bottom";
        }

        // Calculate the distance
        x_center[j] = -(x_roi + (roi_width - width) / 2);    // -(center_of_roi - center_of_image) : to flip the signage
        top_dists[j] = height / 2 - top_center.y;
        bottom_dists[j] = bottom_center.y - height / 2;

        // Draw Houghlines
        if (top_lines.data()) {
            line(cdst, Point(top_lines[0][0] + x_roi, top_lines[0][1]),
                 Point(top_lines[0][2] + x_roi, top_lines[0][3]), Scalar(0, 0, 255), 3, 8);
        }
        if (bottom_lines.data()) {
            line(cdst, Point(bottom_lines[0][0] + x_roi, bottom_lines[0][1] + roi_height),
                 Point(bottom_lines[0][2] + x_roi, bottom_lines[0][3] + roi_height), Scalar(0, 0, 255), 3, 8);
        }

        // Draw Center of lines
        circle(cdst, top_center, 3, Scalar(0, 255, 0), 2);
        circle(cdst, bottom_center, 3, Scalar(0, 255, 0), 2);
        line(cdst, top_center, Point(top_center.x, roi_height), Scalar(0, 255, 255), 1, 8);
        line(cdst, bottom_center, Point(bottom_center.x, roi_height), Scalar(0, 0, 255), 1);

    }

    while (true) {
        cv::imshow("Hough Lines", cdst);
        char c = cvWaitKey(33); // press escape to quit
        if (c == 27) break;
    }
    return 0;
}


int main(int argc, char **argv) {

    init_images(1280, 720);
    img = imread(argv[1]);
    //img = imread("/Users/nhtheswede/repos/openrov/src/laserlines/resources/3_plane.png");
    find_ranges(img);
}
