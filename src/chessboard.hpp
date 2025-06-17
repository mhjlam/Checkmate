#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

/**
 * @class Chessboard
 * @brief Detect and process chessboard patterns for camera calibration and pose estimation
 *
 * Provide methods to find chessboard corners, reorder them to a canonical orientation,
 * identify the A1 origin corner, and generate 3D object points for calibration
 */
class Chessboard {
public:
    /**
     * @brief Construct a new Chessboard object
     * @param corners_x Number of inner corners along the X direction
     * @param corners_y Number of inner corners along the Y direction
     * @param square_size Physical size of a chessboard square, arbitrary units
     */
    Chessboard(int corners_x, int corners_y, float square_size);

    /**
     * @brief Find chessboard corners in the input frame
     * @param frame Input image, grayscale or color
     * @param corners Output vector of detected 2D corner points
     * @return true if corners are found and refined, false otherwise
     */
    bool find_corners(const cv::Mat& frame, std::vector<cv::Point2f>& corners) const;

    /**
     * @brief Reorder the detected corners so that the specified index is the A1 origin corner
     * @param corners Input/output vector of corners to reorder
     * @param a1_index Index of the outer corner to use as A1, 0=TL, 1=TR, 2=BL, 3=BR
     */
    void reorder_corners(std::vector<cv::Point2f>& corners, int a1_index) const;

    /**
     * @brief Find the index of the A1 origin corner by checking the brightness of the outer squares
     * @param gray Grayscale image
     * @param corners Detected chessboard corners
     * @return Index 0-3 of the darkest outer square, or -1 if not found
     */
    int find_a1_index(const cv::Mat& gray, const std::vector<cv::Point2f>& corners) const;

    /**
     * @brief Find all candidate A1 corners, those with brightness close to the minimum
     * @param gray Grayscale image
     * @param corners Detected chessboard corners
     * @return Indices 0-3 of all outer corners with brightness near the minimum
     */
    std::vector<int> find_a1_candidates(const cv::Mat& gray, const std::vector<cv::Point2f>& corners) const;

    /**
     * @brief Generate the 3D object points for the chessboard corners, Z=0 plane
     * @return Vector of 3D points in chessboard model coordinates
     */
    std::vector<cv::Point3f> generate_object_points() const;

private:
    int corners_x_;
    int corners_y_;

    float square_size_;
};
