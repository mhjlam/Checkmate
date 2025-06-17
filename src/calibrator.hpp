#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


/**
 * @class Calibrator
 * @brief Handle camera calibration logic: collect calibration points, run calibration, and save results
 *
 * Accumulate 2D-3D point correspondences from chessboard detections
 * Run OpenCV camera calibration and provide access to the resulting camera matrix,
 * distortion coefficients, and reprojection error
 */
class Calibrator {
public:
    /**
     * @brief Construct a new Calibrator object
     *
     * Initialize the camera matrix and distortion coefficients
     */
    Calibrator();

    /**
     * @brief Add a new calibration sample
     * @param image_pts 2D image points, detected corners
     * @param object_pts Corresponding 3D object points, chessboard model
     */
    void add_sample(const std::vector<cv::Point2f>& image_pts, const std::vector<cv::Point3f>& object_pts);

    /**
     * @brief Run camera calibration using all collected samples
     * @param image_size Size of the calibration images
     * @return true if calibration is performed, false if not enough data
     */
    bool calibrate(const cv::Size& image_size);

    /**
     * @brief Save the calibration results, camera matrix and distortion coefficients, to a file
     * @param filename Output filename, YAML or XML supported by OpenCV
     */
    void save(const std::string& filename) const;
    
    /**
     * @brief Get the camera matrix, intrinsic parameters
     * @return Reference to the 3x3 camera matrix
     */
    const cv::Mat& get_camera_matrix() const;

    /**
     * @brief Get the distortion coefficients
     * @return Reference to the distortion coefficients vector
     */
    const cv::Mat& get_dist_coeffs() const;

    /**
     * @brief Get the overall reprojection error from the last calibration
     * @return Reprojection error, lower is better
     */
    double get_reproj_error() const;

private:
    double reproj_error_ = 0.0;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<std::vector<cv::Point3f>> object_points_;
};
