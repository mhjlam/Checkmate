#include "calibrator.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


/**
 * @brief Construct a new Calibrator object
 *
 * Initialize the camera matrix and distortion coefficients to default values
 */
Calibrator::Calibrator() {
    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
}

/**
 * @brief Add a new calibration sample, image points and corresponding object points
 *
 * @param image_pts 2D image points, detected corners
 * @param object_pts Corresponding 3D object points, chessboard model
 */
void Calibrator::add_sample(const std::vector<cv::Point2f>& image_pts, const std::vector<cv::Point3f>& object_pts) {
    image_points_.push_back(image_pts);
    object_points_.push_back(object_pts);
}

/**
 * @brief Run camera calibration using all collected samples
 *
 * Use OpenCV calibrateCamera to estimate intrinsic and extrinsic parameters
 * @param image_size Size of the calibration images
 * @return true if calibration is performed, false if not enough data
 */
bool Calibrator::calibrate(const cv::Size& image_size) {
    if (image_points_.empty() || object_points_.empty()) {
        return false; // No data to calibrate
    }
    std::vector<cv::Mat> rvecs, tvecs;
    
    // Calibrate the camera and compute reprojection error
    double err = cv::calibrateCamera(object_points_, image_points_, image_size, camera_matrix_, dist_coeffs_, rvecs, tvecs);
    reproj_error_ = err;
    return true;
}

/**
 * @brief Save the calibration results, camera matrix and distortion coefficients, to a file
 *
 * The file can be loaded later using OpenCV FileStorage
 * @param filename Output filename, YAML or XML supported by OpenCV
 */
void Calibrator::save(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "cameraMatrix" << camera_matrix_;
    fs << "distCoeffs" << dist_coeffs_;
    fs.release();
}

/**
 * @brief Get the camera matrix, intrinsic parameters
 * @return Reference to the 3x3 camera matrix
 */
const cv::Mat& Calibrator::get_camera_matrix() const {
    return camera_matrix_;
}

/**
 * @brief Get the distortion coefficients
 * @return Reference to the distortion coefficients vector
 */
const cv::Mat& Calibrator::get_dist_coeffs() const {
    return dist_coeffs_;
}

/**
 * @brief Get the overall reprojection error from the last calibration
 * @return Reprojection error, lower is better
 */
double Calibrator::get_reproj_error() const {
    return reproj_error_;
}
