#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief Namespace containing functions for rendering 3D objects and labels on images
 *
 * Provide utilities to draw axes, cubes, and chessboard labels using camera calibration and pose
 */
namespace Renderer {
    /**
     * @brief Draw 3D axes, X green, Y red, Z blue, on the image using the given camera pose and offset
     * @param image Image on which to draw
     * @param K Camera intrinsic matrix
     * @param dist Camera distortion coefficients
     * @param rvec Rotation vector, Rodrigues
     * @param tvec Translation vector
     * @param offset 3D offset for the axes origin
     */
    void draw_axes(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist, 
                   const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3f offset);
    
    /**
     * @brief Draw a 3D cube in the scene projected onto the image
     * @param image Image on which to draw
     * @param K Camera intrinsic matrix
     * @param dist Camera distortion coefficients
     * @param rvec Rotation vector, Rodrigues
     * @param tvec Translation vector
     * @param base 3D base corner of the cube
     * @param color Color of the cube edges, default white
     */
    void draw_cube(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist, 
                   const cv::Mat& rvec, const cv::Mat& tvec, 
                   cv::Point3f base, cv::Scalar color = cv::Scalar(255,255,255));

    /**
     * @brief Draw chessboard row and column labels, A-H and 1-8, on the image using 3D projection
     * @param image Image on which to draw
     * @param rows Number of chessboard rows
     * @param cols Number of chessboard columns
     * @param square_size Size of each chessboard square
     * @param K Camera intrinsic matrix
     * @param dist Camera distortion coefficients
     * @param rvec Rotation vector, Rodrigues
     * @param tvec Translation vector
     * @param offset 3D offset for the label origin
     */
    void draw_labels(cv::Mat& image, int rows, int cols, float square_size, 
                     const cv::Mat& K, const cv::Mat& dist, 
                     const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3f offset);
}
