#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


/**
 * @brief Namespace containing utility functions for image processing and device enumeration
 */
namespace Utils
{
    /**
     * @brief Check if an image is blurred using the Laplacian variance method
     * @param gray Grayscale image to check
     * @param use_camera If true, use a lower threshold suitable for live camera input
     * @return true if the image is considered blurred, false otherwise
     */
    bool is_blurred(const cv::Mat& gray, bool use_camera = false);

    /**
     * @brief Generate a filename with a timestamp, for example prefix_YYYYMMDD_HHMMSS.ext
     * @param prefix Prefix for the filename
     * @param ext File extension, without dot
     * @return Generated filename string
     */
    std::string filename_timestamp(const std::string& prefix, const std::string& ext);

    /**
     * @brief Enumerate available camera devices and retrieve their names
     * @param device_names Output vector to store device names
     * @return Vector of device indices
     */
    std::vector<int> enumerate_camera_devices(std::vector<std::string>& device_names);

    /**
     * @brief Query the Windows system for a camera device friendly name
     * @param device_id Camera device index
     * @return Friendly name string if available, otherwise a default name
     */
    std::string get_device_name(int device_id);

    /**
     * @brief Move the OpenCV window to the center of the primary screen
     * @param window_name Name of the OpenCV window to move
     * @param width Width of the window (in pixels)
     * @param height Height of the window (in pixels)
     */
    void center_opencv_window(const char* window_name, int width, int height);

    /**
     * @brief Focus the OpenCV window to bring it to the foreground
     * @param window_name Name of the OpenCV window to focus
     */
    void focus_opencv_window(const char* window_name);
}
