#include "frame_loader.hpp"

#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>


/**
 * @brief Construct a CameraFrameLoader for a given device ID
 *
 * Open the camera and store the frame size if successful
 */
CameraFrameLoader::CameraFrameLoader(int device_id) {
    // Try to open the camera using the default backend
    if (!cap_.isOpened()) {
#if defined(_WIN32)
        cap_.open(device_id, cv::CAP_DSHOW);
#else
        cap_.open(device_id);
#endif
    }

    if (cap_.isOpened()) {
        // Store the frame size
        frame_size_ = cv::Size(
            static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
    }
}

/**
 * @brief Destructor, release the camera resource if still open
 */
CameraFrameLoader::~CameraFrameLoader() {
    // Release the camera resource if still open
    if (cap_.isOpened()) {
        cap_.release();
    }
}

/**
 * @brief Grab the next frame from the camera
 * @param frame Output parameter to store the captured frame
 * @return true if a valid frame is captured, false otherwise
 */
bool CameraFrameLoader::next_frame(cv::Mat& frame) {
    if (!cap_.isOpened()) {
        return false; // Camera not available
    }

    cap_ >> frame; // Capture a frame
    return !frame.empty(); // Return true if a valid frame is captured
}


/**
 * @brief Construct an ImageSequenceLoader for a directory of images
 * @param directory Path to the directory containing image files
 *
 * Collect all regular files in the directory, sort them, and determine frame size from the first image
 */
ImageSequenceLoader::ImageSequenceLoader(const std::string& directory) : current_idx_{0} {
    // Collect all regular files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            filenames_.push_back(entry.path().string());
        }
    }

    // Sort filenames for consistent order
    std::sort(filenames_.begin(), filenames_.end());

    // If there are images, determine the frame size from the first image
    if (!filenames_.empty()) {
        cv::Mat img = cv::imread(filenames_[0]);
        if (!img.empty()) {
            frame_size_ = img.size();
        }
    }
}

/**
 * @brief Retrieve the next image in the sequence
 * @param frame Output parameter to store the loaded image
 * @return true if the image is successfully loaded, false if no more images or error
 */
bool ImageSequenceLoader::next_frame(cv::Mat& frame) {
    if (current_idx_ >= filenames_.size()) {
        return false; // No more images
    }

    frame = cv::imread(filenames_[current_idx_++]);
    return !frame.empty(); // Return true if the image is loaded successfully
}
