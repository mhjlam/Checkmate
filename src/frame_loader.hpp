#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

/**
 * @brief Abstract base class for loading frames from various sources
 *
 * Provide a unified interface for retrieving frames, querying frame size, and number of frames
 */
class FrameLoader {
public:
    virtual ~FrameLoader() = default;

    /**
     * @brief Retrieve the next frame from the source
     * @param frame Output parameter to store the loaded frame, cv::Mat
     * @return true if a frame is successfully loaded, false if no more frames or error
     */
    virtual bool next_frame(cv::Mat& frame) = 0;

    /**
     * @brief Check if the frame source is successfully open
     * @return true if the source is open and ready, false otherwise
     */
    virtual bool is_opened() const = 0;

    /**
     * @brief Get the size of frames provided by the source
     * @return Frame size as cv::Size
     */
    virtual cv::Size get_frame_size() const = 0;

    /**
     * @brief Get the total number of frames available
     * @return Number of frames, or -1 if unknown, for example live camera
     */
    virtual int get_num_frames() const = 0;
};

/**
 * @brief Load frames from live camera devices using OpenCV VideoCapture
 */
class CameraFrameLoader : public FrameLoader {
public:
    /**
     * @brief Construct a CameraFrameLoader for a given device ID
     * @param device_id Camera device index, default 0
     */
    CameraFrameLoader(int device_id = 0);
    ~CameraFrameLoader() override;

    bool next_frame(cv::Mat& frame) override;

    bool is_opened() const override {
        return cap_.isOpened();
    }

    cv::Size get_frame_size() const override {
        return frame_size_;
    }

    int get_num_frames() const override {
        return -1;
    }

private:
    cv::VideoCapture cap_; // OpenCV video capture object
    cv::Size frame_size_;  // Size of captured frames
};

/**
 * @brief Load frames from a directory of image files
 */
class ImageSequenceLoader : public FrameLoader {
public:
    /**
     * @brief Construct an ImageSequenceLoader for a directory of images
     * @param directory Path to the directory containing image files
     */
    ImageSequenceLoader(const std::string& directory);
    /**
     * @brief Retrieve the next image in the sequence
     * @param frame Output parameter to store the loaded image
     * @return true if the image is successfully loaded, false if no more images or error
     */
    bool next_frame(cv::Mat& frame) override;
    /**
     * @brief Check if the image sequence is successfully open
     * @return true if the sequence is open and ready, false otherwise
     */
    bool is_opened() const override { return !filenames_.empty(); }
    /**
     * @brief Get the size of images in the sequence
     * @return Image size as cv::Size
     */
    cv::Size get_frame_size() const override { return frame_size_; }
    /**
     * @brief Get the total number of images in the sequence
     * @return Number of images, or -1 if unknown
     */
    int get_num_frames() const override { return static_cast<int>(filenames_.size()); }
private:
    std::vector<std::string> filenames_; // List of image filenames
    size_t current_idx_ = 0;             // Current index in the sequence
    cv::Size frame_size_;                // Size of images
};
