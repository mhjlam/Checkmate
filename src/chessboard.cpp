#include "chessboard.hpp"

#include <opencv2/opencv.hpp>


/**
 * @brief Construct a new Chessboard object
 *
 * Store the board size and square size for later use in detection and model generation
 * @param corners_x Number of inner corners along the X direction
 * @param corners_y Number of inner corners along the Y direction
 * @param square_size Physical size of a chessboard square, arbitrary units
 */
Chessboard::Chessboard(int corners_x, int corners_y, float square_size)
    : corners_x_(corners_x), corners_y_(corners_y), square_size_(square_size) {}

/**
 * @brief Find chessboard corners in the input frame, refine corners if found
 *
 * Use OpenCV findChessboardCorners and cornerSubPix for subpixel accuracy
 * @param frame Input image, grayscale or color
 * @param corners Output vector of detected 2D corner points
 * @return true if corners are found and refined, false otherwise
 */
bool Chessboard::find_corners(const cv::Mat& frame, std::vector<cv::Point2f>& corners) const {
    // Try to find the chessboard pattern
    bool found = cv::findChessboardCorners(frame, cv::Size(corners_x_, corners_y_), corners);
    
    if (found) {
        // Convert to grayscale if needed
        cv::Mat gray = frame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }

        // Refine corner locations for subpixel accuracy
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);
        cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1), criteria);
    }
    return found;
}

/**
 * @brief Reorder the detected corners so that the specified index is the A1 origin corner
 *
 * The chessboard can be detected in any orientation. This function reorders the corners
 * so that the specified outer corner, A1, is at the origin and the order matches the model
 * @param corners Input/output vector of corners to reorder
 * @param a1_index Index of the outer corner to use as A1, 0=TL, 1=TR, 2=BL, 3=BR
 */
void Chessboard::reorder_corners(std::vector<cv::Point2f>& corners, int a1_index) const {    
    std::vector<cv::Point2f> ordered;

    // Determine direction of iteration for rows and columns
    int step_x = (a1_index == 0 || a1_index == 2) ? 1 : -1;
    int step_y = (a1_index == 0 || a1_index == 1) ? 1 : -1;
    int start_y = (step_y == 1) ? 0 : corners_y_ - 1;
    int start_x = (step_x == 1) ? 0 : corners_x_ - 1;
    
    // Walk through the grid in the correct order
    for (int y = 0; y < corners_y_; ++y) {
        for (int x = 0; x < corners_x_; ++x) {
            int col = start_x + x * step_x;
            int row = start_y + y * step_y;
            ordered.push_back(corners[row * corners_x_ + col]);
        }
    }

    corners = ordered;
}

/**
 * @brief Find the index of the A1 origin corner by checking the brightness of the outer squares
 *
 * Assume the A1 square is the darkest outer square, usually black in a chessboard
 * @param gray Grayscale image
 * @param corners Detected chessboard corners
 * @return Index 0-3 of the darkest outer square, or -1 if not found
 */
int Chessboard::find_a1_index(const cv::Mat& gray, const std::vector<cv::Point2f>& corners) const {
    int rows = corners_y_;
    int cols = corners_x_;

    // Get the four outer corners
    cv::Point2f tl = corners[0];
    cv::Point2f tr = corners[cols - 1];
    cv::Point2f bl = corners[(rows - 1) * cols];
    cv::Point2f br = corners[rows * cols - 1];
    cv::Point2f dx = (tr - tl) / (cols - 1);
    cv::Point2f dy = (bl - tl) / (rows - 1);
    
    // Compute the four outer square centers
    std::vector<cv::Point2f> outer_corners = {
        tl - dx / 2 - dy / 2,
        tr + dx / 2 - dy / 2,
        bl - dx / 2 + dy / 2,
        br + dx / 2 + dy / 2
    };

    int a1_index = -1;
    double min_val = 1e6;

    // Find the darkest outer square, assume to be A1
    for (int i = 0; i < 4; ++i) {
        auto pt = outer_corners[i];
        if (pt.x < 2 || pt.y < 2 || pt.x > gray.cols - 3 || pt.y > gray.rows - 3) {
            continue;
        }

        cv::Rect roi(pt - cv::Point2f(2,2), cv::Size(5,5));
        double val = cv::mean(gray(roi))[0];

        if (val < min_val) {
            min_val = val;
            a1_index = i;
        }
    }

    return a1_index;
}

/**
 * @brief Find all candidate A1 corners, those with brightness close to the minimum
 *
 * Use for ambiguous cases where multiple corners are similarly dark
 * @param gray Grayscale image
 * @param corners Detected chessboard corners
 * @return Indices 0-3 of all outer corners with brightness near the minimum
 */
std::vector<int> Chessboard::find_a1_candidates(const cv::Mat& gray, const std::vector<cv::Point2f>& corners) const {
    int rows = corners_y_;
    int cols = corners_x_;

    cv::Point2f tl = corners[0];
    cv::Point2f tr = corners[cols - 1];
    cv::Point2f bl = corners[(rows - 1) * cols];
    cv::Point2f br = corners[rows * cols - 1];
    cv::Point2f dx = (tr - tl) / (cols - 1);
    cv::Point2f dy = (bl - tl) / (rows - 1);

    std::vector<cv::Point2f> outer_corners = {
        tl - dx / 2 - dy / 2,
        tr + dx / 2 - dy / 2,
        bl - dx / 2 + dy / 2,
        br + dx / 2 + dy / 2
    };

    std::vector<double> vals(4, 1e6);
    double min_val = 1e6;

    // Measure brightness for each outer square
    for (int i = 0; i < 4; ++i) {
        auto pt = outer_corners[i];

        if (pt.x < 2 || pt.y < 2 || pt.x > gray.cols - 3 || pt.y > gray.rows - 3) {
            continue;
        }

        cv::Rect roi(pt - cv::Point2f(2,2), cv::Size(5,5));
        double val = cv::mean(gray(roi))[0];
        vals[i] = val;
        if (val < min_val) {
            min_val = val;
        }
    }

    // Return all indices within 10 of the minimum value
    std::vector<int> candidates;
    for (int i = 0; i < 4; ++i) {
        if (vals[i] < min_val + 10.0) {
            candidates.push_back(i);
        }
    }

    return candidates;
}

/**
 * @brief Generate the 3D object points for the chessboard corners, Z=0 plane
 *
 * Generate points in row-major order, Z=0 for all points
 * @return Vector of 3D points in chessboard model coordinates
 */
std::vector<cv::Point3f> Chessboard::generate_object_points() const {
    std::vector<cv::Point3f> obj_pts;
    for (int y = 0; y < corners_y_; ++y) {
        for (int x = 0; x < corners_x_; ++x) {
            obj_pts.emplace_back(y * square_size_, x * square_size_, 0.0f);
        }
    }
    return obj_pts;
}
