#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "renderer.hpp"
#include "calibrator.hpp"
#include "chessboard.hpp"
#include "frame_loader.hpp"


constexpr int CORNERS_X = 7;
constexpr int CORNERS_Y = 7;
constexpr int KEY_ESCAPE = 27;
constexpr int REQUIRED_FRAMES = 12;
constexpr float SQUARE_SIZE = 1.0f;
constexpr const char* WINDOW_NAME = "Checkmate";


struct FrameCorners {
    std::vector<cv::Point2f> a1;
    std::vector<cv::Point2f> h8;
};


int main(int argc, char** argv) {
    // Verbose debug option
    bool verbose_debug = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v") {
            verbose_debug = true;
        }
    }

    // Enumerate available input sources (still frames and cameras)
    std::vector<int> available_devices;
    std::vector<std::string> device_names;
    device_names.push_back("Still frames from disk");
    available_devices.push_back(-1); // -1 means still frames

    std::vector<std::string> cam_names;
    std::vector<int> cam_ids = Utils::enumerate_camera_devices(cam_names);
    for (size_t i = 0; i < cam_ids.size(); ++i) {
        available_devices.push_back(cam_ids[i]);
        device_names.push_back(cam_names[i]);
    }

    if (available_devices.empty()) {
        std::cerr << "No camera capture devices found. Please check your hardware." << '\n';
        return -1;
    }

    // Print available input sources
    std::cout << "Available input sources:" << '\n';
    for (size_t idx = 0; idx < available_devices.size(); ++idx) {
        std::cout << "  [" << idx << "] " << device_names[idx] << '\n';
    }

    // Prompt user for input source
    int device_choice = 0;
    std::cout << "Enter input source ID (default 0): ";
    std::string input;
    std::getline(std::cin, input);

    if (!input.empty()) {
        try {
            int user_id = std::stoi(input);
            if (user_id >= 0 && user_id < (int)available_devices.size()) {
                device_choice = user_id;
            }
            else {
                std::cerr << "Invalid input, using default 0 (still frames)." << '\n';
            }
        }
        catch (...) {
            std::cerr << "Invalid input, using default 0 (still frames)." << '\n';
        }
    }

    // Frame loader setup 
    std::unique_ptr<FrameLoader> loader;
    bool use_camera = false;

    // Still frames
    if (available_devices[device_choice] == -1) {
        std::string frames_dir = argc > 1 ? argv[1] : "res/frames";
        loader = std::make_unique<ImageSequenceLoader>(frames_dir);
        if (!loader->is_opened()) {
            std::cerr << "No images found in " << frames_dir << '\n';
            return -1;
        }
        std::cout << "Loaded " << loader->get_num_frames() << " frames from disk." << '\n';
    }
    else { // Camera
        use_camera = true;
        int device_id = available_devices[device_choice];
        loader = std::make_unique<CameraFrameLoader>(device_id);
        if (!loader->is_opened()) {
            std::cerr << "Could not open camera device " << device_id << ". Please check device permissions or try another ID." << '\n';
            return -1;
        }
    }

    // Calibration and chessboard setup
    Chessboard detector(CORNERS_X, CORNERS_Y, SQUARE_SIZE);
    Calibrator calibrator;
    std::vector<FrameCorners> all_frame_corners;
    cv::Mat last_valid_frame;
    int last_valid_corners_idx = -1;
    int frame_count = 0;

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

    // Center the OpenCV window on the screen
    Utils::center_opencv_window(WINDOW_NAME, 1280, 720);

    // Show a blank frame to ensure the window is created and can be focused
    cv::Mat blank_frame = cv::Mat::zeros(720, 1280, CV_8UC3);
    cv::imshow(WINDOW_NAME, blank_frame);
    Utils::focus_opencv_window(WINDOW_NAME);

    // Lambda for drawing overlays (axes, labels, etc.)
    auto draw_overlays = [](cv::Mat& img, const cv::Mat& K, const cv::Mat& dist, const cv::Mat& rvec, const cv::Mat& tvec, float square_size) {
        cv::Point3f outer_corner_offset(-square_size, -square_size, 0);
        Renderer::draw_axes(img, K, dist, rvec, tvec, outer_corner_offset);
        Renderer::draw_labels(img, CORNERS_Y, CORNERS_X, square_size, K, dist, rvec, tvec, outer_corner_offset);
    };

    // Lambda for creating a default camera matrix
    auto make_camera_matrix = [](int width, int height) {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = 1000.0;
        K.at<double>(1, 1) = 1000.0;
        K.at<double>(0, 2) = width / 2.0;
        K.at<double>(1, 2) = height / 2.0;
        return K;
    };

    // Lambda for drawing error overlays
    auto draw_error = [](cv::Mat& frame, const std::string& msg, cv::Point pos, cv::Scalar color) {
        cv::putText(frame, msg, pos, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    };

    // Lambda for finding the best pose (PnP) and corner order
    auto find_best_pose = [&](const std::vector<cv::Point2f>& corners, const cv::Mat& frame, const cv::Mat& gray, 
                              std::vector<double>& outer_vals, std::vector<cv::Point2f>& best_corners, 
                              cv::Mat& best_rvec, cv::Mat& best_tvec, int& best_a1, double& best_reproj_err)
    {
        std::vector<int> a1_candidates = {0, 1, 2, 3};

        int rows = CORNERS_Y;
        int cols = CORNERS_X;

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
        
        outer_vals.assign(4, 1e6);
        for (int i = 0; i < 4; ++i) {
            auto pt = outer_corners[i];
            if (pt.x < 2 || pt.y < 2 || pt.x > gray.cols - 3 || pt.y > gray.rows - 3) {
                continue;
            }
            cv::Rect roi(pt - cv::Point2f(2,2), cv::Size(5,5));
            double val = cv::mean(gray(roi))[0];
            outer_vals[i] = val;
        }

        best_reproj_err = 1e9;
        best_a1 = -1;
        best_corners.clear();
        best_rvec.release();
        best_tvec.release();

        double max_reproj_err = 8.0;
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

        K.at<double>(0, 0) = 1000.0;
        K.at<double>(1, 1) = 1000.0;
        K.at<double>(0, 2) = frame.cols / 2.0;
        K.at<double>(1, 2) = frame.rows / 2.0;

        cv::Mat dist_coeffs = cv::Mat::zeros(5,1,CV_64F);

        for (int a1_index : a1_candidates) {
            std::vector<cv::Point2f> test_corners = corners;
            
            detector.reorder_corners(test_corners, a1_index);
            auto obj_pts = detector.generate_object_points();

            cv::Mat rvec, tvec;
            bool pnp_ok = cv::solvePnP(obj_pts, test_corners, K, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            
            bool z_ok = pnp_ok && (R.at<double>(2,2) > 0);
            double reproj_err = 1e9;
            
            if (z_ok) {
                std::vector<cv::Point2f> proj_pts;
                cv::projectPoints(obj_pts, rvec, tvec, K, dist_coeffs, proj_pts);
                double err = 0.0;

                for (size_t i = 0; i < proj_pts.size(); ++i) {
                    err += cv::norm(proj_pts[i] - test_corners[i]);
                }

                reproj_err = err / proj_pts.size();
                if (reproj_err > 15.0) {
                    z_ok = false;
                }
            }

            if (verbose_debug) {
                std::cout << "A1 candidate " << a1_index
                          << ": pixel value=" << outer_vals[a1_index]
                          << ", solvePnP=" << (pnp_ok ? "true" : "false")
                          << ", z_outwards=" << (z_ok ? "true" : "false")
                          << ", reprojErr=" << reproj_err << (z_ok ? " (OK)" : " (FAIL)") << '\n';
            }

            if (z_ok && reproj_err < best_reproj_err) {
                best_reproj_err = reproj_err;
                best_a1 = a1_index;
                best_corners = test_corners;
                best_rvec = rvec.clone();
                best_tvec = tvec.clone();
            }
        }
    };

    // Frame processing
    cv::Mat frame;
    while (frame_count < (use_camera ? REQUIRED_FRAMES : loader->get_num_frames()) && loader->next_frame(frame)) {
        // Store a clean copy of the frame before any overlays
        cv::Mat clean_frame = frame.clone();
        bool accepted = false;
        bool show_error = false;
        std::string error_msg;
        cv::Scalar error_color;

        // Show how many frames are still required (overlay on preview) only in camera mode
        if (use_camera && frame_count < REQUIRED_FRAMES) {
            int frames_left = REQUIRED_FRAMES - frame_count;
            std::string frame_msg = "Frames left: " + std::to_string(frames_left);
            cv::putText(frame, frame_msg, {30, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Early continue if blurred
        if (Utils::is_blurred(gray, use_camera)) {
            show_error = true;
            error_msg = "Frame is blurred";
            error_color = cv::Scalar(0,0,255);
        }
        else {
            // Find chessboard corners
            std::vector<cv::Point2f> corners;
            if (!detector.find_corners(frame, corners)) {
                show_error = true;
                error_msg = "Chessboard not found";
                error_color = cv::Scalar(0,255,255);
            }
            else {
                // Try all possible A1 corners and find the best pose
                std::vector<double> outer_vals;
                std::vector<cv::Point2f> best_corners;
                cv::Mat best_rvec, best_tvec;
                int best_a1 = -1;
                double best_reproj_err = 1e9;

                // Find the best pose and corner ordering
                find_best_pose(corners, frame, gray, outer_vals, best_corners, best_rvec, best_tvec, best_a1, best_reproj_err);

                // If no valid pose was found, draw an error message
                if (best_a1 == -1) {
                    show_error = true;
                    error_msg = "Pose not valid";
                    error_color = cv::Scalar(0,165,255);
                }
                else {
                    // Accept this frame for calibration
                    auto obj_pts = detector.generate_object_points();
                    calibrator.add_sample(best_corners, obj_pts);
                    last_valid_frame = clean_frame.clone(); // Use the clean frame without overlays
                    accepted = true;

                    // Store both A1 and H8 corner orderings for visualization
                    std::vector<cv::Point2f> a1_corners = best_corners;
                    std::vector<cv::Point2f> h8_corners = best_corners;
                    detector.reorder_corners(h8_corners, 3 - best_a1);
                    all_frame_corners.push_back({a1_corners, h8_corners});
                    last_valid_corners_idx = (int)all_frame_corners.size() - 1;

                    // Draw chessboard grid
                    cv::drawChessboardCorners(frame, cv::Size(CORNERS_X, CORNERS_Y), best_corners, true);

                    // Draw axes and labels
                    cv::Mat K = make_camera_matrix(frame.cols, frame.rows);
                    cv::Mat dist_coeffs = cv::Mat::zeros(5,1,CV_64F);
                    draw_overlays(frame, K, dist_coeffs, best_rvec, best_tvec, SQUARE_SIZE);

                    if (verbose_debug) {
                        std::cout << "Accepted for calibration. Reprojection error: " << best_reproj_err << " (max 8.0)" << '\n';
                    }
                }
            }
        }

        // Draw error if needed
        if (show_error) {
            draw_error(frame, error_msg, {30,30}, error_color);
        }

        // Always show the frame for smooth camera updates
        cv::imshow(WINDOW_NAME, frame);
        if (use_camera && frame_count == 0) {
            Utils::focus_opencv_window(WINDOW_NAME);
        }

        // Use a very short wait for smooth updates
        int key = cv::waitKey(1);
        if (key == KEY_ESCAPE) {
            std::cout << "Exiting..." << '\n';
            break;
        }

        // Only increment frame count for still frames, or for accepted frames in camera mode
        if (!use_camera || (use_camera && accepted)) {
            ++frame_count;

            // Pause for 1 second to let the user observe the overlay
            int pauseKey = cv::waitKey(1000);
            if (pauseKey == KEY_ESCAPE) {
                std::cout << "Exiting..." << '\n';
                break;
            }
        }
    }

    // Calibration and final overlay visualization
    if (calibrator.calibrate(loader->get_frame_size())) {
        // Save calibration results to file
        std::string calibration_filename = Utils::filename_timestamp("calibration", "yml");
        calibrator.save(calibration_filename);
        std::cout << "Calibration saved as " << calibration_filename << std::endl;

        // If there are valid corners, draw the final visualization
        if (last_valid_corners_idx != -1) {
            // Generate the object points for the chessboard corners
            cv::Mat rvec, tvec;
            Chessboard detector(CORNERS_X, CORNERS_Y, SQUARE_SIZE);
            auto obj_pts = detector.generate_object_points();

            // Use the last valid corners for visualization
            FrameCorners vis_corners = all_frame_corners[last_valid_corners_idx];

            // If the pose is valid, draw the axes and cubes
            cv::Mat out_frame = last_valid_frame.clone();
            if (cv::solvePnP(obj_pts, vis_corners.a1, calibrator.get_camera_matrix(), calibrator.get_dist_coeffs(), rvec, tvec)) {
                // Draw axes and labels
                draw_overlays(out_frame, calibrator.get_camera_matrix(), calibrator.get_dist_coeffs(), rvec, tvec, SQUARE_SIZE);

                // Draw the white cube at E1
                cv::Point3f e1_3d(-SQUARE_SIZE, 3 * SQUARE_SIZE, 0);
                Renderer::draw_cube(out_frame, calibrator.get_camera_matrix(), calibrator.get_dist_coeffs(), rvec, tvec, e1_3d, cv::Scalar(255,255,255));

                // Draw the black cube at E8
                cv::Point3f e8_3d = e1_3d + cv::Point3f(7 * SQUARE_SIZE, 0, 0);
                Renderer::draw_cube(out_frame, calibrator.get_camera_matrix(), calibrator.get_dist_coeffs(), rvec, tvec, e8_3d, cv::Scalar(0,0,0));

                // Project the origin point (0,0,0) in 3D space to 2D
                std::vector<cv::Point2f> origin2d;
                std::vector<cv::Point3f> origin3d = {cv::Point3f(-SQUARE_SIZE, -SQUARE_SIZE, 0)};
                cv::projectPoints(origin3d, rvec, tvec, calibrator.get_camera_matrix(), calibrator.get_dist_coeffs(), origin2d);

                // Draw the origin point (0,0) in 3D space
                cv::circle(out_frame, origin2d[0], 10, cv::Scalar(0,0,255), -1);

                // Draw text
                cv::putText(out_frame, "Chessboard base", {30,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,255,255), 2);

                // Save the final frame
                std::string filename = Utils::filename_timestamp("final_frame", "png");
                cv::imwrite(filename, out_frame);
                std::cout << "Final frame saved as " << filename << std::endl;

                // Show the final frame until user exits
                cv::imshow(WINDOW_NAME, out_frame);
                Utils::focus_opencv_window(WINDOW_NAME);
                // Wait until user exits
                while (true) {
                    int key = cv::waitKey(0);
                    if (key == KEY_ESCAPE || key == 'q') {
                        break;
                    }
                }
            }
        }
    }

    return 0;
}
