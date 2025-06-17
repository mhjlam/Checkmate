#include "renderer.hpp"

#include <string>
#include <vector>


constexpr const float CUBE_SCALE = 1.0f;


namespace Renderer {

/**
 * @brief Draw 3D axes, X green, Y red, Z blue, on the image using the given camera pose and offset
 * @param image Image on which to draw
 * @param K Camera intrinsic matrix
 * @param dist Camera distortion coefficients
 * @param rvec Rotation vector, Rodrigues
 * @param tvec Translation vector
 * @param offset 3D offset for the axes origin
 *
 * Project the axes from 3D to 2D using the camera parameters and draw in color
 */
void draw_axes(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist, const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3f offset) {
    // Define the endpoints of the axes in 3D (relative to offset)
    std::vector<cv::Point3f> axes = {
        {0.0f, 0.0f, 0.0f}, // origin
        {0.0f, 4.0f, 0.0f}, // Y axis
        {4.0f, 0.0f, 0.0f}, // X axis
        {0.0f, 0.0f,-4.0f}  // Z axis (negative for OpenCV)
    };

    // Apply offset to all points
    for (auto& a : axes) { 
        a += offset;
    }

    // Project 3D points to 2D image
    std::vector<cv::Point2f> proj;
    cv::projectPoints(axes, rvec, tvec, K, dist, proj);

    // Draw the axes lines
    cv::line(image, proj[0], proj[1], cv::Scalar(0,0,255), 2);  // Y: red
    cv::line(image, proj[0], proj[2], cv::Scalar(0,255,0), 2);  // X: green
    cv::line(image, proj[0], proj[3], cv::Scalar(255,0,0), 2);  // Z: blue
}

/**
 * @brief Draw a 3D cube in the scene projected onto the image
 * @param image Image on which to draw
 * @param K Camera intrinsic matrix
 * @param dist Camera distortion coefficients
 * @param rvec Rotation vector, Rodrigues
 * @param tvec Translation vector
 * @param base 3D base corner of the cube
 * @param color Color of the cube edges, default white
 *
 * Define the cube by 8 corners and project to 2D for rendering
 */
void draw_cube(cv::Mat& image, const cv::Mat& K, const cv::Mat& dist, 
               const cv::Mat& rvec, const cv::Mat& tvec, 
               cv::Point3f base, cv::Scalar color)
{
    // Define the 8 corners of the cube in 3D
    std::vector<cv::Point3f> cube_pts = {
        {base.x, base.y, base.z},
        {base.x, base.y+CUBE_SCALE, base.z},
        {base.x+CUBE_SCALE, base.y, base.z},
        {base.x, base.y, base.z-CUBE_SCALE},
        {base.x+CUBE_SCALE, base.y+CUBE_SCALE, base.z},
        {base.x, base.y+CUBE_SCALE, base.z-CUBE_SCALE},
        {base.x+CUBE_SCALE, base.y, base.z-CUBE_SCALE},
        {base.x+CUBE_SCALE, base.y+CUBE_SCALE, base.z-CUBE_SCALE}
    };

    // Project 3D cube corners to 2D image
    std::vector<cv::Point2f> proj;
    cv::projectPoints(cube_pts, rvec, tvec, K, dist, proj);

    // Lambda to draw a line between two projected points
    auto draw = [&](int i, int j) {
        cv::line(image, proj[i], proj[j], color, 2);
    };

    // Draw cube edges (bottom, top, sides, verticals)
    draw(0,1); draw(1,4); draw(4,2); draw(2,0); // bottom face
    draw(3,5); draw(5,7); draw(7,6); draw(6,3); // top face
    draw(0,3); draw(1,5); draw(2,6); draw(4,7); // verticals
}

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
 *
 * Project 3D label positions to 2D and draw text labels for rows and columns
 */
void draw_labels(cv::Mat& image, int rows, int cols, float square_size, 
                 const cv::Mat& K, const cv::Mat& dist, 
                 const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3f offset)
{
    // Draw row labels (A, B, ...)
    for (int y = 0; y <= rows; ++y) {
        std::vector<cv::Point2f> pt2d;
        std::vector<cv::Point3f> pt3d = { offset + cv::Point3f(-0.5f * square_size, (y + 0.5f) * square_size, 0) };
        cv::projectPoints(pt3d, rvec, tvec, K, dist, pt2d);
        cv::putText(image, std::string(1, ('A' + y)), pt2d[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    }

    // Draw column labels (1, 2, ...)
    for (int x = 0; x <= cols; ++x) {
        std::vector<cv::Point2f> pt2d;
        std::vector<cv::Point3f> pt3d = { offset + cv::Point3f((x + 0.5f) * square_size, -0.5f * square_size, 0) };
        cv::projectPoints(pt3d, rvec, tvec, K, dist, pt2d);
        cv::putText(image, std::to_string(x+1), pt2d[0], cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    }
}

} // namespace Renderer
