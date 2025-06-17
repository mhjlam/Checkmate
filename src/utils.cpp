#include "utils.hpp"

#include <chrono>
#include <format>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dshow.h>
#pragma comment(lib, "strmiids.lib")
#endif


constexpr double BLUR_THRESHOLD = 100.0;        // Default threshold for blur detection
constexpr double BLUR_THRESHOLD_CAMERA = 70.0;  // Lower threshold for camera input


namespace Utils
{

/**
 * @brief Check if an image is blurred using the Laplacian variance method
 * @param gray Grayscale image to check
 * @param use_camera If true, use a lower threshold suitable for live camera input
 * @return true if the image is considered blurred, false otherwise
 *
 * Compute the variance of the Laplacian of the image. If the variance is below a threshold,
 * the image is considered blurred. The threshold is lower for live camera input
 */
bool is_blurred(const cv::Mat& gray, bool use_camera) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);

    // If use_camera is true, use a lower threshold for blur detection
    double threshold = use_camera ? BLUR_THRESHOLD_CAMERA : BLUR_THRESHOLD;
    return (stddev[0] * stddev[0]) < threshold;
}

/**
 * @brief Generate a filename with a timestamp, for example prefix_YYYYMMDD_HHMMSS.ext
 * @param prefix Prefix for the filename
 * @param ext File extension, without dot
 * @return Generated filename string
 *
 * Use the current local time to generate a unique filename
 */
std::string filename_timestamp(const std::string& prefix, const std::string& ext) {
    auto now = std::chrono::system_clock::now();
    auto now_sec = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto local = std::chrono::zoned_time{std::chrono::current_zone(), now_sec};
    std::string timestamp = std::format("{:%Y%m%d_%H%M%S}", local);
    return std::format("{}_{}.{}", prefix, timestamp, ext);
}

/**
 * @brief Enumerate available camera devices and retrieve their names
 * @param device_names Output vector to store device names
 * @return Vector of device indices
 *
 * Try to open camera devices in sequence and collect their indices and names
 */
std::vector<int> enumerate_camera_devices(std::vector<std::string>& device_names) {
    std::vector<int> indices;
    device_names.clear();
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
#if defined(_WIN32)
            device_names.push_back(get_device_name(i));
#else
            device_names.push_back("Device " + std::to_string(i));
#endif
            indices.push_back(i);
            cap.release();
        }
    }
    return indices;
}

/**
 * @brief Query the Windows system for a camera device friendly name
 * @param device_id Camera device index
 * @return Friendly name string if available, otherwise a default name
 *
 * Use DirectShow to enumerate video input devices and retrieve their friendly names
 */
std::string get_device_name(int device_id) {
    std::string name = "Device " + std::to_string(device_id);
#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);

    // Track if we need to uninitialize COM
    bool co_initialized = (hr == S_OK);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        return name;
    }

    ICreateDevEnum* pDevEnum = nullptr;
    IEnumMoniker* pEnum = nullptr;

    // Create the system device enumerator
    hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (void**)&pDevEnum);
    if (FAILED(hr)) {
        if (co_initialized) { CoUninitialize(); }
        return name;
    }

    // Enumerate video input devices
    hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnum, 0);
    if (hr != S_OK) {
        pDevEnum->Release();
        if (co_initialized) { CoUninitialize(); }
        return name;
    }

    ULONG i = 0;
    IMoniker* pMoniker = nullptr;

    // Iterate through all video devices
    while (pEnum->Next(1, &pMoniker, NULL) == S_OK) {
        if ((int)i == device_id) {
            IPropertyBag* pPropBag;

            // Try to get the friendly name from the property bag
            hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pPropBag);
            if (SUCCEEDED(hr)) {
                VARIANT var;
                VariantInit(&var);

                // Read the "FriendlyName" property
                hr = pPropBag->Read(L"FriendlyName", &var, 0);
                if (SUCCEEDED(hr)) {
                    char buf[256];
                    WideCharToMultiByte(CP_UTF8, 0, V_BSTR(&var), -1, buf, sizeof(buf), NULL, NULL);
                    name = buf;
                }

                // Clean up
                VariantClear(&var);
                pPropBag->Release();
            }
        }
        pMoniker->Release();
        ++i;
    }
    pEnum->Release();
    pDevEnum->Release();
    if (co_initialized) { CoUninitialize(); }

#endif
    // Return the friendly name or default name if not found
    return name;
}

void center_opencv_window(const char* window_name, int width, int height) {
#if defined(_WIN32)
    // Get screen size
    int screen_width = GetSystemMetrics(SM_CXSCREEN);
    int screen_height = GetSystemMetrics(SM_CYSCREEN);

    // Calculate top-left position for centering
    int x = (screen_width - width) / 2;
    int y = (screen_height - height) / 2;

    // Use OpenCV's getWindowProperty to retrieve the window handle
    HWND hwnd = (HWND)(uintptr_t)cv::getWindowProperty(window_name, cv::WND_PROP_FULLSCREEN);
    if (hwnd && hwnd != (HWND)-1) {
        MoveWindow(hwnd, x, y, width, height, TRUE);
    }
    else {
        // Fallback: use OpenCV's moveWindow if HWND is not available
        cv::moveWindow(window_name, x, y);
    }
#else
    // Cross-platform fallback: use OpenCV's moveWindow
    int x = 100; // fallback position
    int y = 100;
    cv::moveWindow(window_name, x, y);
#endif
}

void focus_opencv_window(const char* window_name) {
#if defined(_WIN32)
    // Use OpenCV's getWindowProperty to retrieve the window handle
    HWND hwnd = (HWND)(uintptr_t)cv::getWindowProperty(window_name, cv::WND_PROP_FULLSCREEN);

    // If the window handle is valid, bring it to the foreground
    if (hwnd && hwnd != (HWND)-1) {
        SetForegroundWindow(hwnd);
        SetActiveWindow(hwnd);
    }
#endif
}

}
