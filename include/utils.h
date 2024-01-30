#pragma once
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect box;
    float conf{};
    // 车牌类型,0:单牌,1:双牌
    int classId{};
    // 车配类型置信度
    float clsConf{};
    // 增加关键点
    std::vector<cv::Point2d> points;  
    // 车牌是否有效,0:无效,1:有效
    int flag = 1;
};

struct PlateDetection
{
    // 车牌文字
    std::string text{};
    // 增加车牌颜色
    std::string color{};
    // 颜色置信度
    float conf{};
};

namespace utils
{
    size_t vectorProduct(const std::vector<int64_t> &vector);
    std::wstring charToWstring(const char *str);
    std::vector<std::string> loadNames(const std::string &path);
    void visualizeDetection(cv::Mat &image, std::vector<Detection> &detections, std::vector<PlateDetection> &detections_plate,
                            const std::vector<std::string> &classNames);

    void letterbox(const cv::Mat &image, cv::Mat &outImage,
                   const cv::Size &newShape,
                   const cv::Scalar &color,
                   bool auto_,
                   bool scaleFill,
                   bool scaleUp,
                   int stride);

    void scaleCoords(const cv::Size &imageShape, cv::Rect &box, const cv::Size &imageOriginalShape);
    std::vector<cv::Point2d> scalePoints(const cv::Size &imageShape, std::vector<cv::Point2f> coords, const cv::Size &imageOriginalShape);
    template <typename T>
    T clip(const T &n, const T &lower, const T &upper);

    cv::Mat warpAffineImage(cv::Mat image, std::vector<cv::Point2d> points);
    void get_split_merge(cv::Mat& img);

    /*
    程序运行时间计算，用于调试，使用方法:

        utils::Timer timer{"函数名称，默认为Main function"};

        需要计算时间的代码

        timer.stop();
    */
    class Timer
    {
    public:
        Timer() = default;
        Timer(const std::string &name) : name(name),startTime(std::chrono::high_resolution_clock::now()){};
        ~Timer() = default;
        void stop();

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
        std::string name = "Main function";
    };
}
