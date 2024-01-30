#include "utils.h"
#include <opencv2/freetype.hpp>

// 双层车牌分割后拼接
void utils::get_split_merge(cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    cv::Rect roi_upper(0, 0, w, static_cast<int>(5.0/12.0 * h));
    cv::Rect roi_lower(0, static_cast<int>(1.0/3.0 * h), w, h - static_cast<int>(1.0/3.0 * h));

    cv::Mat img_upper = img(roi_upper);
    cv::Mat img_lower = img(roi_lower);

    cv::resize(img_upper, img_upper, img_lower.size());

    cv::hconcat(img_upper, img_lower, img);

}

// 关键点校正图像
cv::Mat utils::warpAffineImage(cv::Mat image, std::vector<cv::Point2d> points)
{
    cv::Mat src = cv::Mat::zeros(4, 2, CV_32F);
    cv::Mat dst = cv::Mat::zeros(4, 2, CV_32F);
    
    cv::Point2d tl = points[0]; // 左上角
    cv::Point2d bl = points[1]; // 左下角
    cv::Point2d br = points[2]; // 右下角
    cv::Point2d tr = points[3]; // 右上角

    // 从左上角开始，逆时针插入
    for (int i = 0; i < points.size(); i++){
        src.at<float>(i, 0) = points[i].x;
        src.at<float>(i, 1) = points[i].y;
    }


    double widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    double widthB = std::sqrt(std::pow(tl.x - tr.x, 2) + std::pow(tl.y - tr.y, 2));
    double maxWidth = std::max((int)widthA, (int)widthB);

    double heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    double heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    double maxHeight = std::max((int)heightA, (int)heightB);

    for (int i = 0; i < 4; i++){
        dst.at<float>(i, 0) = i < 2 ? 0 : maxWidth - 1;
        dst.at<float>(i, 1) = !(i % 3) ? 0 : maxHeight - 1;
    }

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));
    return warped;
}

size_t utils::vectorProduct(const std::vector<int64_t> &vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto &element : vector)
        product *= element;

    return product;
}

std::wstring utils::charToWstring(const char *str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

std::vector<std::string> utils::loadNames(const std::string &path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline(infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

void utils::visualizeDetection(cv::Mat &image, std::vector<Detection> &detections, std::vector<PlateDetection>& detections_plate,
                               const std::vector<std::string> &classNames)
{
    cv::Mat imageCopy = image.clone();
    std::vector<cv::Mat> resultImages;

    // 中文显示
    cv::Ptr<cv::freetype::FreeType2> ft2;
	ft2 = cv::freetype::createFreeType2();
	ft2->loadFontData("../font/platech.ttf", 0);

    for (int i = 0; i < detections.size();i++)
    {
        // 车牌框绘图
        cv::rectangle(image, detections[i].box, cv::Scalar(229, 160, 21), 2);

        // 车牌号和颜色绘图
        int x = detections[i].box.x;
        int y = detections[i].box.y;

        // 车牌类型置信度
        int type_conf = (int)std::round(detections[i].clsConf * 100);
        // 车牌颜色置信度
        int color_conf = (int)std::round(detections_plate[i].conf * 100);
        // 车牌类型
        int classId = detections[i].classId;
        int baseline = 0;

        std::string label1;
        cv::Scalar color{229, 160, 21};
        label1 = "车牌号：" + detections_plate[i].text + "  颜色：" + detections_plate[i].color+ "  置信度: 0." + std::to_string(color_conf);

        if (!detections[i].flag){
            color = cv::Scalar(0, 0, 255);
            label1 = "车牌号无法识别";
        }

        cv::Size size1 = cv::getTextSize(label1, cv::FONT_ITALIC, 0.25, 2, &baseline);
        cv::rectangle(image,
                    cv::Point(x, y - 40), cv::Point(x + size1.width, y),
                    color, -1);

        ft2->putText(image, label1, cv::Point(x, y - 25), 12, 
                CV_RGB(255,255, 255),cv::FILLED, cv::LINE_AA, true);

        // 车牌类型绘图
        std::string label2 = "车牌类型：" + classNames[detections[i].classId] + "  置信度: 0." + std::to_string(type_conf);
        if (type_conf == 100) label2 = "车牌类型：" + classNames[detections[i].classId] + "  置信度: 1.00";
        cv::Size size2 = cv::getTextSize(label2, cv::FONT_ITALIC, 0.25, 2, &baseline);
        cv::rectangle(image,
                      cv::Point(x, y - 20), cv::Point(x + size2.width, y),
                      color, -1);

        ft2->putText(image, label2, cv::Point(x, y - 3), 12, 
                CV_RGB(255,255, 255),cv::FILLED, cv::LINE_AA, true);
    
        // 关键点绘图
        for (const cv::Point2d &point : detections[i].points){
            cv::circle(image, point, 5, cv::Scalar(0, 0, 255), -1);
        }
    }
}

void utils::letterbox(const cv::Mat &image, cv::Mat &outImage,
                      const cv::Size &newShape = cv::Size(640, 640),
                      const cv::Scalar &color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int newUnpad[2]{(int)std::round((float)shape.width * r),
                    (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::scaleCoords(const cv::Size &imageShape, cv::Rect &coords, const cv::Size &imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.height = (int)std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}

// 关键点坐标变换到原图尺寸
std::vector<cv::Point2d> utils::scalePoints(const cv::Size &imageShape, std::vector<cv::Point2f> coords, const cv::Size &imageOriginalShape)
{
    std::vector<cv::Point2d> scaledCoords{coords.size()};
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    for (int i = 0; i < coords.size(); i++){
        scaledCoords[i].x = (int)std::round(((float)(coords[i].x - pad[0]) / gain));
        scaledCoords[i].y = (int)std::round(((float)(coords[i].y - pad[1]) / gain));
    }
    return scaledCoords;
}

void utils::Timer::stop()
{
    endTime = std::chrono::high_resolution_clock::now();
    std::cout << name << "函数的运行时间: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << " ms" << std::endl;
}

template <typename T>
T utils::clip(const T &n, const T &lower, const T &upper)
{
    return std::max(lower, std::min(n, upper));
}
