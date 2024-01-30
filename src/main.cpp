#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "cmdline.h"
#include "utils.h"
#include "detector.h"

int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to YOLO onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("plate_path", 'p', "Path to Plate onnx model.", true, "plate_rec_color.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    // bool isGPU = cmd.exist("gpu");
    bool isGPU = false;
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = cmd.get<std::string>("image");
    const std::string modelPath = cmd.get<std::string>("model_path");
    const std::string platePath = cmd.get<std::string>("plate_path");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    // 计算时间
    utils::Timer timer{"Main"};
    YOLODetector detector_plate{nullptr};
    PlateDetector detector_plate_rec{nullptr};
    // 模型初始化
    try {
        std::cout << "Loading model.........." << std::endl;
        detector_plate = YOLODetector{modelPath, isGPU, cv::Size(640, 640)};
        detector_plate_rec = PlateDetector{platePath, isGPU, cv::Size(168, 48)};
        std::cout << "Model loaded..........." << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    // 图像读取
    cv::Mat image = cv::imread(imagePath);
    // 1、车牌检测+关键点识别
    std::vector<Detection> result = detector_plate.detect(image, confThreshold, iouThreshold);

    // 车牌预处理
    cv::Mat cropImage = image.clone();
    std::vector<cv::Mat> cropWarpImages;
    for (Detection &detection : result){
        // 车牌校正
        cv::Mat outImage = utils::warpAffineImage(cropImage, detection.points);
        cv::imwrite("../warp.jpg", outImage);

        // 如果车牌类型为双牌,则进行分割
        if (detection.classId == 1){
            utils::get_split_merge(outImage);
        }
        // 车牌太小则不进行识别
        if (outImage.cols < 10 || outImage.rows < 10)
            detection.flag = 0;
        cropWarpImages.push_back(outImage);
    }

    // 2、车牌识别+颜色识别       
    std::vector<PlateDetection> result_plate = detector_plate_rec.detect(cropWarpImages);

    // 3、车牌识别结果可视化
    utils::visualizeDetection(image, result, result_plate ,classNames);
    timer.stop();

    // 结果显示
    // cv::imshow("result", image);
    cv::imwrite("../result.jpg", image);
    // cv::waitKey(0);

    return 0;
}
