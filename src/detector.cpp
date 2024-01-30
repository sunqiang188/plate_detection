#include "detector.h"
#include <algorithm>
#include <numeric>

// 车牌颜色
std::map<int, std::string> colorFlags = {
    {0, "黑色"},
    {1, "蓝色"}, 
    {2, "绿色"},
    {3, "白色"},
    {4, "黄色"}
};

// 车牌文字
std::map<int, std::string> plateFlags = {
    {0, "#"}, {1, "京"}, {2, "沪"}, {3, "津"}, {4, "渝"}, {5, "冀"},
    {6, "晋"}, {7, "蒙"}, {8, "辽"}, {9, "吉"}, {10, "黑"}, {11, "苏"},
    {12, "浙"}, {13, "皖"}, {14, "闽"}, {15, "赣"}, {16, "鲁"}, {17, "豫"},
    {18, "鄂"}, {19, "湘"}, {20, "粤"}, {21, "桂"}, {22, "琼"}, {23, "川"},
    {24, "贵"}, {25, "云"}, {26, "藏"}, {27, "陕"}, {28, "甘"}, {29, "青"},
    {30, "宁"}, {31, "新"}, {32, "学"}, {33, "警"}, {34, "港"}, {35, "澳"},
    {36, "挂"}, {37, "使"}, {38, "领"}, {39, "民"}, {40, "航"}, {41, "危"},
    {42, "0"}, {43, "1"}, {44, "2"}, {45, "3"}, {46, "4"}, {47, "5"},
    {48, "6"}, {49, "7"}, {50, "8"}, {51, "9"}, {52, "A"}, {53, "B"},
    {54, "C"}, {55, "D"}, {56, "E"}, {57, "F"}, {58, "G"}, {59, "H"},
    {60, "J"}, {61, "K"}, {62, "L"}, {63, "M"}, {64, "N"}, {65, "P"},
    {66, "Q"}, {67, "R"}, {68, "S"}, {69, "T"}, {70, "U"}, {71, "V"},
    {72, "W"}, {73, "X"}, {74, "Y"}, {75, "Z"}, {76, "险"}, {77, "品"}
};

// Softmax转换
std::vector<float> softmax2vec(std::vector<float> &input)
{
    std::vector<float> output;
    float sum = 0;
    for (auto p : input){
        sum += exp(p);
    }
    for (auto p : input){
        output.push_back(exp(p) / (sum + 0.000001));
    }
    return output;
}

// 关键点坐标预处理
cv::Mat order_points(std::vector<cv::Point2f> pts) {
    cv::Mat rect(4, 2, CV_32F);

    // Compute the sum of the points
    std::vector<float> s;
    std::transform(pts.begin(), pts.end(), std::back_inserter(s), [](cv::Point2f pt) {
        return pt.x + pt.y;
    });

    // Find the index of the minimum and maximum sum
    int minSumIdx = std::min_element(s.begin(), s.end()) - s.begin();
    int maxSumIdx = std::max_element(s.begin(), s.end()) - s.begin();

    rect.at<float>(0, 0) = pts[minSumIdx].x;
    rect.at<float>(0, 1) = pts[minSumIdx].y;
    rect.at<float>(2, 0) = pts[maxSumIdx].x;
    rect.at<float>(2, 1) = pts[maxSumIdx].y;

    // Compute the difference of the points
    std::vector<float> diff;
    std::transform(pts.begin(), pts.end(), std::back_inserter(diff), [](cv::Point2f pt) {
        return pt.x - pt.y;
    });

    // Find the index of the minimum and maximum difference
    int minDiffIdx = std::min_element(diff.begin(), diff.end()) - diff.begin();
    int maxDiffIdx = std::max_element(diff.begin(), diff.end()) - diff.begin();

    rect.at<float>(1, 0) = pts[minDiffIdx].x;
    rect.at<float>(1, 1) = pts[minDiffIdx].y;
    rect.at<float>(3, 0) = pts[maxDiffIdx].x;
    rect.at<float>(3, 1) = pts[maxDiffIdx].y;

    return rect;
}

YOLODetector::YOLODetector(const std::string &modelPath,
                           const bool &isGPU = true,
                           const cv::Size &inputSize = cv::Size(640, 640))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "YOLO inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "YOLO inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "YOLO inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    // for (auto shape : inputTensorShape)
    //     std::cout << "Input shape: " << shape << std::endl;

    Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr::pointer ptr_input = input_name_Ptr.release();
    inputNames.push_back(ptr_input);

    Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr::pointer ptr_output = output_name_Ptr.release();
    outputNames.push_back(ptr_output);


    // std::cout << "Input name: " << inputNames[0] << std::endl;
    // std::cout << "Output name: " << outputNames[0] << std::endl;


    this->inputImageShape = cv::Size2f(inputSize);
}

// 最大类别分数
void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int &numClasses,
                                    float &bestConf, int &bestClassId)
{
    // 14和15是类别置信度
    bestClassId = 1;
    bestConf = 0;

    for (int i = 13; i < numClasses + 13; i++){
        if (it[i] > bestConf){
            bestConf = it[i];
            bestClassId = i - 13;
        }
    }
}

void YOLODetector::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLODetector::postprocessing(const cv::Size &resizedImageShape,
                                                    const cv::Size &originalImageShape,
                                                    std::vector<Ort::Value> &outputTensors,
                                                    const float &confThreshold, const float &iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs, cls_confs;
    std::vector<int> classIds;
    //增加关键点坐标
    std::vector<std::vector<cv::Point2f>> points;

    auto *rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // 前13个元素分别为：boxes[4] + obj_conf + points[8]
    int numClasses = (int)outputShape[2] - 13;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float objConf = it[4];
        std::vector<cv::Point2f> point;

        if (objConf > confThreshold){
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            // 4个关键点坐标
            for (int i = 0; i < 4; i++){
                point.push_back(cv::Point2f(it[5 + i * 2], it[6 + i * 2 ]));
            }

            float clsConf;
            int classId;
            this->getBestClassInfo(it, numClasses, clsConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            cls_confs.emplace_back(clsConf);
            classIds.emplace_back(classId);
            points.emplace_back(point);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);

        // 关键点坐标变换
        cv::Mat rect = order_points(points[idx]);
        std::vector<cv::Point2f> scaledPoints;
        scaledPoints.push_back(cv::Point2f(rect.at<float>(0, 0), rect.at<float>(0, 1)));
        scaledPoints.push_back(cv::Point2f(rect.at<float>(1, 0), rect.at<float>(1, 1)));
        scaledPoints.push_back(cv::Point2f(rect.at<float>(2, 0), rect.at<float>(2, 1)));
        scaledPoints.push_back(cv::Point2f(rect.at<float>(3, 0), rect.at<float>(3, 1)));
        det.points = utils::scalePoints(resizedImageShape, scaledPoints, originalImageShape);

        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);
        det.conf = confs[idx];
        det.clsConf = cls_confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat &image, const float &confThreshold = 0.4,
                                            const float &iouThreshold = 0.45)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;
}


PlateDetector::PlateDetector(const std::string &modelPath,
                           const bool &isGPU = true,
                           const cv::Size &inputSize = cv::Size(168, 48))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Plate inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Plate inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Plate inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    // for (auto shape : inputTensorShape)
    //     std::cout << "Input shape: " << shape << std::endl;
    outputDims = static_cast<int>(session.GetOutputCount());
    // std::cout << "output_nodes_num: " << outputDims << std::endl;

    Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr::pointer ptr_input = input_name_Ptr.release();
    inputNames.push_back(ptr_input);

    for (int i = 0; i < outputDims; i++) {
        Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(i, allocator);
        Ort::AllocatedStringPtr::pointer ptr_output = output_name_Ptr.release();
        outputNames.push_back(ptr_output);
    }

    // std::cout << "Input name: " << inputNames[0] << std::endl;
    // std::cout << "Output name: " << outputNames[0] << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

std::vector<PlateDetection> PlateDetector::detect(std::vector<cv::Mat> images)
{
    int batch_size = images.size();
    std::vector<int64_t> inputTensorShape{batch_size, 3, 48, 168};
    std::vector<float> inputTensorValues;
    for (int i = 0; i < batch_size; i++){
        float *blob = nullptr;
        std::vector<int64_t> inputTensorShape_{1, 3, -1, -1};
        this->preprocessing(images[i], blob, inputTensorShape_);
        size_t inputTensorSize_ = utils::vectorProduct(inputTensorShape_);
        inputTensorValues.insert(inputTensorValues.end(), blob, blob + inputTensorSize_);
        delete[] blob;
    }
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));


    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                            inputNames.data(),
                                                            inputTensors.data(),
                                                            1,
                                                            outputNames.data(),
                                                            outputDims);

    // 打印outputTensors
    // for (int i = 0;i < outputDims;i++){
    //     std::vector<int64_t> outputShape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
    //     for (auto shape : outputShape)
    //         std::cout << "Output" << i << " shape: " << shape << std::endl;
    // }
    std::vector<PlateDetection> result = this->postprocessing(outputTensors);
    return result;
}

std::vector<PlateDetection> PlateDetector::detect(cv::Mat &image)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    // for (auto input : inputTensorValues){
    //     std::cout << input << " ";
    // }
    // std::cout << std::endl;

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              outputDims); 
    // 打印outputTensors
    // for (int i = 0;i < outputDims;i++){
    //     std::vector<int64_t> outputShape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
    //     for (auto shape : outputShape)
    //         std::cout << "Output" << i << " shape: " << shape << std::endl;
    // }

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<PlateDetection> result = this->postprocessing(outputTensors);

    delete[] blob;

    return result;
}


void PlateDetector::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    double mean_value = 0.588, std_value = 0.193;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    cv::resize(image, resizedImage, cv::Size(168, 48));
    // utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
    //                  cv::Scalar(114, 114, 114), this->isDynamicInputShape,
    //                  false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);

    floatImage = (floatImage - mean_value) / std_value;
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<PlateDetection> PlateDetector::postprocessing(std::vector<Ort::Value> &outputTensors)
{
    std::vector<PlateDetection> plateDetection;
    colorNums = colorFlags.size();
    // plate_output[0]为车牌文字识别结果，plate_output[1]为车牌颜色识别结果
    std::vector<std::vector<float>> plate_output;
    for (int i = 0;i < outputDims;i++){
        auto *rawOutput = outputTensors[i].GetTensorData<float>();
        std::vector<int64_t> outputShape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
        size_t count = outputTensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output(rawOutput, rawOutput + count);
        plate_output.push_back(output);
    }

    // plate_output[0]: b * 78 * 21, plate_output[1]: b * 5
    int batch_size = plate_output[0].size() / (78 * 21);

    for (int i = 0; i < batch_size;i++){
        PlateDetection plate_detection;

        // 车牌文字识别
        std::vector<int> batch_index;
        for (int j = 0; j < 21; j++)
        {
            int base_index = i * 78 * 21 + j * 78;
            std::vector<float>::iterator maxelement = std::max_element(plate_output[0].begin() + base_index, plate_output[0].begin() + base_index + 78);
            int max_index = std::distance(plate_output[0].begin() + base_index, maxelement);
            batch_index.push_back(max_index);
        }
        std::vector<int> no_repeat_blank_label;
        int pre_c = batch_index[0];
        if (pre_c != 0){
            no_repeat_blank_label.push_back(pre_c);
        }
        for (auto c : batch_index){
            if (pre_c == c || c == 0){
                if (c == 0){
                    pre_c = c;
                }
                continue;
            }
            no_repeat_blank_label.push_back(c);
            pre_c = c;
        }

        for (auto i : no_repeat_blank_label){
            plate_detection.text += plateFlags[i];
        }

        // 车牌颜色识别
        std::vector<float> split_output(colorNums, 0.0);
        for (int k = 0;k < colorNums;k++){
            split_output[k] = plate_output[1][i * colorNums + k];
        }
        std::vector<float> softmax_output = softmax2vec(split_output);

        std::vector<float>::iterator maxelement = std::max_element(std::begin(softmax_output), std::end(softmax_output));
        int color_index = std::distance(std::begin(softmax_output), maxelement);
        // std::cout << "color_index:" << color_index << std::endl;

        plate_detection.color = colorFlags[color_index];
        plate_detection.conf = *maxelement;

        plateDetection.push_back(plate_detection);
    }

    // 打印车牌识别结果
    for (auto i : plateDetection){
        std::cout << "车牌文字: " << i.text << std::endl;
        std::cout << "车牌颜色: " << i.color << std::endl;
    }

    return plateDetection;
}