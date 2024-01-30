# PlateDetection-onnxruntime

车牌检测+车牌校正+车牌识别（文字和颜色）

#### **支持如下：**

- 1.单行蓝牌
- 2.单行黄牌
- 3.新能源车牌
- 4.白色警用车牌
- 5.教练车牌
- 6.武警车牌
- 7.双层黄牌
- 8.双层白牌
- 9.使馆车牌
- 10.港澳粤Z牌
- 11.双层绿牌
- 12.民航车牌

## Dependecies:
- OpenCV 4.x
- ONNXRuntime 1.7+
- OS: Tested on Windows 10 and Ubuntu 20.04
- CUDA 11+ [Optional]


## Build
```bash
mkdir build
cd build
cmake .. 
cmake --build .
```

## Run
```bas
# On Linux ./yolo_ort --model_path "../models/plate_detect.onnx" --plate_path "../models/plate_rec_color.onnx" --image "../images/Quicker_20220930_180919.png" --class_names "../models/plate.names" --gpu
```


## References

- YOLO v5 repo: https://github.com/ultralytics/yolov5
- YOLOv5 Runtime Stack repo: https://github.com/zhiqwang/yolov5-rt-stack
- ONNXRuntime Inference examples: https://github.com/itsnine/yolov5-onnxruntime 
- 车牌检测训练：https://github.com/we0091234/Chinese_license_plate_detection_recognition
- 车牌识别训练：https://github.com/we0091234/crnn_plate_recognition
