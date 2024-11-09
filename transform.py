from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8-SODA.pt')

# 导出ONNX格式
model.export(format='onnx', imgsz=640)  # 设置输入尺寸 (可以自定义)

# 转换ONNX模型为TensorRT FP16精度
# trtexec --onnx=yolov8-SODA.onnx --saveEngine=yolov8_SODA_fp32.trt --fp32 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw