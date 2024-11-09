import time
import torch
import numpy as np
import psutil
import tensorrt as trt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from PIL import Image
from ultralytics import YOLO

dataset_root = 'dataset/SSLAD-2D'
annotations_file = f"{dataset_root}/annotations/val.json"
images_dir = f"{dataset_root}/images/val/"

# Load YOLOv8 model
def load_pytorch_model(model_path):
    model = YOLO(model_path)
    model.eval()
    return model

# Load TensorRT model
def load_trt_model(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# input data
def preprocess_input(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).cuda()  # 加载到GPU
    return image

# load dataset
def load_dataloader(batch_size=8):
    # COCO Format
    dataset = CocoDetection(root=images_dir, annFile=annotations_file,
                            transform=transforms.Compose([
                                transforms.Resize((640, 640)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# Evaluate reasoning accuracy
def evaluate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Evaluate TensorRT accuracy
def evaluate_trt_accuracy(engine, dataloader):
    context = engine.create_execution_context()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.cuda()
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.float32
            bindings.append(torch.empty(size, dtype=dtype, device='cuda'))

        bindings[0].copy_(images.view(-1))

        context.execute_v2(bindings)
        outputs = bindings[1]
        predicted = outputs.view(images.size(0), -1).argmax(dim=1)

        true_labels = [label[0]['category_id'] for label in labels]
        true_labels = torch.tensor(true_labels).cuda()
        total += len(true_labels)
        correct += (predicted == true_labels).sum().item()

    accuracy = correct / total
    return accuracy

# Evaluate reasoning time
def measure_inference_time(model, input_data, num_iterations=100):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_data)
    elapsed_time = (time.time() - start_time) / num_iterations
    return elapsed_time

# Evaluate gpu-memory usage
def measure_gpu_memory(model, input_data):
    torch.cuda.reset_max_memory_allocated()
    with torch.no_grad():
        output = model(input_data)
    memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return memory

# Evaluate TensorRT time
def measure_trt_inference_time(engine, input_data, num_iterations=100):
    context = engine.create_execution_context()
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.float32
        bindings.append(torch.empty(size, dtype=torch.float32, device='cuda'))
    bindings[0].copy_(input_data.contiguous().view(-1))
    start_time = time.time()
    for _ in range(num_iterations):
        context.execute_v2(bindings)
    elapsed_time = (time.time() - start_time) / num_iterations
    return elapsed_time

# Evaluate TensorRT gpu-memory
def measure_trt_gpu_memory(engine, input_data):
    torch.cuda.reset_max_memory_allocated()
    context = engine.create_execution_context()
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.float32
        bindings.append(torch.empty(size, dtype=torch.float32, device='cuda'))
    bindings[0].copy_(input_data.contiguous().view(-1))
    context.execute_v2(bindings)
    memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return memory


def main():
    pytorch_model_path = 'yolov8-SODA.pt'
    trt_model_path = 'yolov8_SODA_fp32.trt'

    pytorch_model = load_pytorch_model(pytorch_model_path).cuda()
    trt_model = load_trt_model(trt_model_path)

    test_image_path = 'D://Desktop//hw4//HT_TEST_000005_SH_011.jpg'
    input_data = preprocess_input(test_image_path)

    dataloader = load_dataloader()
    pytorch_accuracy = evaluate_accuracy(pytorch_model, dataloader)
    print(f"Yolov8 model accuracy: {pytorch_accuracy:.3f}")
    trt_accuracy = evaluate_trt_accuracy(trt_model, dataloader)
    print(f"TensorRT model accuracy: {trt_accuracy:.3f}")

    pytorch_time = measure_inference_time(pytorch_model, input_data)
    trt_time = measure_trt_inference_time(trt_model, input_data)

    print(f"Yolov8 average time: {pytorch_time:.3f} s")
    print(f"TensorRT average time: {trt_time:.3f} s")

    pytorch_memory = measure_gpu_memory(pytorch_model, input_data)
    trt_memory = measure_trt_gpu_memory(trt_model, input_data)

    print(f"Yolov8 gpu-memory: {pytorch_memory:.2f} MB")
    print(f"TensorRT gpu-memory: {trt_memory:.2f} MB")

if __name__ == "__main__":
    main()