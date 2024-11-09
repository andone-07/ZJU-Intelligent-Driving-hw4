# HW4: Acceleration and deployment of autonomous driving target detection model

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github)]()

### Name: Jiawen Zhu
### Student ID: 22451318

This repository stores the assignment 4 code of the Intelligent driving course of Autumn Semester 2024 of the Software School of Zhejiang University.

This project selected YOLOv8 model trained on SODA10M, and converted the model through TensorRT.

## Table of Contents

- [Project Structure](#project-structure)
- [Install](#install)
- [Transform](#transform)
- [Evaluate](#evaluate)
- [Result](#result)
- [Download](#download)


## Project Structure

```md
.
├── dataset/
├── HT_TEST_000005_SH_011.jpg
├── requirements.txt
├── transform.py
├── test.py
├── YOLOv8-SODA.pt
└── YOLOv8_SODA_fp32.trt
```

- `dataset/`: Directory for dataset SODA10M.
- `HT_TEST_000005_SH_011.jpg`: A single image for testing.
- `transform.py`: Convert yolov8.pt through TensorRT.
- `test.py`: Compare model performance before and after conversion.

## Install

This code runs in the conda virtual environment.

You can see the libraries you need in the requirements.txt.

```
# First, you need to install Python and Anaconda3

# Create a virtual environment and activate it
conda create -n your_env_name python=3.10
conda activate your_env_name

# install
pip install -r requirements.txt
```

## Transform

Convert yolov8.pt through TensorRT using transform.py.
```
python transform.py

# Run the following command on the terminal
trtexec --onnx=yolov8-SODA.onnx --saveEngine=yolov8_SODA_fp32.trt --fp32 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
```

## test

Compare model performance before and after conversion

```
python test.py
```

## Result

### Results are as follows:

![result](https://github.com/andone-07/ZJU-Intelligent-Driving-hw4/blob/master/images/result.png)

### The following conclusions can be drawn:

- Inference accuracy decreased slightly.

  Reason: TensorRT performs floating-point precision optimization during model conversion, which may introduce subtle accuracy losses. But these losses usually do not significantly affect inference.

- Reasoning speed is significantly improved.

  Reason: TensorRT is an inference acceleration library optimized for NVIDIA Gpus and specifically designed for high-performance computing. The converted TensorRT model is optimized for GPU architecture, utilizing tensor core and memory optimization techniques for parallel computation.

- GPU memory usage is reduced.

  Reason: TensorRT uses a dedicated memory management strategy during inference to dynamically allocate and reuse the memory required for the computation based on GPU memory, further reducing GPU memory pressure.

## Download

The pre - and post-conversion models can be downloaded at the following link:

[yolov8-SODA.pt](https://github.com/andone-07/ZJU-Intelligent-Driving-hw4/blob/master/yolov8-SODA.pt)

[yolov8_SODA_fp32.trt](https://github.com/andone-07/ZJU-Intelligent-Driving-hw4/blob/master/yolov8_SODA_fp32.trt)
