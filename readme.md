## Introduction

Anomalib inference with TensorRT (python).

## Requirements
albumentations==1.3.1

anomalib==1.0.0.dev0

anomalib.egg==info

numpy==1.23.1

omegaconf==2.3.0

opencv_python==4.8.1.78

opencv_python_headless==4.8.1.78

pycuda==2023.1

tensorrt==8.5.1.7


## Usage
### step 1
Train and export anomalib models to onnx.

### step 2
Convert onnx to trt engine.

Example:

`trtexec --onnx=efficient_ad.onnx --saveEngine=efficient_ad.engine --minShapes=input:1x3x256x256 --optShapes=input:4x3x256x256 --maxShapes=input:8x3x256x256`

### step 3
Do inference.

Example:

`python inference.py --batchsize 4 --weights weights/efficient_ad.engine --metadata data/metadata_transistor_efficient_ad.json  --input D:/surface_defect_datasets/mvtec_anomaly_detection/transistor/test  --output result --visualize 1 --task segmentation --visualization_mode full --show 0`

`python inference.py --batchsize 1 --weights weights/dfkde.engine --metadata data/metadata_transistor_dfkde.json  --input D:/surface_defect_datasets/mvtec_anomaly_detection/transistor/test  --output result --visualize 1 --task classification --visualization_mode full --show 0`

> Note: some bugs occur when doing inference with dfkde model (batchsize > 1).

## References

https://github.com/openvinotoolkit/anomalib

https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp

## Todo

int8 calibration