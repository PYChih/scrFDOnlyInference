# scrFDOnlyInference
- scrFD is an efficient, high-accuracy face detection approach initially described in Arxiv, accepted by ICLR-2022, and released by InsightFace.
- We export various scrFD models from ONNX to TFLite (float/int) for face and 5-point landmark detection tasks. Additionally, we provide minimal example code for image/video inference.
- Evaluation on the WiderFace dataset is performed using several self-defined protocols.
- We provide error analysis for cases where only rough face detection is necessary, disregarding accurate localization or small face detection.
- Speed tests are conducted on CPU/GPU, and we include metrics such as FLOPs, number of parameters, memory footprint, and runtime memory estimation.

# Environment:
## Model Convertion
- same as the tool: onnx2tf
## Model analysis
- same as the tool: tflite2json2tflite
## Only Inference
- tensorflow(or tflite-runtime)
- python-opencv
## Evaluation and Error Analysis

# TODO:
- [ ] convert(export) official insight's scrfd onnx models to tflite(int/float) format with various input shape
- [ ] decode netout and image/video inference
- [ ] fix quantized error in tflite(int) format
- [ ] flops and input_shape table
- [ ] speed test
- [ ] evaluation on widerface
- [ ] error analysis

# Reference:
- [\[github\]Insightface/detection/scrfd](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [\[github\]onnx2tf](https://github.com/PINTO0309/onnx2tf)
- [\[github\]tflite2json2tflite](https://github.com/PINTO0309/tflite2json2tflite)
- [\[github\]WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation)
- [\[dataset\]WIDER Face](http://shuoyang1213.me/WIDERFACE/)

