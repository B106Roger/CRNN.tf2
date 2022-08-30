# Convolutional Recurrent Neural Network for End-to-End Text Recognition - TensorFlow 2

![TensorFlow version](https://img.shields.io/badge/TensorFlow->=2.3-FF6F00?logo=tensorflow)
![Python version](https://img.shields.io/badge/Python->=3.6-3776AB?logo=python)
[![Paper](https://img.shields.io/badge/paper-arXiv:1507.05717-B3181B?logo=arXiv)](https://arxiv.org/abs/1507.05717)
[![Zhihu](https://img.shields.io/badge/知乎-文本识别网络CRNN—实现简述-blue?logo=zhihu)](https://zhuanlan.zhihu.com/p/122512498)

This is a re-implementation of the CRNN network, build by TensorFlow 2. This repository may help you to understand how to build an End-to-End text recognition network easily. Here is the official [repo](https://github.com/bgshih/crnn) implemented by [bgshih](https://github.com/bgshih).

## Usuage
1. Specify the directory of your dataset in configs/*yml file
2. Use `python ./crnn/train.py --configs CONFIG_PATH --save_dir MODEL_OUTPUT_DIR --ext_bg_ratio RATIO` to train crnn
    - for --ext_bg_ratio, it specify the ratio between foreground and background
    - for --use_stn, it specify the model should include light(tiny)-stn on top of crnn model, use --no_stn to make sure not using light-stn
    - for --train_stn, it specify the stn should whether the light(tiny)-stn use supervise label or not. \
    it only work when --use_stn is true.
    - note that the config file should use configs/DigitForground_final.yml.
3. Use `python ./crnn/train_stn.py --config CONFIG_PATH --save_dir MODEL_OUTPUT_DIR`
    - note that the config file should use configs/DigitBackground_final{16|32}.yml.
4. Use `python ./crnn/merge_model.py --config HEAVY_STN_CONFIG --crnn_weight CRNN_h5_WEIGHT --stn_weight HEAVY_STN_h5_WEIGHT --merge --merge_path OUTPUT_MERGE_MODEL_PATH`
    - note that the output of model would be tensorflow `savemodel` format
5. Use `python ./crnn/tflite_converter.py --weight SAVEMODEL_FORMAT_WEIGHT_PATH --fp32 --fp16 --int8 --export_dir OUTPUT_DIRECTORY --config HEAVY_STN_CONFIG`
    - note that use --fp32, --fp16, --int8 to specify the model you want to export \
        if you only want to export int8 model, then you can just use --int8 \ 
        and drop the other two flags.
6. Use `python ./crnn/eval_full_tflite.py --config HEAVY_STN_CONFIG --crnn_weight TFLITE_WEIGHT`
    - this script would evaluate the performance of tflite model