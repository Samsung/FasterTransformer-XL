# FasterTransformer XLNet and Transformer-XL
NVIDIA's highly optimized transformer-based XLNet and Transformer-XL implementation for ASR LM written in cuda and supports memory mechanism.
The original NVIDIA's code is missing the most important memory mechanism in XLNet and Transformer-XL. So this project added support to this.
This project also added support for Transformer-XL, an ancestor of XLNet.
As a result of applying it to commercial services, it was confirmed that the performance improved four times compared to the previous one.

### Install dependency
    pip install tensorflow transformers datasets sacremoses

### Make a dummy model

```bash
cd train
python make_xlnet_dummy_model.py
```

### Setup

1. Build the FasterTransformer with C++:
    ```bash
    mkdir -p build
    cd build
    cmake -DSM=75,80 -DCMAKE_BUILD_TYPE=Release ..
    make
    ```
Note: SM is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100).  

#### Verify the correctness  
```bash
cd  examples/tensorflow/xlnet
bash verifyCorrectness.sh # For FP32 model
```
- Args for the verifyCorrectness.sh
```doc
Usage: bash verifyCorrectness_FP32.sh -d data_dir -m model_dir -s npz_dir -e mem_len -g gpu_id -f is_use_fp16
        -d The directory of input data. Default: ./data/STS-B
        -n The data name. Default: sts-b
        -m The directory of the xlnet models. Default: ./data/xlnet_cased_L-12_H-768_A-12
        -s The directory which stores the generated npz files. Default: ./data
        -e memory length
        -g Specify which GPU to use. Default: 0
        -f Specify use float16 or not. 1 means run in float16 mode. Default: 0
```

### Notes
Currently specialized for ASR LM, the invokeRelShiftBd function will be updated for other applications.