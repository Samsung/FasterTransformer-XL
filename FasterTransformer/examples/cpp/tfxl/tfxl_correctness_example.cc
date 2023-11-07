/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cnpy.h"
#include "src/fastertransformer/models/tfxl/Tfxl.h"

using namespace fastertransformer;
using namespace std;
#include <iomanip>
#include <cudnn.h>
#include "src/fastertransformer/utils/cuda_utils.h"

//#define __XLNET_DEBUG__

template<typename T>
int tfxlCorrectnessExample(size_t max_batch_size,
                            size_t num_layers,
                            size_t seq_len,
                            size_t d_model,
                            size_t inter_size,
                            size_t head_num,
                            size_t size_per_head,
                            size_t num_token,
                            size_t mem_len,
                            string input_name,
                            string model_name,
                            string check_name,
                            bool allow_gemm_test = false);

/*************** NPZ related operations *****************/
template<typename T>
void printDevPtr(const T* d_cache, int len, int col, char* name, bool print)
{
    T* res = (T*)malloc(sizeof(T) * len);
    cudaMemcpy(res, d_cache, sizeof(T) * len, cudaMemcpyDeviceToHost);

    printf("%s ", name);
    int j = 0;
    int cnt = 0;

    printf("%d:\n", cnt++);
    for (int i = 0; i < len; i++) {
        printf("%f ", (float)res[i]);
        if ((j+1) % col == 0) {
            printf("\n%d:\n", cnt++);
        }
        j = j + 1;
    }
    free(res);
    printf("\n");
}

template<typename T>
void validateInput(T* d_cache, int len, int num_token)
{
    T* res = (T*)malloc(sizeof(T) * len);
    cudaMemcpy(res, d_cache, sizeof(T) * len, cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++) {
        if (res[i] >= num_token) {
            int new_val = res[i] % num_token;

            printf("ERROR : Check input data. wrong input[%d] = %d, num_token = %d, new temp val = %d\n", i, (int)res[i], num_token, new_val);
            res[i] = new_val;
        }
    }
    cudaMemcpy(d_cache, res, sizeof(T) * len, cudaMemcpyHostToDevice);
    free(res);
}

template<typename T>
float castToFloat(T input)
{
    float output = (T)(input);
    return output;
}

template<>
float castToFloat(__half input)
{
    float output = __half2float(input);
    return output;
}

template<typename T>
void setByNpz(cnpy::npz_t& my_npz, std::string name, T* d_ptr, int size, int offset = 0)
{
    auto search2 = my_npz.find(name);
    if (search2 == my_npz.end()) {
        std::cout << "Invalid name : " << name << std::endl;
        return;
    }
    cout << "loading " << name << ", offset " << offset << ", size "<< size << endl;
    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];
    // load it into a new array
    T* loaded_data = arr.data<T>();

    check_cuda_error(cudaMemcpy(d_ptr, loaded_data + offset, sizeof(T) * size, cudaMemcpyHostToDevice));
}
template<>
void setByNpz<__half>(cnpy::npz_t& my_npz, std::string name, __half* d_ptr, int size, int offset)
{
    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    cout << "fp16 " << name << endl;

    // load it into a new array
    float* loaded_data = arr.data<float>();
    __half* half_data = (__half*)malloc(sizeof(__half) * size);

    loaded_data = loaded_data + offset;
    for (int i = 0; i < size; i++) {
        half_data[i] = __float2half_rn(loaded_data[i]);
    }

    check_cuda_error(cudaMemcpy(d_ptr, half_data, sizeof(__half) * size, cudaMemcpyHostToDevice));
    free(half_data);
}

std::string paraName(int i_layer, std::string sub_para)
{
    std::ostringstream s;
    s << "model/transformer/layer_._" << i_layer << sub_para;
    std::string str = s.str();
    return str;
}

std::string paraName(std::string s)
{
    std::string str = s;
    return str;
}

template<typename T>
void checkByNpz(cnpy::npz_t& data_npz, cudaStream_t stream, std::string name, T* d_ptr, int size, int index = -1)
{
    std::string key;

    if( index == -1)    {
        key = name;
    }
    else {
        key = name + "_" + std::to_string(index);
    }
    auto search2 = data_npz.find(key);
    if (search2 == data_npz.end()) {
        std::cout << "Invalid key : " << key << std::endl;
        return;
    }

    bool ifCorrect = 1;
    cnpy::NpyArray arr = data_npz[key];
    float* loaded_data = arr.data<float>();

    T* h_ptr = (T*)malloc(size * sizeof(T));
    check_cuda_error(cudaMemcpyAsync(h_ptr, d_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));

    float err = 0;
    float max = castToFloat(h_ptr[0]);
    int i = 0;

    std::cout << key << " size : " << size << std::endl;

    for (i = 0; i < size; i++) {
        float sub = abs(castToFloat(h_ptr[i]) - loaded_data[i]);
        if (sub > err) {
            err = sub;
        }
        if (max < castToFloat(h_ptr[i])) {
            max = castToFloat(h_ptr[i]);
        }
    }

    std::cout << name << " Max err :" << err << " Max value :" << max << " Ralative error rate: " << err / max
              << std::endl;

    free(h_ptr);
}

template<typename T>
void printNpz(cnpy::npz_t& my_npz, std::string name, int size, int offset = 0)
{
    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];
    // load it into a new array
    T* loaded_data = arr.data<T>();
    for (int i = 0; i < size; i++) {
        cout << loaded_data[i] << " ";
        if (i % 9 == 0) {
            cout << endl;
        }
    }
}
/*************** Main Program *****************/
int main(int argc, char** argv)
{
    string input_name = "./data/data.npz";
    string model_name = "./data/model.npz";
    string check_name = "./data/output.npz";

    if (argc != 14) {
        printf("[ERROR] ./bin/tfxl_correctness_example max_batch_size num_layers seq_len d_model inter_size "
               "head_num size_per_head num_token mem_len input_name model_name check_name "
               "is_fp16\n");
        printf("e.g., ./bin/tfxl_correctness_example 8 12 128 12 64 32000 16 256 512"
               "./data/data.npz ./data/model.npz ./data/output.npz 0\n");
        return 0;
    }
    bool allow_gemm_test = false;

    int max_batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    int num_token = atoi(argv[6]);
    int mem_len = atoi(argv[7]);
    int d_model = atoi(argv[8]);
    int inter_size = atoi(argv[9]);
    input_name = argv[10];
    model_name = argv[11];
    check_name = argv[12];
    bool is_fp16 = atoi(argv[13]);

    cout << " " << max_batch_size << " " << num_layers << " " << seq_len << " " << head_num << " " << size_per_head << " "
         << num_token << " " << mem_len << " " << d_model << " " << input_name << " " << model_name << " " << check_name << " " << is_fp16 << endl;

    if (is_fp16 == 0) {
        return tfxlCorrectnessExample<float>(max_batch_size,
                                              num_layers,
                                              seq_len,
                                              d_model,
                                              inter_size,
                                              head_num,
                                              size_per_head,
                                              num_token,
                                              mem_len,
                                              input_name,
                                              model_name,
                                              check_name,
                                              allow_gemm_test);
    }
    else if (is_fp16 == 1) {
        return tfxlCorrectnessExample<half>(max_batch_size,
                                             num_layers,
                                             seq_len,
                                             d_model,
                                             inter_size,
                                             head_num,
                                             size_per_head,
                                             num_token,
                                             mem_len,
                                             input_name,
                                             model_name,
                                             check_name,
                                             allow_gemm_test);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

/*************** Correctness Check*****************/
template<typename T>
int tfxlCorrectnessExample(size_t max_batch_size,
                            size_t num_layers,
                            size_t seq_len,
                            size_t d_model,
                            size_t inter_size,
                            size_t head_num,
                            size_t size_per_head,
                            size_t num_token,
                            size_t mem_len,
                            string input_name,
                            string model_name,
                            string check_name,
                            bool allow_gemm_test)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t cat_len = mem_len + seq_len;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudnnHandle_t cudnn_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    checkCUDNN(cudnnCreate(&cudnn_handle));
    cudnnSetStream(cudnn_handle, stream);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    cublasSetMathMode(cublas_handle, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_TF32_TENSOR_OP_MATH));

    // Set layer weight
    std::vector<TfxlLayerWeight<T>> tfxl_layer_weights(num_layers, TfxlLayerWeight<T>(d_model, hidden_units, inter_size));
    const int weight_nums = 15;
    string weight_name[15] = {"/rel_attn/q:0",
                              "/rel_attn/k:0",
                              "/rel_attn/v:0",
                              "/rel_attn/r:0",
                              "/rel_attn/r_w_bias:0",
                              "/rel_attn/r_r_bias:0",
//                              "/rel_attn/r_s_bias:0",
#ifdef USE_TOKEN_TYPE_IDS   //if use this, you should update setWeightPtr function.
                              "/rel_attn/seg_embed:0",
#endif
                              "/rel_attn/o:0",
                              "/rel_attn/layer_norm/gamma:0",
                              "/rel_attn/layer_norm/beta:0",
                              "/ff/layer_1/kernel:0",
                              "/ff/layer_1/bias:0",
                              "/ff/layer_2/kernel:0",
                              "/ff/layer_2/bias:0",
                              "/ff/layer_norm/gamma:0",
                              "/ff/layer_norm/beta:0"};

    cnpy::npz_t model_npz = cnpy::npz_load(model_name);

    for (int i = 0; i < num_layers; i++) {
        T** weight_ptrs = tfxl_layer_weights[i].getWeightPtrs();
        int* weight_sizes = tfxl_layer_weights[i].getWeightSizes();
        for (int j = 0; j < weight_nums; j++) {
            string str;
            if (j < 3) {
                str = paraName(i, weight_name[j]);
                setByNpz(model_npz, str, weight_ptrs[0] + j * d_model * hidden_units, d_model * hidden_units);
            }
            else {
                str = paraName(i, weight_name[j]);
//                cout << str << endl;
                setByNpz(model_npz, str, weight_ptrs[j - 2], weight_sizes[j - 2]);
            }  // end for else
        }      // end for j
    }          // end for i

    // Allocate logit bias
    T* logit_bias;
    deviceMalloc(&logit_bias, num_token, false);
    setByNpz(model_npz, "model/lm_loss/bias:0", logit_bias, num_token);

    // Allocate Input & Output
    int* inp_k;
    deviceMalloc(&inp_k, max_batch_size * seq_len, false);
    T* params_word_emb_k;
    deviceMalloc(&params_word_emb_k, num_token * d_model, false);
    setByNpz(model_npz, "model/transformer/word_embedding/weight:0", params_word_emb_k, num_token * d_model);

    T* h_states; //tf.concat([mems, h], axis=0)
    deviceMalloc(&h_states, max_batch_size * num_layers * cat_len * d_model, false);
    deviceMemSetZero(h_states, max_batch_size * num_layers * cat_len * d_model);
    T* input_ptr = h_states + mem_len * d_model;

    T* out_tensor;
    deviceMalloc(&out_tensor, max_batch_size * seq_len * d_model, false);

    T* logit;
    deviceMalloc(&logit, max_batch_size * seq_len * num_token, false);

    T* probs;
    deviceMalloc(&probs, max_batch_size * seq_len * num_token, false);

    std::vector<int> inputCpu;

    inputCpu.resize(max_batch_size * seq_len);

//    cnpy::npz_t input_npz = cnpy::npz_load(input_name);
    cnpy::npz_t check_npz = cnpy::npz_load(check_name);

    // Prepare for the inputs and outputs as vector
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{max_batch_size, num_layers, cat_len, d_model}, h_states}};
    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{max_batch_size, seq_len, d_model}, out_tensor}};
    Tfxl<T> tfxl = Tfxl<T>(max_batch_size,
                              seq_len,
                              d_model,
                              mem_len,
                              head_num,
                              size_per_head,
                              inter_size,
                              num_layers,
                              1.0f,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false);

#if 0
    // warmup
    for (int i = 0; i < 16; i++) {
        tfxl.forward(&output_tensors, &input_tensors, &tfxl_layer_weights, max_batch_size, inputCpu);
    }
    tfxl.resetMem();
    deviceMemSetZero(h_states, max_batch_size * num_layers * cat_len * d_model);
#endif

    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);

    float w = 0.15;
    float zero = 0;

    // profile time
    int inp[max_batch_size];
    const int test_count = 1;

    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    int batch = 0;
    int cnt = 0;

    for(int i = 0; i < test_count; i++)    {
        for(int i = 0; i < max_batch_size * seq_len; i++) {
            inputCpu[i] = -1;
        }

        inputCpu[0] = 0 + cnt++;
    //    inputCpu[1] = 101;
    //    inputCpu[2] = 102;
    //    inputCpu[3] = 103;

        for(int i = max_batch_size - 1; i >= 0; i--)  {
            if(inputCpu[i] != -1)  {
            batch = i + 1;
            break;
            }
        }

        cudnnSetTensor4dDescriptor (srcTensorDesc, CUDNN_TENSOR_NHWC, (typeid(T) == typeid(float))?CUDNN_DATA_FLOAT:CUDNN_DATA_HALF,
                batch, num_token, 1, seq_len);
        cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NHWC, (typeid(T) == typeid(float))?CUDNN_DATA_FLOAT:CUDNN_DATA_HALF,
                batch, num_token, 1, seq_len);

        check_cuda_error(cudaMemcpy(inp_k, inputCpu.data(), batch * sizeof(T), cudaMemcpyHostToDevice));
        genWordEmdK(batch, seq_len, cat_len, d_model, input_ptr, params_word_emb_k, inp_k, stream);
        tfxl.forward(&output_tensors, &input_tensors, &tfxl_layer_weights, batch, inputCpu);

        cublas_wrapper.Gemm(CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            num_token,
                            batch * seq_len,
                            d_model,
                            params_word_emb_k,
                            d_model,
                            out_tensor,
                            d_model,
                            logit_bias,
                            logit,
                            num_token);
        cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, &w,
                srcTensorDesc, logit, &zero, sftTensorDesc, probs);
    }
    sync_check_cuda_error();

    float total_time = cuda_timer.stop();

    printTensor(out_tensor, 1, batch, seq_len, d_model, "output_h");
    printTensor(h_states, num_layers, max_batch_size, cat_len, d_model, "h_states");
    printTensor(logit, batch, 1, seq_len, num_token, "logit");
    printTensorLong(probs, batch, 1, seq_len, num_token, "log_softmax");
    checkByNpz(check_npz, stream, "output_h", out_tensor, batch * seq_len * d_model, test_count - 1);
    checkByNpz(check_npz, stream, "logits", logit, batch * seq_len * num_token, test_count - 1);
    checkByNpz(check_npz, stream, "log_softmax", probs, batch * seq_len * num_token, test_count - 1);

//    printDevPtr(probs, max_batch_size * seq_len * num_token, num_token, "probs", true);

    printf("[INFO] max_batch_size %ld real_batch_size %ld seq_len %ld layer %ld "
           "FT-CPP-average time %.2f ms (%d iterations) \n",
           max_batch_size,
           batch,
           seq_len,
           num_layers,
           total_time / test_count,
           test_count);

#if 0
    T* h_ptr = (T*)malloc(num_token * sizeof(T));
    check_cuda_error(cudaMemcpyAsync(h_ptr, logit, sizeof(T) * num_token, cudaMemcpyDeviceToHost, stream));
    cudaDeviceSynchronize();
    for(int i=0;i<num_token;++i)    {
        printf("%f\n", h_ptr[i]);
    }
#endif

    // Check result
    std::ostringstream s;
    s << "layer_" << (num_layers - 1);
    std::string label = s.str();
//    checkByNpz(check_npz, stream, label, out_tensor, max_batch_size * seq_len * d_model);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudnnDestroyTensorDescriptor(srcTensorDesc);
    cudnnDestroyTensorDescriptor(sftTensorDesc);

    cudaFree(logit_bias);
    cudaFree(inp_k);
    cudaFree(params_word_emb_k);
    cudaFree(h_states);

    cudaFree(out_tensor);
    cudaFree(logit);
    cudaFree(probs);
    return 0;
}
