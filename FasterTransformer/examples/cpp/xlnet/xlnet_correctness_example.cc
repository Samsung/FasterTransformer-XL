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
#include "src/fastertransformer/models/xlnet/Xlnet.h"

using namespace fastertransformer;
using namespace std;
#include <iomanip>
#include <cudnn.h>
#include "src/fastertransformer/utils/cuda_utils.h"

//#define __XLNET_DEBUG__

template<typename T>
int xlnetCorrectnessExample(size_t batch_size,
                            size_t num_layers,
                            size_t seq_len,
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
//    printKey(my_npz);
    cout << name << endl;

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

    if (argc != 12) {
        printf("[ERROR] ./bin/xlnet_correctness_example batch_size num_layers seq_len "
               "head_num size_per_head num_token mem_len input_name model_name check_name "
               "is_fp16\n");
        printf("e.g., ./bin/xlnet_correctness_example 8 12 128 12 64 32000 16"
               "./data/data.npz ./data/model.npz ./data/output.npz 0\n");
        return 0;
    }
    bool allow_gemm_test = false;

    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    int num_token = atoi(argv[6]);
    int mem_len = atoi(argv[7]);
    input_name = argv[8];
    model_name = argv[9];
    check_name = argv[10];
    bool is_fp16 = atoi(argv[11]);

    cout << " " << batch_size << " " << num_layers << " " << seq_len << " " << head_num << " " << size_per_head << " "
         << num_token << " " << mem_len << " " << input_name << " " << model_name << " " << check_name << " " << is_fp16 << endl;

    if (is_fp16 == 0) {
        return xlnetCorrectnessExample<float>(batch_size,
                                              num_layers,
                                              seq_len,
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
        return xlnetCorrectnessExample<half>(batch_size,
                                             num_layers,
                                             seq_len,
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
int xlnetCorrectnessExample(size_t batch_size,
                            size_t num_layers,
                            size_t seq_len,
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
    const size_t inter_size = 4 * hidden_units;
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
    std::vector<XlnetLayerWeight<T>> xlnet_layer_weights(num_layers, XlnetLayerWeight<T>(hidden_units, inter_size));
    const int weight_nums = 16;
    string weight_name[16] = {"/rel_attn/q:0",
                              "/rel_attn/k:0",
                              "/rel_attn/v:0",
                              "/rel_attn/r:0",
                              "/rel_attn/r_w_bias:0",
                              "/rel_attn/r_r_bias:0",
                              "/rel_attn/r_s_bias:0",
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
        T** weight_ptrs = xlnet_layer_weights[i].getWeightPtrs();
        int* weight_sizes = xlnet_layer_weights[i].getWeightSizes();
        for (int j = 0; j < weight_nums; j++) {
            string str;
            if (j < 3) {
                str = paraName(i, weight_name[j]);
//                cout << str << endl;
                setByNpz(model_npz, str, weight_ptrs[0] + j * hidden_units * hidden_units, hidden_units * hidden_units);
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
    deviceMalloc(&inp_k, batch_size * seq_len, false);
    T* params_word_emb_k;
    deviceMalloc(&params_word_emb_k, num_token * hidden_units, false);
    setByNpz(model_npz, "model/transformer/word_embedding/weight:0", params_word_emb_k, num_token * hidden_units);

    T* h_states; //tf.concat([mems, h], axis=0)
    deviceMalloc(&h_states, batch_size * num_layers * cat_len * hidden_units, false);
//    deviceMemSetZero(h_states, batch_size * num_layers * cat_len * hidden_units);
    T* input_ptr = h_states + mem_len * hidden_units;

    T* out_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * hidden_units, false);

    T* logit;
    deviceMalloc(&logit, batch_size * seq_len * num_token, false);

    T* probs;
    deviceMalloc(&probs, batch_size * seq_len * num_token, false);

    cnpy::npz_t input_npz = cnpy::npz_load(input_name);
    cnpy::npz_t check_npz = cnpy::npz_load(check_name);

    // Prepare for the inputs and outputs as vector
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, num_layers, cat_len, hidden_units}, h_states}};
    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, out_tensor}};
    Xlnet<T> xlnet = Xlnet<T>(batch_size,
                              seq_len,
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

    // warmup
    for (int i = 0; i < 16; i++) {
        xlnet.forward(&output_tensors, &input_tensors, &xlnet_layer_weights);
    }
    xlnet.resetMem();

    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);
    cudnnSetTensor4dDescriptor (srcTensorDesc, CUDNN_TENSOR_NHWC, (typeid(T) == typeid(float))?CUDNN_DATA_FLOAT:CUDNN_DATA_HALF,
            batch_size, num_token, 1, seq_len);
    cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NHWC, (typeid(T) == typeid(float))?CUDNN_DATA_FLOAT:CUDNN_DATA_HALF,
            batch_size, num_token, 1, seq_len);
    float w = 0.15;
    float zero = 0;

    // profile time
    int inp[batch_size];
    const int test_count = 32;

    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    int cnt = 0;

    for(int i = 0; i < test_count; i++)    {
        // set input
        for(int j=0;j<batch_size;j++)   {
            inp[j] = cnt++;
        }
        check_cuda_error(cudaMemcpy(inp_k, inp, sizeof(inp), cudaMemcpyHostToDevice));
//            validateInput(inp_k, batch_size * seq_len, num_token);
        genWordEmdK(batch_size, seq_len, cat_len, hidden_units, input_ptr, params_word_emb_k, inp_k, stream);
        xlnet.forward(&output_tensors, &input_tensors, &xlnet_layer_weights);

        cublas_wrapper.Gemm(CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            num_token,
                            batch_size * seq_len,
                            hidden_units,
                            params_word_emb_k,
                            hidden_units,
                            out_tensor,
                            hidden_units,
                            logit_bias,
                            logit,
                            num_token);
        cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, &w,
                srcTensorDesc, logit, &zero, sftTensorDesc, probs);
    }
    sync_check_cuda_error();

    float total_time = cuda_timer.stop();

    printTensor(out_tensor, 1, batch_size, seq_len, hidden_units, "output_h");
    printTensor(h_states, num_layers, batch_size, cat_len, hidden_units, "h_states");
    printTensor(logit, batch_size, 1, seq_len, num_token, "logit");
    checkByNpz(check_npz, stream, "output_h", out_tensor, batch_size * seq_len * hidden_units, test_count - 1);
    checkByNpz(check_npz, stream, "logits", logit, batch_size * seq_len * num_token, test_count - 1);
    checkByNpz(check_npz, stream, "log_softmax", probs, batch_size * seq_len * num_token, test_count - 1);

//    printDevPtr(probs, batch_size * seq_len * num_token, num_token, "probs", true);

    printf("[INFO] batch_size %ld seq_len %ld layer %ld "
           "FT-CPP-average time %.2f ms (%d iterations) \n",
           batch_size,
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
//    checkByNpz(check_npz, stream, label, out_tensor, batch_size * seq_len * hidden_units);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudnnDestroyTensorDescriptor(srcTensorDesc);
    cudnnDestroyTensorDescriptor(sftTensorDesc);

//    cudaFree(logit_bias);
    cudaFree(inp_k);
    cudaFree(params_word_emb_k);
    cudaFree(h_states);

    cudaFree(out_tensor);
    cudaFree(logit);
    cudaFree(probs);
    return 0;
}
