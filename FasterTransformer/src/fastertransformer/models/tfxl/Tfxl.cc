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

#include "src/fastertransformer/models/tfxl/Tfxl.h"

namespace fastertransformer {

template<typename T>
void Tfxl<T>::initialize()
{
    allocateBuffer();
    attention_layer_ = new TfxlAttentionLayer<T>(max_batch_size_,
                                                  max_seq_len_,
                                                  d_model_,
                                                  max_mem_len_,
                                                  head_num_,
                                                  size_per_head_,
                                                  q_scaling_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_);
    ffn_layer_ = new ReluFfnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     d_model_ / head_num_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);
}

template<typename T>
Tfxl<T>::Tfxl(size_t max_batch_size,
                size_t max_seq_len,
                size_t d_model,
                size_t mem_len,
                size_t head_num,
                size_t size_per_head,
                size_t inter_size,
                size_t num_layer,
                float q_scaling,
                cudaStream_t stream,
                cublasMMWrapper* cublas_wrapper,
                IAllocator* allocator,
                bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    d_model_(d_model),
    max_mem_len_(mem_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    q_scaling_(q_scaling)
{
    initialize();
}

template<typename T>
Tfxl<T>::Tfxl(Tfxl<T> const& tfxl):
    BaseLayer(tfxl),
    max_batch_size_(tfxl.max_batch_size_),
    max_seq_len_(tfxl.max_seq_len_),
    d_model_(tfxl.d_model_),
    max_mem_len_(tfxl.max_mem_len_),
    head_num_(tfxl.head_num_),
    size_per_head_(tfxl.size_per_head_),
    inter_size_(tfxl.inter_size_),
    hidden_units_(tfxl.hidden_units_),
    num_layer_(tfxl.num_layer_),
    q_scaling_(tfxl.q_scaling_)
{
    initialize();
}

template<typename T>
Tfxl<T>::~Tfxl()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void Tfxl<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        const size_t cat_len = max_mem_len_ + max_seq_len_;
        const size_t pos_len = cat_len;

        attr_k_head_r_ = (T*)allocator_->malloc(sizeof(T) * pos_len * d_model_, false);
        attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        output_fc2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * d_model_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Tfxl<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attr_k_head_r_);
        allocator_->free(attn_out_buf_);
        allocator_->free(output_fc2_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void Tfxl<T>::forward(std::vector<Tensor>* output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const std::vector<TfxlLayerWeight<T>>* tfxl_layer_weights,
                       int req_batch_size,
                       const std::vector<int>& inputCpu)
{
    // input_tensors:
    // h_start [batch_size_, max_mem_len_ + seq_len_, d_model_],

    // output_tensors:
    // out_tensor [batch_size, seq_len, d_model_]

    const size_t max_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[2] - max_mem_len_;
    const size_t max_cat_len = input_tensors->at(0).shape[2];
    const size_t max_pos_len = max_cat_len;
    const size_t cat_len = cur_mem_len_ + request_seq_len;
    const size_t pos_len = cat_len;
    T* h_states = (T*)input_tensors->at(0).data;
    T* output_ptr = (T*)output_tensors->at(0).data;

    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(isValidBatchSize(max_batch_size));
    FT_CHECK(isValidSeqLen(request_seq_len));
    FT_CHECK(input_tensors->at(0).shape.size() == 4);
//    FT_CHECK(max_mem_len_ >= request_seq_len);

    preProcess(req_batch_size,
               request_seq_len,
               max_mem_len_,
               d_model_,
               attr_k_head_r_,
               stream_);

    DataType data_type = getTensorType<T>();
    for (uint i = 0; i < num_layer_; i++) {
        T* h_start = h_states + i * max_batch_size * max_cat_len * d_model_; 
        T* cat = h_start + (max_mem_len_ - cur_mem_len_) * d_model_; 
        T* h = h_start + max_mem_len_ * d_model_;
        T* out_tensor = output_ptr;

        // set input
        if(i != 0)  {
            for(int batch = 0; batch < req_batch_size; batch++) {
                T* base = h_start + batch * max_cat_len * d_model_;
                T* inp = base + max_mem_len_ * d_model_;

                if(inputCpu[batch] == -1) continue;

                    cudaMemcpyAsync(inp, out_tensor + batch * request_seq_len * d_model_, sizeof(T) * request_seq_len * d_model_, cudaMemcpyDeviceToDevice, stream_);
            }
        }

        std::vector<Tensor> attn_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{max_batch_size, cat_len, d_model_},
                   cat},
            Tensor{MEMORY_GPU, data_type, std::vector<size_t>{pos_len, d_model_}, attr_k_head_r_ + (max_pos_len - pos_len) * d_model_}};

        std::vector<Tensor> attn_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{max_batch_size, request_seq_len, d_model_},
                   attn_out_buf_}};

        attention_layer_->forward(
            &attn_output_tensors, &attn_input_tensors, &tfxl_layer_weights->at(i).attention_weights, req_batch_size);

        //printTensor(tfxl_layer_weights->at(i).attn_layernorm_weights.gamma, 1, 1, 1, d_model_, "attn_layernorm_weights.gamma");
        //printTensor(tfxl_layer_weights->at(i).attn_layernorm_weights.beta, 1, 1, 1, d_model_, "attn_layernorm_weights.beta");

        //printTensor(out_tensor, max_batch_size, 1/*num_layers*/, request_seq_len, d_model_, "before out_tensor");


        invokeAddResidualLayerNorm(req_batch_size,
                                   request_seq_len,
                                   max_cat_len,
                                   d_model_,
                                   out_tensor,
                                   attn_out_buf_,
                                   h,
                                   tfxl_layer_weights->at(i).attn_layernorm_weights.gamma,
                                   tfxl_layer_weights->at(i).attn_layernorm_weights.beta,
                                   stream_);

        //printTensor(out_tensor, max_batch_size, 1/*num_layers*/, request_seq_len, d_model_, "invokeAddResidualLayerNorm out_tensor");

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{req_batch_size * request_seq_len, d_model_},
                   out_tensor}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{req_batch_size * request_seq_len, d_model_},
                   output_fc2_}};
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &tfxl_layer_weights->at(i).ffn_weights);

        //printTensor(output_fc2_, max_batch_size, 1/*num_layers*/, request_seq_len, d_model_, "output_fc2_");

        invokeAddBiasResidualLayerNorm(out_tensor,
                                       output_fc2_,
                                       tfxl_layer_weights->at(i).ffn_weights.output_weight.bias,
                                       tfxl_layer_weights->at(i).ffn_layernorm_weights.gamma,
                                       tfxl_layer_weights->at(i).ffn_layernorm_weights.beta,
                                       req_batch_size * request_seq_len,
                                       d_model_,
                                       stream_);

        //printTensor(out_tensor, max_batch_size, 1/*num_layers*/, request_seq_len, d_model_, "final attn_output");

        // update states
        if(max_mem_len_)    {
            int len = cur_mem_len_;
            int start = max_mem_len_ - cur_mem_len_;
            int dst = start - request_seq_len;

            if(dst < 0) {
                len += dst;
                start += -dst;
                dst = 0;
            }
            for(int batch = 0; batch < req_batch_size; batch++) {
                T* base = h_start + batch * max_cat_len * d_model_;
                T* inp = base + max_mem_len_ * d_model_;

                if(inputCpu[batch] == -1) continue;

                invokeCudaD2DcpyState(base + dst * d_model_, base + start * d_model_, d_model_, len, stream_);
                cudaMemcpyAsync(base + (max_mem_len_ - request_seq_len) * d_model_, inp, sizeof(T) * request_seq_len * d_model_, cudaMemcpyDeviceToDevice, stream_);
            }
        }
    }  // end for num_layer
    cur_mem_len_ += request_seq_len;
    if(cur_mem_len_ > max_mem_len_)
        cur_mem_len_ = max_mem_len_;
}

template<typename T>
bool Tfxl<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool Tfxl<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class Tfxl<float>;
template class Tfxl<half>;

}  // namespace fastertransformer
