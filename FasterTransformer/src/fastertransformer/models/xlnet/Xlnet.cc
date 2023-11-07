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

#include "src/fastertransformer/models/xlnet/Xlnet.h"

namespace fastertransformer {

template<typename T>
void Xlnet<T>::initialize()
{
    allocateBuffer();
    attention_layer_ = new XlnetAttentionLayer<T>(max_batch_size_,
                                                  max_seq_len_,
                                                  max_mem_len_,
                                                  head_num_,
                                                  size_per_head_,
                                                  q_scaling_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_);
    ffn_layer_ = new GeluFfnLayer<T>(max_batch_size_,
                                     max_seq_len_,
                                     head_num_,
                                     size_per_head_,
                                     inter_size_,
                                     stream_,
                                     cublas_wrapper_,
                                     allocator_,
                                     is_free_buffer_after_forward_,
                                     false);
}

template<typename T>
Xlnet<T>::Xlnet(size_t max_batch_size,
                size_t max_seq_len,
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
Xlnet<T>::Xlnet(Xlnet<T> const& xlnet):
    BaseLayer(xlnet),
    max_batch_size_(xlnet.max_batch_size_),
    max_seq_len_(xlnet.max_seq_len_),
    max_mem_len_(xlnet.max_mem_len_),
    head_num_(xlnet.head_num_),
    size_per_head_(xlnet.size_per_head_),
    inter_size_(xlnet.inter_size_),
    hidden_units_(xlnet.hidden_units_),
    num_layer_(xlnet.num_layer_),
    q_scaling_(xlnet.q_scaling_)
{
    initialize();
}

template<typename T>
Xlnet<T>::~Xlnet()
{
    delete attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void Xlnet<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        const size_t cat_len = max_mem_len_ + max_seq_len_;
        const size_t pos_len = cat_len + max_seq_len_;

        attr_k_head_r_ = (T*)allocator_->malloc(sizeof(T) * pos_len * hidden_units_, false);
        attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        output_fc2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Xlnet<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attr_k_head_r_);
        allocator_->free(output_fc2_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void Xlnet<T>::forward(std::vector<Tensor>* output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const std::vector<XlnetLayerWeight<T>>* xlnet_layer_weights)
{
    // input_tensors:
    // h_start [batch_size_, max_mem_len_ + seq_len_, hidden_units_],

    // output_tensors:
    // out_tensor [batch_size, seq_len, hidden_units]

    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[2] - max_mem_len_;
    const size_t max_cat_len = input_tensors->at(0).shape[2];
    const size_t max_pos_len = max_cat_len + request_seq_len;
    const size_t cat_len = cur_mem_len_ + request_seq_len;
    const size_t pos_len = cat_len + request_seq_len;
    T* h_states = (T*)input_tensors->at(0).data;
    T* output_ptr = (T*)output_tensors->at(0).data;

    FT_CHECK(input_tensors->size() == 1);
    FT_CHECK(isValidBatchSize(request_batch_size));
    FT_CHECK(isValidSeqLen(request_seq_len));
    FT_CHECK(input_tensors->at(0).shape.size() == 4);
//    FT_CHECK(max_mem_len_ >= request_seq_len);

    preProcess(request_batch_size,
               request_seq_len,
               max_mem_len_,
               hidden_units_,
               attr_k_head_r_,
               stream_);

    DataType data_type = getTensorType<T>();
    for (uint i = 0; i < num_layer_; i++) {
        T* h_start = h_states + i * request_batch_size * max_cat_len * hidden_units_; 
        T* cat = h_start + (max_mem_len_ - cur_mem_len_) * hidden_units_; 
        T* h = h_start + max_mem_len_ * hidden_units_;
        T* out_tensor = output_ptr;

        // set input
        if(i != 0)  {
            for(int batch = 0; batch < request_batch_size; batch++) {
                T* base = h_start + batch * max_cat_len * hidden_units_;
                T* inp = base + max_mem_len_ * hidden_units_;

                    cudaMemcpyAsync(inp, out_tensor + batch * request_seq_len * hidden_units_, sizeof(T) * request_seq_len * hidden_units_, cudaMemcpyDeviceToDevice, stream_);
            }
        }

        std::vector<Tensor> attn_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{request_batch_size, cat_len, hidden_units_},
                   cat},
            Tensor{MEMORY_GPU, data_type, std::vector<size_t>{pos_len, hidden_units_}, attr_k_head_r_ + (max_pos_len - pos_len) * hidden_units_}};

        std::vector<Tensor> attn_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{request_batch_size, request_seq_len, hidden_units_},
                   attn_out_buf_}};

        attention_layer_->forward(
            &attn_output_tensors, &attn_input_tensors, &xlnet_layer_weights->at(i).attention_weights);

        invokeAddResidualLayerNorm(request_batch_size,
                                   request_seq_len,
                                   max_cat_len,
                                   hidden_units_,
                                   out_tensor,
                                   attn_out_buf_,
                                   h,
                                   xlnet_layer_weights->at(i).attn_layernorm_weights.gamma,
                                   xlnet_layer_weights->at(i).attn_layernorm_weights.beta,
                                   stream_);

        //printTensor(out_tensor, request_batch_size, 1/*num_layers*/, request_seq_len, hidden_units_, "invokeAddResidualLayerNorm out_tensor");

        std::vector<Tensor> ffn_input_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_},
                   out_tensor}};
        std::vector<Tensor> ffn_output_tensors{
            Tensor{MEMORY_GPU,
                   data_type,
                   std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_},
                   output_fc2_}};
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &xlnet_layer_weights->at(i).ffn_weights);

        invokeAddBiasResidualLayerNorm(out_tensor,
                                       output_fc2_,
                                       xlnet_layer_weights->at(i).ffn_weights.output_weight.bias,
                                       xlnet_layer_weights->at(i).ffn_layernorm_weights.gamma,
                                       xlnet_layer_weights->at(i).ffn_layernorm_weights.beta,
                                       request_batch_size * request_seq_len,
                                       hidden_units_,
                                       stream_);

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
            for(int batch = 0; batch < request_batch_size; batch++) {
                T* base = h_start + batch * max_cat_len * hidden_units_;
                T* inp = base + max_mem_len_ * hidden_units_;

                invokeCudaD2DcpyState(base + dst * hidden_units_, base + start * hidden_units_, hidden_units_, len, stream_);
                cudaMemcpyAsync(base + (max_mem_len_ - request_seq_len) * hidden_units_, inp, sizeof(T) * request_seq_len * hidden_units_, cudaMemcpyDeviceToDevice, stream_);
            }
        }
    }  // end for num_layer
    cur_mem_len_ += request_seq_len;
    if(cur_mem_len_ > max_mem_len_)
        cur_mem_len_ = max_mem_len_;
}

template<typename T>
bool Xlnet<T>::isValidBatchSize(size_t batch_size)
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
bool Xlnet<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class Xlnet<float>;
template class Xlnet<half>;

}  // namespace fastertransformer
