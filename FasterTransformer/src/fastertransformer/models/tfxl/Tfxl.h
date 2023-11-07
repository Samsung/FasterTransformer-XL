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

#pragma once

#include <vector>

#include "src/fastertransformer/kernels/tfxl_preprocess_kernels.h"
#include "src/fastertransformer/layers/tfxl_attention_layers/TfxlAttentionLayer.h"
#include "src/fastertransformer/models/tfxl/TfxlLayerWeight.h"

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"

namespace fastertransformer {

template<typename T>
class Tfxl: public BaseLayer {
private:
    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_ = 0;
    size_t d_model_ = 0;
    size_t max_mem_len_ = 0;
    size_t cur_mem_len_ = 0;
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t hidden_units_;
    size_t num_layer_;
    float q_scaling_;

    bool is_allocate_buffer_ = false;
    FfnLayer<T>* ffn_layer_;

    void allocateBuffer();
    void freeBuffer();
    bool isValidBatchSize(size_t batch_size);
    bool isValidSeqLen(size_t seq_len);
    void initialize();

protected:
    // Preprocess data
    T* word_emb_k_;
    T* attr_k_head_r_;

    // Postprocess data
    T* attn_out_buf_;
    T* output_fc2_;

    TfxlAttentionLayer<T>* attention_layer_;

public:
    Tfxl(size_t max_batch_size,
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
          bool is_free_buffer_after_forward);

    Tfxl(Tfxl<T> const& tfxl_layer);

    ~Tfxl();

    void forward(std::vector<Tensor>* output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const std::vector<TfxlLayerWeight<T>>* tfxl_layer_weights,
                 int req_batch_size,
                 const std::vector<int>& inputCpu);
    void resetMem() {cur_mem_len_ = 0;}
};

}  // namespace fastertransformer
