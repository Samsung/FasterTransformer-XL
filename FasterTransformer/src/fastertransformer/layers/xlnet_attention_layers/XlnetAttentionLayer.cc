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

#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionLayer.h"

#define __XLNET_DEBUG__

namespace fastertransformer {
template<typename T>
void XlnetAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                     const XlnetAttentionWeight<T>* attention_weights)
{
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t max_cat_len = max_mem_len_ + max_seq_len_;
    const size_t cat_len = input_tensors->at(0).shape[1];
    const size_t mem_len = cat_len - max_seq_len_;
    const size_t request_seq_len = max_seq_len_;
    const size_t pos_len = cat_len + request_seq_len;

    FT_CHECK(isValidBatchSize(input_tensors->at(0).shape[0]));
    FT_CHECK(isValidSeqLen(request_seq_len));

    FT_CHECK(input_tensors->size() == 2);
    FT_CHECK(input_tensors->at(0).shape.size() == 3);
    FT_CHECK(input_tensors->at(1).shape.size() == 2);

    FT_CHECK(input_tensors->at(0).shape[0] == request_batch_size);

    T* out_tensor = (T*)output_tensors->at(0).data;
    T* cat_base = (T*)input_tensors->at(0).data;
    T* cat_ptr;
    T* h = cat_base + mem_len * hidden_units_;
    T* attr_k_head_r = (T*)input_tensors->at(1).data;

    // content-based query head
    // q_head_h(query_buf_) = tf.einsum("ibh,hnd->ibnd", h, self.q)
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        hidden_units_,
                                        request_seq_len * request_batch_size,
                                        hidden_units_,
                                        attention_weights->attr_kernel_Q,
                                        hidden_units_,
                                        h,
                                        hidden_units_ * max_cat_len,
                                        query_buf_,
                                        hidden_units_);

    if(cat_len == max_cat_len)  {
        cat_ptr = cat_base;
    }   
    else    {
        cat_ptr = cat_tmp_;
        for(int i = 0; i < request_batch_size; i++) {
            int len = cat_len * hidden_units_;

            cudaMemcpy(cat_tmp_ + i * len, cat_base + i * max_cat_len * hidden_units_, sizeof(T) * len, cudaMemcpyDeviceToDevice);
        }                                        
    }

    // content-based key head
    // k_head_h(key_buf_) = tf.einsum("ibh,hnd->ibnd", cat_base, self.k)
    // content-based value head
    // v_head_h(value_buf_) = tf.einsum("ibh,hnd->ibnd", cat_base, self.v)
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        hidden_units_,
                                        cat_len * request_batch_size,
                                        hidden_units_,
                                        attention_weights->attr_kernel_K,
                                        hidden_units_,
                                        hidden_units_ * hidden_units_,
                                        cat_ptr,
                                        hidden_units_,
                                        0,
                                        key_buf_,
                                        hidden_units_,
                                        hidden_units_ * max_cat_len * request_batch_size,
                                        2);

    //printTensor(attr_k_head_r, 1 , pos_len, 1, hidden_units_, "attr_k_head_r");    

    // positional heads
    // k_head_r = tf.einsum("ibh,hnd->ibnd", r, self.r)
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          hidden_units_,
                          pos_len,
                          hidden_units_,
                          attention_weights->attr_pos_emb,
                          hidden_units_,
                          attr_k_head_r,
                          hidden_units_,
                          k_head_r_,
                          hidden_units_);

    invokePrepareMatrixes(request_batch_size,
                          request_seq_len,
                          mem_len,
                          hidden_units_,
                          size_per_head_,
                          q_buf_,
                          q_buf_bd_,
                          k_buf_,
                          k_buf_bd_,
                          query_buf_,
                          key_buf_,
                          k_head_r_,
                          attention_weights->attr_bias_Q_w,
                          attention_weights->attr_bias_Q_r,
                          attention_weights->attr_bias_Q_s,
                          stream_);

    //printTensor(query_buf_,request_batch_size , request_seq_len, head_num_, size_per_head_, "q_head_h org");    
    //printTensor(key_buf_,request_batch_size , cat_len, head_num_, size_per_head_, "k_head_h org");    
    //printTensor(k_buf_,request_batch_size , cat_len, head_num_, size_per_head_, "k_head_h");    

    // content based attention score
    // ac(qk_buf_) = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_w_bias, k_head_h)
    // ex) seq_len=10, mem_len=16, batch=1, num_head=8, d_head=40
    //     q_head (10, 1, 8, 40), k_head_h (26, 1, 8, 40), ac (10, 26, 1, 8)
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        cat_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_,
                                        size_per_head_,
                                        cat_len * size_per_head_,
                                        q_buf_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        cat_len,
                                        cat_len * request_seq_len,
                                        request_batch_size * head_num_);

    //printTensor(qk_buf_, request_batch_size, head_num_, request_seq_len, cat_len, "ac");    

    // position based attention score
    // bd(qk_buf_bd_) = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_r_bias, k_head_r)
    //printTensor(q_buf_bd_,request_batch_size , head_num_, request_seq_len, size_per_head_, "q_head + self.r_r_bias");    
    //printTensor(k_head_r_,1 , 1, pos_len, hidden_units_, "k_head_r org");    
    //printTensor(k_buf_bd_,request_batch_size , head_num_, pos_len, size_per_head_, "k_head_r");    
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        pos_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_bd_,
                                        size_per_head_,
                                        pos_len * size_per_head_,
                                        q_buf_bd_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_bd_,
                                        pos_len,
                                        pos_len * request_seq_len,
                                        request_batch_size * head_num_);


    //printTensor(qk_buf_bd_, 1, request_batch_size, head_num_, pos_len, "bd");    

    //TODO : add support for seq_len > 1 cases to invokeRelShiftBd
    FT_CHECK(request_seq_len == 1);
    invokeRelShiftBd(request_batch_size, head_num_, request_seq_len, pos_len, qk_buf_bd_shift_, qk_buf_bd_, stream_);

    //printTensor(qk_buf_bd_shift_, 1, request_batch_size, head_num_, cat_len, "bd shift");    
    //printTensor(value_buf_, 1, request_batch_size, head_num_, size_per_head_, "v_head_h");    

    // merge attention scores and perform masking
    invokeCalAttnScore(request_batch_size,
                       head_num_,
                       request_seq_len,
                       cat_len,
                       max_cat_len,
                       size_per_head_,
                       q_scaling_,
                       attn_score_,
                       qk_buf_,
                       qk_buf_bd_shift_,
                       value_buf_trans_,
                       value_buf_,
                       stream_);

    // CAUTION : In the case of attn_score_, there could be a big difference with the python value because the implementation method is different.
    //printTensor(attn_score_, 1, request_batch_size, request_seq_len, head_num_, "attn_score_");    

    // attention output
    // attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)
    // ex) seq_len=10, mem_len=16, batch=1, num_head=8, d_head=40
    //     attn_prob (10, 26, 1, 8), v_head_h (26, 1, 8, 40), attn_vec (10, 1, 8, 40)
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        cat_len,
                                        value_buf_trans_,
                                        size_per_head_,
                                        cat_len * size_per_head_,
                                        attn_score_,
                                        cat_len,
                                        cat_len * request_seq_len,
                                        attn_vec_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        request_batch_size * head_num_);

    invokeTranspose102v2(
        request_batch_size, request_seq_len, head_num_, size_per_head_, attn_vec_trans_, attn_vec_, stream_);

    // post-attention projection (back to `d_model`)
    // attn_out = tf.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          request_seq_len * request_batch_size,
                          hidden_units_,
                          attention_weights->attr_proj_o,
                          hidden_units_,
                          attn_vec_trans_,
                          hidden_units_,
                          out_tensor,
                          hidden_units_);

    //printTensor(out_tensor, request_batch_size, 1/*num_layers*/, request_seq_len, hidden_units_, "attn_out");
}

template<typename T>
XlnetAttentionLayer<T>::XlnetAttentionLayer(size_t max_batch_size,
                                            size_t max_seq_len,
                                            size_t mem_len,
                                            size_t head_num,
                                            size_t size_per_head,
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
    q_scaling_(q_scaling)
{
    hidden_units_ = head_num_ * size_per_head_;
    allocateBuffer();
}

template<typename T>
void XlnetAttentionLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        const size_t max_cat_len = max_mem_len_ + max_seq_len_;
        const size_t pos_len = max_cat_len + max_seq_len_;

        k_head_r_ = (T*)allocator_->malloc(sizeof(T) * pos_len * hidden_units_, false);
        query_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * (max_seq_len_ + 2*max_cat_len) * hidden_units_, false);
        key_buf_ = query_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        value_buf_ = query_buf_ + max_batch_size_ * (max_seq_len_ + max_cat_len) * hidden_units_;
        q_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_cat_len * hidden_units_, false);
        qk_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_cat_len * head_num_, false);
        q_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * pos_len * hidden_units_, false);
        qk_buf_bd_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * pos_len, false);
        qk_buf_bd_shift_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * max_cat_len, false);
        attn_score_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_cat_len * max_seq_len_ * head_num_, false);
        value_buf_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_cat_len * hidden_units_, false);
        attn_vec_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_vec_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_out_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        cat_tmp_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_cat_len * hidden_units_, false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
bool XlnetAttentionLayer<T>::isValidBatchSize(size_t batch_size)
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
bool XlnetAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template<typename T>
void XlnetAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(k_head_r_);
        allocator_->free(query_buf_);
        allocator_->free(q_buf_);
        allocator_->free(k_buf_);
        allocator_->free(qk_buf_);
        allocator_->free(q_buf_bd_);
        allocator_->free(k_buf_bd_);
        allocator_->free(qk_buf_bd_);
        allocator_->free(qk_buf_bd_shift_);
        allocator_->free(attn_score_);
        allocator_->free(value_buf_trans_);
        allocator_->free(attn_vec_);
        allocator_->free(attn_vec_trans_);
        allocator_->free(attn_out_);
        allocator_->free(cat_tmp_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
XlnetAttentionLayer<T>::~XlnetAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class XlnetAttentionLayer<float>;
template class XlnetAttentionLayer<half>;

}  // namespace fastertransformer
