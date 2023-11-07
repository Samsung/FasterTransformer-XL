from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# usage example
# python ./convertModel.py -i $ckpt_file -o $model_file


import getopt
import sys
import numpy as np
import absl.logging as _logging    # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras
from transformers import TFTransfoXLLMHeadModel

from absl import flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def usage():
    print(" -o output_file")
    print(" -i saved_model")
    print("Example: python convertModel.py -i xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt -o ./model.npz ")


if __name__ == "__main__":
    m = {}
    saved_model = "../../../../train/saved_model"
    output_file = "./data/model.npz"

    # Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
    for op, value in opts:
        if op == "-i":
            saved_model = value
        if op == "-o":
            output_file = value
        if op == "-h":
            usage()
            sys.exit()

    print('saved_model', saved_model)
    model = TFTransfoXLLMHeadModel.from_pretrained(saved_model)
    names = [weight.name.replace('tf_transfo_xllm_head_', '') for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    mapping = [ 
        ('layers', 'layer'), 
        ('dec_attn', 'rel_attn'), 
        ('pos_ff', 'ff'), 
        ('CoreNet_._0', 'layer_1'), 
        ('CoreNet_._3', 'layer_2'), 
        ('r_net/kernel', 'r'), 
        ('o_net/kernel', 'o') 
    ]

    idx = 0
    for name, weight in zip(names, weights):
        if 'crit' in name:
            continue

        for k, v in mapping:
            name = name.replace(k, v)

        if 'rel_attn/o' in name:
            weight = tf.transpose(weight)
        elif 'qkv' in name:
            name = name.replace('/qkv_net/kernel:0', '')
            w_head_q, w_head_k, w_head_v = tf.split(weight, 3, axis=-1)
            m[name + '/q:0'] = w_head_q
            m[name + '/k:0'] = w_head_k
            m[name + '/v:0'] = w_head_v
            continue

        m[name] = weight

    for name, weight in m.items():
        print(idx, name, weight.shape)
        idx += 1
        if '/rel_attn/layer_norm' in name:
            print(weight)

    np.savez(output_file, **m)
