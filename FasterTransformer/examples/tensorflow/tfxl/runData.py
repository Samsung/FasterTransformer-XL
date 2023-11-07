from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# python ./runData.py -i $data_file -o $output_file -j $json_file -m $ckpt_file -b $max_batch_size -l $seq_len -f $use_float16 -n $index


import getopt
import modeling
import numpy as np
from tensorflow.python.client import timeline
import tensorflow as tf
from datetime import datetime
import json
import sys
import absl.logging as _logging    # pylint: disable=unused-import
from transformers import TFTransfoXLLMHeadModel
from transformers import PretrainedConfig

from absl import flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def usage():
    print(" -i input_file")
    print(" -o output_file")
    print(" -j json_file")
    print(" -m model_file")
    print(" -l max_seq_length")
    print(" -b max_batch_size")
    print(" -o output_file")
    print(" -f use_float16")
    print(" -n index of the inputdata batch")
    print(" -e length of the memory")
    print(" -h output help info")
    print("Example: python runData.py -i ./data.npz -o output.npz -j xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
            -m xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt -b 8 -l 128 -n 12")


if __name__ == "__main__":
    # Init perameter
    seq_len = 128
    max_batch_size = 8
    real_batch_size = 1
    use_float16 = False
    input_file = "./data.npz"
    index = 0

    json_file = "../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json"
    model_file = "../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt"
    output_file = "./output.npz"

    # Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "hi:j:m:b:r:l:f:o:n:e:d:")
    for op, value in opts:
        if op == "-i":
            input_file = value
        elif op == "-o":
            output_file = value
        elif op == "-j":
            json_file = value
        elif op == "-m":
            model_file = value
        elif op == "-b":
            max_batch_size = int(value)
        elif op == "-r":
            real_batch_size = int(value)
        elif op == "-l":
            seq_len = int(value)
        elif op == "-f":
            use_float16 = bool(int(value))
        elif op == "-n":
            index = int(value)
        elif op == "-e":
            mem_len = int(value)
        elif op == "-d":
            use_dynamic_mem_len = int(value)
        elif op == "-h":
            usage()
            sys.exit()

    print("USE FLOAT 16: ", str(use_float16))

    config = PretrainedConfig.from_pretrained(model_file)
    config.mem_len = mem_len
    config.save_pretrained(model_file)

    model = TFTransfoXLLMHeadModel.from_pretrained(model_file)
    # Set input
#    inputs = model.prepare_inputs_for_generation(input_ids, past=False, use_mems=True if mem_len != 0 else False)

    if use_dynamic_mem_len == 1:
        initial_mem_len = 0
    else:
        initial_mem_len = mem_len

    if mem_len != 0:
        mems = [tf.zeros([initial_mem_len, real_batch_size, model.transformer.d_model], tf.float32) for i in range(model.transformer.n_layer)]
    else:
        mems = None

    # Save result
    data = {}
    i = 0
    cnt = 0
    for i in range(1):
        # Get Input Value
        input_ids = tf.convert_to_tensor([[cnt + j + 0] for j in range(real_batch_size)], dtype=tf.int32)
        cnt += real_batch_size
        print('input_ids', input_ids)
        print('mems', mems)
        output = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True,mems=mems)

#        print('output.hidden_states', output.hidden_states)
#        print('output.attentions', output.attentions)

        data["input_h_{}".format(i)] = output.hidden_states[0][:,0,:]    #ignore mask token
        data["output_h_{}".format(i)] = output.hidden_states[-1]
#        data["logits_{}".format(i)] = output.logits
#        data["softmax_{}".format(i)] = tf.nn.softmax(output.logits)

        if mem_len != 0:
            data["before mems_{}".format(i)] = tf.transpose(mems, perm=[0, 2, 1, 3])
            mems = output.mems
            data["after mems_{}".format(i)] = tf.transpose(mems, perm=[0, 2, 1, 3])

#            print(i, 'before mmm', data["before mems_{}".format(i)])
#            print(i, 'after mmm', data["after mems_{}".format(i)])

#    print(data) 

    np.savez(output_file, **data)
