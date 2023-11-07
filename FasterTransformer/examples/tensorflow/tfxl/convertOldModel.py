import os
import re
import getopt
import sys
import numpy as np

import tensorflow as tf

def usage():
    print(" -o output_file")
    print(" -i saved_model")
    print("Example: python convertModel.py -i xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt -o ./model.npz ")

def get_enc_layer() :
  max_enc_num = 0

  pat = re.compile("transformer/layer_([0-9]*)/ff/layer_1/bias")
  for var in var_list:
    mat = pat.match(var[0])
    if mat: max_enc_num = max(max_enc_num, int(mat.group(1)))

  return max_enc_num + 1

if __name__ == "__main__":
    m = {}
    checkpoint = "lm_model_ckpt/tfxl/tf1/model.epoch-19.0.valid-50.82"
    output_file = "./data/model.npz"

    # Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
    for op, value in opts:
        if op == "-i":
            checkpoint = value
        if op == "-o":
            output_file = value
        if op == "-h":
            usage()
            sys.exit()

    reader = tf.train.load_checkpoint(checkpoint)
    var_list = tf.train.list_variables(checkpoint)

    var_to_shape_map = reader.get_variable_to_shape_map()
    idx = 0

    mapping = [ 
        ('layer_', 'layer_._'), 
        ('LayerNorm', 'layer_norm'), 
        ('/ff/layer_._1/', '/ff/layer_1/'), 
        ('/ff/layer_._2/', '/ff/layer_2/'),
        ('/r/kernel', '/r'), 
        ('/o/kernel', '/o'), 
        ('transformer/adaptive_softmax', 'lm_loss'), 
        ('adaptive_embed/lookup_table', 'word_embedding/weight')
    ]

    for key in var_to_shape_map:
        if 'transformer' in key and 'Adam' not in key:
            tensor = reader.get_tensor(key)
            key = 'model/' + key

            for k, v in mapping:
                key = key.replace(k, v)

            if 'qkv' in key:
                key = key.replace('/qkv/kernel', '')
                w_head_q, w_head_k, w_head_v = tf.split(tensor, 3, axis=-1)
                m[key + '/q:0'] = w_head_q
                m[key + '/k:0'] = w_head_k
                m[key + '/v:0'] = w_head_v
                continue
            elif ('r_r_bias' in key) or ('r_w_bias' in key):
                key = key.replace('model/transformer/', '')
                for i in range(len(tensor)):
                    tmp = 'model/transformer/layer_._{}/rel_attn/{}:0'.format(i, key)
                    m[tmp] = tensor[i]
                continue

            if '/rel_attn/o' in key:
                tensor = tf.transpose(tensor)

            key += ':0'
            m[key] = tensor

    for key, value in sorted(m.items(), key=lambda x: x[0]): 
        print(idx, '\t', key, value.shape)
        idx += 1

    np.savez(output_file, **m)

    enc_layer = get_enc_layer()
    print("enc_layer = %d" % enc_layer)
