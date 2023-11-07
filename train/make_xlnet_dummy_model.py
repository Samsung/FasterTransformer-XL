import os

import tensorflow as tf
import numpy as np
from transformers import XLNetConfig
from transformers import TransfoXLTokenizer
from transformers import TFXLNetLMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import TFTrainer, TFTrainingArguments
from datasets import load_dataset
from tensorflow.keras.callbacks import ModelCheckpoint

config = XLNetConfig(
    cutoffs=[],
    vocab_size=4362,
    n_layer=12,
    mem_len=16,
    d_model=320,
    n_head=8,
    d_head=40,      #d_model // n_head
    d_inner=1280,   #4 * n_head * d_head
#    div_val=1,
    pre_lnorm=False,
#    reuse_len=16,
    clamp_len=820,
    same_length=False,
    proj_share_all_but_first=False,
    attn_type='bi',
    sample_softmax=-1,
    adaptive=False,
    dropout=0.1,
    dropatt=0.0,
    untie_r=True,
    init="normal",
    init_range=0.01,
    proj_init_std=0.01,
    init_std=0.02,
    layer_norm_epsilon=1e-5,
    eos_token_id=0,
#    use_mems_train=True
)

tokenizer = TransfoXLTokenizer(vocab_file='vocab.txt', lower_case=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dataset = load_dataset('text', data_files={'train': ['train.txt'], 'test': 'train.txt'})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="do_not_pad", truncation=False, pad_to_multiple_of=0)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# trainset-set
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")

tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["input_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=1,
)

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=["input_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=1,
)

def create_model():
    model = TFXLNetLMHeadModel(config=config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    return model

model = create_model()
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=1)

path = './saved_model'
model.save_pretrained(path)

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

#model.save('saved_model/xlnet')
