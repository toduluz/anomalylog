#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning sequential sentence classification models"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from models.hat import HATConfig, HATTokenizer,  HATModelForLogsPreTraining
# from models.longformer import LongformerTokenizer, LongformerModelForSentenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score, roc_auc_score
from data_collator import DataCollatorForLogsPreTraining

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

# @dataclass
# class DataCollatorForSentenceOrder:
#     tokenizer: False

#     def __call__(self, features):
#         import torch
#         first = features[0]
#         batch = {}

#         # Special handling for labels.
#         # Ensure that tensor is created with the correct type
#         # (it should be automatically the case, but let's make sure of it.)
#         if "label" in first and first["label"] is not None:
#             label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
#             dtype = torch.long if isinstance(label, int) else torch.float
#             batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
#         elif "label_ids" in first and first["label_ids"] is not None:
#             if isinstance(first["label_ids"], torch.Tensor):
#                 batch["labels"] = torch.stack([f["label_ids"] for f in features])
#             else:
#                 dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
#                 batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

#         # Handling of all other possible keys.
#         # Again, we will use the first element to figure out which key/values are not None for this model.
#         for k, v in first.items():
#             if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
#                 if isinstance(v, torch.Tensor):
#                     batch[k] = torch.stack([f[k] for f in features])
#                 else:
#                     batch[k] = torch.tensor([f[k] for f in features])

        # Check-up examples
        # for example_ids, labels in zip(batch['input_ids'], batch['labels']):
        #     for idx in range(8):
        #         print('-' * 100)
        #         print(f'{idx+1} [Label={int(labels[idx])}]:', self.tokenizer.decode(example_ids[idx*128:(idx+1)*128]))
        #     print('-' * 100)
        #     print('Labels:')
        #     print('-' * 100)

        # return batch


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_sentences: int = field(
        default=32,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # print(type(training_args.label_names[2]))

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    
    if training_args.do_train:
        train_dataset = load_dataset(
            'csv',
            data_files=data_args.dataset_name+'/train.csv',
            split="train",
            # data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_eval:
        eval_dataset = load_dataset(
            'csv',
            data_files=data_args.dataset_name+'/validation.csv',
            split="train",
            # data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_predict:
        predict_dataset = load_dataset(
            'csv',
            data_files=data_args.dataset_name+'/test.csv',
            split="train",
            # data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'hat' in model_args.model_name_or_path:
        config = HATConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=2,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = HATTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = HATModelForLogsPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # else:
    #     config = AutoConfig.from_pretrained(
    #         model_args.model_name_or_path,
    #         num_labels=1,
    #         finetuning_task="sentence-order",
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    #     config.max_sentence_length = 128
    #     config.max_sentence_length = 128
    #     config.max_sentences = data_args.max_sentences
    #     tokenizer = LongformerTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         do_lower_case=model_args.do_lower_case,
    #         cache_dir=model_args.cache_dir,
    #         use_fast=model_args.use_fast_tokenizer,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    #     model = LongformerModelForSentenceClassification.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        secondary_labels = []
        tertiary_labels = []
        for example_idx, input_ids in enumerate(batch['input_ids']):
            
            #throw dice
            to_shuffle = False
            to_reverse = False
            # neg_sample_prob = np.random.uniform()
            if np.random.uniform() < 0.5:
                to_shuffle = True
            if np.random.uniform() < 0.5:
                to_reverse = True

            # count true sentences
            sentence_ids = input_ids[::config.max_sentence_length]
            n_sentences = sum(np.asarray(sentence_ids) != config.pad_token_id)
            # sentence order
            sentence_positions = list(range(n_sentences))
            secondary_positions = sentence_positions.copy()
            tertiary_positions = sentence_positions.copy()

            # secondary and tertiary
            if to_shuffle:
                random.shuffle(secondary_positions)
            if to_reverse:
                tertiary_positions = np.flip(tertiary_positions)
            
            # adapt sequence inputs
            temp_input_ids = []
            temp_attention_mask = []

            # original
            for idx in sentence_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])

            num_pad_sentences = config.max_sentences - n_sentences

            temp_input_ids.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            temp_attention_mask.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))

            # secondary
            for idx in secondary_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
            
            temp_input_ids.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            temp_attention_mask.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))

            # tertiary
            for idx in tertiary_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])

            batch_input_ids.append(temp_input_ids +
                                      [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            batch_attention_mask.append(temp_attention_mask +
                                           [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            
            # token_type_ids for all
            batch_token_type_ids.append([0] * data_args.max_seq_length * 3)

            # # Fix sentence delimiters for Longformer
            # if config.model_type == 'longformer':
            #     temp_input_ids[0] = tokenizer.cls_token_id
            #     for idx in range(1, n_sentences):
            #         temp_input_ids[idx*config.max_sentence_length] = tokenizer.sep_token_id

            # num_pad_sentences = config.max_sentences - n_sentences
            # batch_input_ids.append(temp_input_ids +
            #                           [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            # batch_attention_mask.append(temp_attention_mask +
            #                                [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            # batch_token_type_ids.append([0] * data_args.max_seq_length)
            # secondary_labels.append([float(pos) for pos in sentence_positions] + [-100] * num_pad_sentences)
            # tmp_labels = np.eye(data_args.max_sentences)[sentence_positions]
            # tmp_labels = np.pad(tmp_labels, ((0, num_pad_sentences), (0, 0)))
            # secondary_labels.append(tmp_labels)
            # print(secondary_labels)
            secondary_labels.append([1., 0.] if not to_shuffle else [0., 1.])
            
            tertiary_labels.append([1., 0.] if not to_reverse else [0., 1.])

        batch['input_ids'] = batch_input_ids
        batch['attention_mask'] = batch_attention_mask
        batch['token_type_ids'] = batch_token_type_ids
        batch['secondary_labels'] = secondary_labels
        batch['tertiary_labels'] = tertiary_labels

        return batch
    
    def preprocess_function_eval(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        secondary_labels = []
        tertiary_labels = []
        for example_idx, input_ids in enumerate(batch['input_ids']):


            # count true sentences
            sentence_ids = input_ids[::config.max_sentence_length]
            n_sentences = sum(np.asarray(sentence_ids) != config.pad_token_id)
            # sentence order
            sentence_positions = list(range(n_sentences))
            secondary_positions = sentence_positions.copy()
            tertiary_positions = sentence_positions.copy()
            
            # adapt sequence inputs
            temp_input_ids = []
            temp_attention_mask = []

            # original
            for idx in sentence_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])

            num_pad_sentences = config.max_sentences - n_sentences

            temp_input_ids.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            temp_attention_mask.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))

            # secondary
            for idx in secondary_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
            
            temp_input_ids.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            temp_attention_mask.extend([config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))

            # tertiary
            for idx in tertiary_positions:
                temp_input_ids.extend(batch['input_ids'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])
                temp_attention_mask.extend(batch['attention_mask'][example_idx][config.max_sentence_length * idx:config.max_sentence_length * (idx+1)])

            batch_input_ids.append(temp_input_ids +
                                      [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            batch_attention_mask.append(temp_attention_mask +
                                           [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            
            # token_type_ids for all
            batch_token_type_ids.append([0] * data_args.max_seq_length * 3)

            # # Fix sentence delimiters for Longformer
            # if config.model_type == 'longformer':
            #     temp_input_ids[0] = tokenizer.cls_token_id
            #     for idx in range(1, n_sentences):
            #         temp_input_ids[idx*config.max_sentence_length] = tokenizer.sep_token_id

            # num_pad_sentences = config.max_sentences - n_sentences
            # batch_input_ids.append(temp_input_ids +
            #                           [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            # batch_attention_mask.append(temp_attention_mask +
            #                                [config.pad_token_id] * (config.max_sentence_length * num_pad_sentences))
            # batch_token_type_ids.append([0] * data_args.max_seq_length)
            # secondary_labels.append([float(pos) for pos in sentence_positions] + [-100] * num_pad_sentences)
            # tmp_labels = np.eye(data_args.max_sentences)[sentence_positions]
            # tmp_labels = np.pad(tmp_labels, ((0, num_pad_sentences), (0, 0)))
            # secondary_labels.append(tmp_labels)
            # print(secondary_labels)
            secondary_labels.append([1., 0.] if examples['labels'][example_idx] == 0 else [0., 1.])
            
            tertiary_labels.append([1., 0.] if examples['labels'][example_idx] == 0 else [0., 1.])

        batch['input_ids'] = batch_input_ids
        batch['attention_mask'] = batch_attention_mask
        batch['token_type_ids'] = batch_token_type_ids
        batch['secondary_labels'] = secondary_labels
        batch['tertiary_labels'] = tertiary_labels

        return batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
                num_proc=data_args.num_workers,
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                num_proc=data_args.num_workers,
            )
    
    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                num_proc=data_args.num_workers,
            )

    # train_len = len(train_dataset['input_ids'])
    # eval_len = len(eval_dataset['input_ids'])
    # predict_len = len(predict_dataset['input_ids'])
    # train_count = 0
    # eval_count = 0
    # predict_count = 0
    # with open("data/example.txt", "r") as file:
    # # Read the first 100 lines
    #     for i, line in enumerate(file):
    #         if i < train_len:
    #             number = int(line.strip())
    #             train_count += number
    #         elif i < train_len+eval_len:
    #             number = int(line.strip())
    #             eval_count += number
    #         else:
    #             number = int(line.strip())
    #             predict_count += number
    # print(f"Avg Train Tokens: {train_count/train_len}, Avg Eval Tokens: {eval_count/eval_len}, Avg Predict Tokens: {predict_count/predict_len}")


    def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                mlm_logits = logits[3]
                secondary_logits = logits[4]
                tertiary_logits = logits[5]
            mlm_indices = torch.topk(mlm_logits, k=3, dim=-1)[1]
            # print(mlm_logits.shape)
            # print(sent_order_logits.shape)
            return (mlm_indices, secondary_logits, tertiary_logits)
    
    def compute_metrics(p: EvalPrediction):

        mlm_preds = p.predictions[0]
        secondary_preds = p.predictions[1]
        tertiary_preds = p.predictions[2]

        mlm_labels = p.label_ids[1][:, :mlm_preds.shape[1]]
        # secondary_labels = p.label_ids[2]
        # tertiary_labels = p.label_ids[3]
        logs_labels = p.label_ids[0]

        k = 3
        # Create an empty list to store intermediate results
        mlm_labels = mlm_labels[:, :, np.newaxis].repeat(k, axis=2)
        top_k_list = []

        # Loop through each row along dim=0
        count = 0
        for i in range(mlm_preds.shape[0]):
            # Check for condition
            mask = (mlm_labels[i, :, 0] != -100)
            condition = (mlm_labels[i] == mlm_preds[i]) & (mlm_labels[i, :, 0] != -100)[:, np.newaxis].repeat(k, axis=1)
            tokens = len(mlm_preds[i][condition])
            total = len(mlm_preds[i][mask])
            if total <= 0:
                count += 1
            top_k_list.append(tokens / total if total > 0 else 1.0)
        print(top_k_list)
        print('*' * 40)
        print(count)
        print('*' * 40)
        # secondary_list = [0 if np.argmax(secondary_preds[i]) == np.argmax(secondary_labels[i]) else 1 for i in range(secondary_preds.shape[0])]
        # tertiary_list = [0 if np.argmax(tertiary_preds[i]) == np.argmax(tertiary_labels[i]) else 1 for i in range(tertiary_preds.shape[0])]

        score = [secondary_preds[i][1] + tertiary_preds[i][1] + 1-top_k_list[i] for i in range(logs_labels.shape[0])]

        list_of_anomaly_score = list(filter(lambda x: x != -1, [score[i] if logs_labels[i] == 1 else -1 for i in range(logs_labels.shape[0])]))
        mean_of_anomaly_score = sum(list_of_anomaly_score) / len(list_of_anomaly_score)
        list_of_normal_score = list(filter(lambda x: x != -1, [score[i] if logs_labels[i] == 0 else -1 for i in range(logs_labels.shape[0])]))
        mean_of_normal_score = sum(list_of_normal_score) / len(list_of_normal_score)
        percentile =  np.percentile(list_of_anomaly_score, 1)

        logs_preds = [1 if s >= percentile else 0 for s in score]

        auroc = roc_auc_score(y_score=score, y_true=logs_labels)
        f1 = f1_score(y_pred=logs_preds, y_true=logs_labels)
        p = precision_score(y_pred=logs_preds, y_true=logs_labels)
        r = recall_score(y_pred=logs_preds, y_true=logs_labels)

        # score_str = ', '.join(score.tolist())
        # logs_labels_str = ', '.join(logs_labels.tolist())

        return {'auroc':auroc, 'f1':f1, 'r':r, 'p':p, 'mean of anomaly score':mean_of_anomaly_score, 'mean of normal score':mean_of_normal_score}
        # so_preds = so_preds.astype(int)
        # so_acc_list = []
        # for i in range(so_preds.shape[0]):
        # # Remove ignored index (special tokens)
        #     # true_predictions = np.argsort([
        #     #     p[0] for p, l in zip(so_preds[i], so_labels[i]) if l != -100
        #     # ])
        #     # true_predictions = [
        #     #     p for p, l in zip(so_preds[i], so_labels[i]) if l != 100
        #     # ]
        #     true_predictions = [
        #         p for p, l in zip(so_preds[i], so_labels[i]) if any(l)
        #     ]
        #     # print(true_predictions)
        #     # true_labels = [
        #     #     l for p, l in zip(so_preds[i], so_labels[i]) if l != 100
        #     # ]
        #     true_labels = [
        #         np.where(l == 1.)[0][0] for l in so_labels[i] if any(l)
        #     ]
        #     # print(true_labels)
        #     acc = top_k_accuracy_score(y_true=true_labels, y_score=true_predictions, labels=np.arange(data_args.max_sentences), k=2)
        #     # acc = accuracy_score(y_true=true_labels, y_pred=true_predictions)
        #     so_acc_list.append(acc)


    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator = DataCollatorForLogsPreTraining(tokenizer, pad_to_multiple_of=8)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # print(len(predictions))
        # print(predictions[0].shape)
        # print(predictions[1].shape)
        # predictions = np.argmax(predictions, axis=1)
        # output_predict_file = os.path.join(training_args.output_dir, "predictions.csv")
        # if trainer.is_world_process_zero():
        #     with open(output_predict_file, "w") as writer:
        #         for index, item in enumerate(predictions):
        #             writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()