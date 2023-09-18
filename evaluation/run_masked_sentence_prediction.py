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
""" Finetuning sentence classification models"""
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from models.hat import HATConfig, HATTokenizer, HATForMultipleChoice
from models.longformer import LongformerTokenizer, LongformerForMultipleChoice
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from language_modelling.data_collator import DataCollatorForMultipleChoice

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


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
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    sentence_bert_path: Optional[str] = field(
        default=None, #'all-MiniLM-L6-v2',
        metadata={
            "help": "The name of the sentence-bert to use (via the transforrmers library)"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pooling: str = field(
        default='first', metadata={"help": "Which pooling method to use (first or last token)."}
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
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="train",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_eval:
        eval_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="validation",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_predict:
        predict_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="test",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    # Labels
    num_labels = 5
    random.seed(training_args.seed)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'hat' in model_args.model_name_or_path:
        config = HATConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="sentence-mcqa",
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
        model = HATForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="sentence-mcqa",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.max_sentence_size = 128
        config.max_sentence_length = 128
        config.max_sentences = data_args.max_sentences
        tokenizer = LongformerTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = LongformerForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Extract sentences from training dataset
    batch = tokenizer(
        train_dataset["text"],
        padding="max_length",
        max_length=data_args.max_seq_length,
        truncation=True,
    )
    sentences = [copy.deepcopy(example[idx * config.max_sentence_size + 1: (idx+1) * config.max_sentence_size]) for example in batch['input_ids']
                 for idx in range(int(len(example) / config.max_sentence_size)) if example[idx * config.max_sentence_size] != tokenizer.pad_token_id]
    del batch

    # Compute sentence embeddings
    sentence_embedder = None
    if data_args.sentence_bert_path:
        sentences_text = tokenizer.batch_decode(sentences)
        sentences_text = [sentence.replace(' [PAD]', '') for sentence in sentences_text]
        sentence_embedder = SentenceTransformer(data_args.sentence_bert_path)
        sentence_embeddings = sentence_embedder.encode(sentences_text, show_progress_bar=True, normalize_embeddings=True)
        logger.info(f'{len(sentence_embeddings)} sentences were embedded using {data_args.sentence_bert_path}!')

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

        mc_input_ids = []
        labels = []
        for example_idx, input_ids in enumerate(batch['input_ids']):
            # pad example to full length
            batch['input_ids'][example_idx] = batch['input_ids'][example_idx] + [tokenizer.pad_token_id] * ((config.max_sentences*config.max_sentence_size) - len(batch['input_ids'][example_idx]))
            # count true sentences
            sentence_ids = input_ids[::config.max_sentence_size]
            n_sentences = sum(np.asarray(sentence_ids) != config.pad_token_id)
            # select a random sentence to mask
            masked_sentence_id = random.choice(range(n_sentences-1))
            # collect masked sentence input ids
            masked_sentence_input_ids = copy.deepcopy(batch['input_ids'][example_idx][config.max_sentence_size * masked_sentence_id:config.max_sentence_size * (masked_sentence_id+1)])
            # mask sentence input ids
            batch['input_ids'][example_idx][(config.max_sentence_size * masked_sentence_id)+1:config.max_sentence_size * (masked_sentence_id+1)] = [tokenizer.mask_token_id] * (config.max_sentence_size - 1)
            example_input_ids = copy.deepcopy(batch['input_ids'][example_idx])
            # choose correct choice position
            correct_choice_id = random.choice(range(5))
            if sentence_embedder is not None:
                # select negative samples based on sentence embeddings
                sentence_embedding = sentence_embedder.encode(tokenizer.decode(masked_sentence_input_ids[1:]), normalize_embeddings=True)
                most_similar_ids = list(np.argsort(util.dot_score(sentence_embedding, sentence_embeddings).numpy()[0])[-21:-1])
            negative_sample_id = -1
            for i in range(5):
                choice_input_ids = copy.deepcopy(example_input_ids)
                if i == correct_choice_id:
                    choice_input_ids[config.max_sentence_size * (config.max_sentences - 1) + 1:config.max_sentence_size * config.max_sentences] = masked_sentence_input_ids[1:]
                else:
                    if sentence_embedder is None:
                        # select negative randomly out of all sentences
                        negative_sample_id = random.choice(range(len(sentences)))
                        negative_sample = copy.deepcopy(sentences[negative_sample_id])
                    else:
                        # select negative randomly out of top-20 similar sentences
                        if negative_sample_id != -1:
                            most_similar_ids.remove(negative_sample_id)
                        negative_sample_id = random.choice(most_similar_ids)
                        negative_sample = copy.deepcopy(sentences[random.choice(most_similar_ids)])

                    choice_input_ids[(config.max_sentence_size * (config.max_sentences - 1)) + 1:config.max_sentence_size * config.max_sentences] = negative_sample
                    choice_input_ids[config.max_sentence_size * (config.max_sentences - 1)] = tokenizer.sep_token_id \
                        if config.model_type == 'longformer' else tokenizer.cls_token_id
                mc_input_ids.append(choice_input_ids)
                labels.append(correct_choice_id)

        batch['input_ids'] = mc_input_ids
        batch['attention_mask'] = [[1 if idx != tokenizer.pad_token_id else 0 for idx in example] for example in batch['input_ids']]
        batch['token_type_ids'] = [[0 for _ in example] for example in batch['input_ids']]
        batch['labels'] = labels

        return {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in batch.items()}

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
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        acc = accuracy_score(y_true=p.label_ids, y_pred=preds)

        return {'accuracy_score': acc}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = DataCollatorForMultipleChoice(tokenizer)
    elif training_args.fp16:
        data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()