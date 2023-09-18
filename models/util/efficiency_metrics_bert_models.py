import copy
import warnings

import torch
import time
from transformers import AutoConfig

import numpy as np
from data import DATA_DIR
from models.hat import HATForMaskedLM, HATConfig, HATForSequenceClassification, \
    HATForMultipleChoice
from models.longformer import LongformerForMaskedLM, LongformerModelForSequenceClassification, LongformerForMultipleChoice

warnings.filterwarnings("ignore")

LAYOUTS = {
    # 'f6': 'S|S|S|S|S|S',
    # 'f8': 'S|S|S|S|S|S',
    's1': 'SD|SD|SD|SD|SD|SD',
    's2': 'S|SD|D|S|SD|D|S|SD|D',
    'p1': 'S|SD|S|SD|S|SD|S|SD',
    'p2': 'S|S|SD|S|S|SD|S|S|SD',
    'e1': 'SD|SD|SD|S|S|S|S|S|S',
    'e2': 'S|SD|D|S|SD|D|S|S|S|S',
    'l1': 'S|S|S|S|S|S|SD|SD|SD',
    'l2': 'S|S|S|S|S|SD|D|S|SD|D',
    # 'b1': 'S|S|SD|D|S|SD|D|S|S|S',
    # 'b2': 'S|S|SD|SD|SD|S|S|S|S',
}


TASK_MODEL = {'lm': {'longformer': LongformerForMaskedLM, 'hilm': HATForMaskedLM},
              'doc_cls': {'longformer': LongformerModelForSequenceClassification, 'hilm': HATForSequenceClassification},
              'mc_qa': {'longformer': LongformerForMultipleChoice, 'hilm': HATForMultipleChoice},
              }


def test_memory_usage(model, steps=40, batch_size=2, seq_length=1024,  mode='test', task_type='lm'):
    model.to('cuda')
    if task_type != 'mc_qa':
        input_ids = torch.randint(1, 30000, (batch_size, seq_length), dtype=torch.long).to('cuda')
        input_ids[:, :: 128] = model.config.bos_token_id
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int).to('cuda')
    else:
        input_ids = torch.randint(1, 30000, (batch_size, 2, seq_length), dtype=torch.long).to('cuda')
        input_ids[:, :: 128] = model.config.bos_token_id
        attention_mask = torch.ones((batch_size, 2, seq_length), dtype=torch.int).to('cuda')
    if mode == 'train':
        if task_type == 'lm':
            labels = input_ids.clone()
        else:
            labels = torch.ones((batch_size, ), dtype=torch.int).long().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    max_time = []
    max_mem = []
    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        if mode == 'train':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.time()
        total_time = (end - start)
        max_time.append(total_time)
        max_mem.append(torch.cuda.max_memory_allocated() / 1e9)

    return np.mean(max_mem), np.mean(max_time)


def efficiency_metrics():
    MAX_SENTENCE_LENGTH = 128
    CONFIGS = [{'num_hidden_layers': 6,
                'hidden_size': 256,
                'intermediate_size': 1024,
                'num_attention_heads': 4},
               # {'num_hidden_layers': 6,
               #  'hidden_size': 768,
               #  'intermediate_size': 3072,
               #  'num_attention_heads': 12}
               ]

    for mode in ['train', 'test']:
        print(F'MODE: {mode.upper()}')
        for task in ['lm', 'doc_cls', 'mc_qa']:
            for CONFIG in CONFIGS:
                print('-' * 150)
                print(F'TASK: {task.upper()}\t'
                      F'NUM LAYERS: {CONFIG["num_hidden_layers"]}\t'
                      F'NUM HIDDEN: {CONFIG["hidden_size"]}\t'
                      F'ATTENTION HEADS: {CONFIG["num_attention_heads"]}')
                print('-' * 150)

                for max_sentences in [8]:
                    print('-' * 150)
                    print(F'MAX SEQ LENGTH: {int(max_sentences * MAX_SENTENCE_LENGTH)}')
                    print('-' * 150)

                    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
                    lf_config.num_hidden_layers = CONFIG['num_hidden_layers']
                    # Transformer parameters
                    lf_config.hidden_size = CONFIG['hidden_size']
                    lf_config.intermediate_size = CONFIG['intermediate_size']
                    lf_config.num_attention_heads = CONFIG['num_attention_heads']
                    # Vocabulary parameters
                    lf_config.vocab_size = 32000
                    lf_config.type_vocab_size = 2
                    lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * max_sentences)
                    lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * max_sentences) + 2
                    lf_config.attention_window = [128] * CONFIG['num_hidden_layers']
                    lf_config.max_sentence_length = MAX_SENTENCE_LENGTH
                    lf_config.max_sentences = max_sentences
                    lf_config.cls_token_id = 100
                    # load dummy longformer model
                    htf_model = TASK_MODEL[task]['longformer'].from_config(lf_config)
                    model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
                    model_total_params = model_total_params / 1e6
                    memory_use, time_use = test_memory_usage(htf_model, seq_length=lf_config.model_max_length, mode=mode, task_type=task)
                    lf_mem_use = copy.deepcopy(memory_use)
                    lf_time_use = copy.deepcopy(time_use)
                    print(f'Longformer model has {model_total_params:.1f}M number of parameters '
                          f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
                    print('-' * 150)

                    for layout in LAYOUTS:
                        ENCODER_LAYOUT = {}
                        for idx, block_pattern in enumerate(LAYOUTS[layout].split('|')):
                            ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                                        "document_encoder": True if 'D' in block_pattern else False}

                        # load dummy config and change specifications
                        htf_config = HATConfig.from_pretrained(f'{DATA_DIR}/hat')
                        # Text length parameters
                        htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
                        htf_config.max_sentences = max_sentences
                        htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
                        htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * max_sentences)
                        htf_config.num_hidden_layers = len(ENCODER_LAYOUT.keys())
                        # Transformer parameters
                        htf_config.hidden_size = CONFIG['hidden_size']
                        htf_config.intermediate_size = CONFIG['intermediate_size']
                        htf_config.num_attention_heads = CONFIG['num_attention_heads']
                        htf_config.encoder_layout = ENCODER_LAYOUT
                        # Vocabulary parameters
                        htf_config.vocab_size = 32000
                        htf_config.type_vocab_size = 2

                        # load dummy hat model
                        htf_model = TASK_MODEL[task]['hilm'].from_config(htf_config)
                        model_total_params = sum(p.numel() for p in htf_model.hat.parameters() if p.requires_grad)
                        model_total_params = model_total_params / 1e6
                        memory_use, time_use = test_memory_usage(htf_model, seq_length=int(MAX_SENTENCE_LENGTH * max_sentences), mode=mode, task_type=task)
                        mem_gains = (lf_mem_use / memory_use) - 1
                        time_gains = (lf_time_use / time_use) - 1
                        print(f'Hi-transformer model with layout {layout} has {model_total_params:.1f}M number of parameters '
                              f'{memory_use:.2f}GB peak memory use (-{mem_gains*100:.2f}%) and {time_use:.3f} batch/second (-{time_gains*100:.2f}%)!')


if __name__ == '__main__':
    efficiency_metrics()
