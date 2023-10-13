export PYTHONPATH=":."
LAYOUT='p1'
MODEL_MAX_LENGTH=1024
MAX_SENTENCES=16
MAX_SENTENCE_LENGTH=64
DATASET_NAME_1='hdfs'
DATASET_NAME_2='bgl'
DATASET_NAME_3='tbird'
MLM=40
TOP_K=3
LR=1e-4
STEPS=1000
SAMPLE=1000
SR=20

# python3 models/hat/convert_roberta_to_htf.py --layout ${LAYOUT} --max_sentences ${MAX_SENTENCES} --max_sentence_length ${MAX_SENTENCE_LENGTH} \

accelerate launch --main_process_port 29500 evaluation/run_logs_ad3.py \
    --model_name_or_path data/PLMs/hat-${LAYOUT}-roberta-${MAX_SENTENCES}-${MAX_SENTENCE_LENGTH}-${MODEL_MAX_LENGTH} \
    --dataset_name data/${DATASET_NAME_2} \
    --output_dir data/PLMs/hat/${LAYOUT}-roberta-${MAX_SENTENCES}-${MAX_SENTENCE_LENGTH}-${MODEL_MAX_LENGTH}/${DATASET_NAME_2}/mlm${MLM}-top${TOP_K}-so-sr${SR}-${LR}-win50\
    --do_train \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps ${STEPS} \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps ${STEPS} \
    --label_names labels primary_labels secondary_labels tertiary_labels \
    --do_predict \
    --logging_strategy steps \
    --logging_steps ${STEPS} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --metric_for_best_model 'auroc' \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_workers 120 \
    --mlm ${MLM} \
    --topk ${TOP_K} \
    --sr ${SR} \
    --max_steps 50000 \
    # --max_train_samples ${SAMPLE} \
    # --max_eval_samples ${SAMPLE} \
    # --max_predict_samples ${SAMPLE} \
    # --num_train_epochs 10 \