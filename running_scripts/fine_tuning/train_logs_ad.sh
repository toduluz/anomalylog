export PYTHONPATH=":."
LAYOUT='p1'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=64
MAX_SENTENCE_LENGTH=64
DATASET_NAME_1='hdfs'
DATASET_NAME_2='bgl'
DATASET_NAME_3='tbird'
python3 models/hat/convert_roberta_to_htf.py --layout ${LAYOUT} --max_sentences ${MAX_SENTENCES} --max_sentence_length ${MAX_SENTENCE_LENGTH} \

accelerate launch evaluation/run_logs_ad.py \
    --model_name_or_path data/PLMs/hat-${LAYOUT}-roberta-${MAX_SENTENCES}-${MAX_SENTENCE_LENGTH}-${MODEL_MAX_LENGTH} \
    --dataset_name data/${DATASET_NAME_3} \
    --output_dir data/PLMs/hat/${LAYOUT}-roberta-${MAX_SENTENCES}-${MAX_SENTENCE_LENGTH}-${MODEL_MAX_LENGTH}/${DATASET_NAME_3}/loss3-mlm-shuf-rev-mlm40-win50\
    --do_train \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps .1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps .1 \
    --label_names labels primary_labels secondary_labels tertiary_labels \
    --do_predict \
    --logging_strategy steps \
    --logging_steps .1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --metric_for_best_model 'auroc' \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --max_steps 1000000 \
    # --max_train_samples 100000 \
    # --max_eval_samples 32 \
    # --max_predict_samples 32 \
    # --num_train_epochs 10 \