#        BiSon
#
#   File:     bison_dd.sh

REPO_DIR=/path/to/repo
DATA=$REPO_DIR/data/dailydialog
EP=40
LR=1e-4
AVG=0.5
STD=0.6
OUTPUT_DIR=$REPO_DIR/local_models/dd.e$EP.$LR.gaussian.$AVG-$STD

mkdir -p $OUTPUT_DIR

python $REPO_DIR/run_bison.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA/train.tsv \
  --predict_file $DATA/dev.tsv \
  --train_batch_size 16 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 1 \
  --learning_rate $LR \
  --num_train_epochs $EP \
  --max_seq_length 510 \
  --max_part_a 410 \
  --max_gen_length 100 \
  --output_dir $OUTPUT_DIR \
  --seed 42 \
  --data_set 'daily_dialog' \
  --valid_gold $DATA/validation/ \
  --valid_every_epoch \
  --masking 'gen' \
  --masking_strategy 'gaussian' \
  --distribution_mean $AVG \
  --distribution_stdev $STD \
  --predict 'left2right' \
  >$OUTPUT_DIR/info.train.log 2>&1