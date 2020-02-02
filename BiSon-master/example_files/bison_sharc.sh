#        BiSon
#
#   File:     bison_sharc.sh

REPO_DIR=/path/to/repo
DATA=$REPO_DIR/data/sharc1-official
EP=20
LR=3e-5
AVG=0.5
STD=0.6
OUTPUT_DIR=$REPO_DIR/local_models/sharc.e$EP.$LR.gaussian.$AVG-$STD

mkdir -p $OUTPUT_DIR

python $REPO_DIR/run_bison.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $DATA/sharc_train.json \
  --predict_file $DATA/sharc_dev.json \
  --train_batch_size 16 \
  --gradient_accumulation_steps 3 \
  --predict_batch_size 1 \
  --learning_rate $LR \
  --num_train_epochs $EP \
  --max_seq_length 400 \
  --max_part_a 350 \
  --max_gen_length 50 \
  --output_dir $OUTPUT_DIR \
  --seed 42 \
  --data_set 'sharc' \
  --valid_gold $DATA/sharc_dev_answer.json \
  --valid_every_epoch \
  --masking 'gen' \
  --masking_strategy 'gaussian' \
  --distribution_mean $AVG \
  --distribution_stdev $STD \
  --predict 'left2right' \
  >$OUTPUT_DIR/info.train.log 2>&1