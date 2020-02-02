#        BiSon
#
#   File:     bison_dd_test.sh

REPO_DIR=/path/to/repo
DATA=$REPO_DIR/data/dailydialog
OUTPUT_DIR=$REPO_DIR/local_models/dd.e40.1e-4.gaussian.0.5-0.6.3

python $REPO_DIR/run_bison.py \
  --bert_model bert-base-uncased \
  --do_predict \
  --do_lower_case \
  --predict_file $DATA/test.tsv \
  --max_seq_length 510 \
  --max_part_a 410 \
  --max_gen_length 100 \
  --output_dir $OUTPUT_DIR \
  --data_set 'daily_dialog' \
  --valid_gold $DATA/test/ \
  --masking 'max' \
  --predict 'left2right' \
  --predict_batch_size 1 \
  >$OUTPUT_DIR/info.test.log 2>&1
