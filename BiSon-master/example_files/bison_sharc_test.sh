#        BiSon
#
#   File:     bison_sharc_test.sh
REPO_DIR=/path/to/repo
DATA=$REPO_DIR/data/sharc1-official
OUTPUT_DIR=$REPO_DIR/sharc.e20.3e-5.gaussian.0.5-0.6

python $REPO_DIR/run_bison.py \
  --bert_model bert-base-uncased \
  --do_predict \
  --do_lower_case \
  --predict_file $DATA/sharc_devtest.json \
  --max_seq_length 400 \
  --max_part_a 350 \
  --max_gen_length 50 \
  --data_set 'sharc' \
  --output_dir $OUTPUT_DIR \
  --valid_gold $DATA/sharc_devtest.json \
  --masking 'gen' \
  --predict 'left2right' \
  --predict_batch_size 1 \
  >$OUTPUT_DIR/info.test.log 2>&1