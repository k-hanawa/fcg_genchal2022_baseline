#!/bin/bash

while getopts ":i:o:" OPT
do
  case $OPT in
    i)
        INPUT_DIR=${OPTARG%/};;
    o)
        OUTPUT_DIR=${OPTARG%/};;
    :)
        echo  "[ERROR] Option argument is undefined."
        exit 1
        ;;
    \?)
        echo "[ERROR] Undefined options."
        exit 1
        ;;
  esac
done


if [ -z $INPUT_DIR ]; then
    echo "-i arguments is required"
    exit 1
fi
if [ ! -d $INPUT_DIR ]; then
    echo "Input directory not exit"
    exit 1
fi

if [ -z $OUTPUT_DIR ]; then
    echo "-o arguments is required"
    exit 1
fi

python src/train_spm_model.py -i ${INPUT_DIR}/TRAIN.prep_feedback_comment.public.tsv -m ${INPUT_DIR}/spm

python src/convert_data.py -i ${INPUT_DIR}/TRAIN.prep_feedback_comment.public.tsv -m ${INPUT_DIR}/spm.model -o ${INPUT_DIR}/TRAIN
python src/convert_data.py -i ${INPUT_DIR}/DEV.prep_feedback_comment.public.tsv -m ${INPUT_DIR}/spm.model -o ${INPUT_DIR}/DEV

fairseq-preprocess \
    --trainpref ${INPUT_DIR}/TRAIN --validpref ${INPUT_DIR}/DEV \
    --source-lang src --target-lang com --joined-dictionary \
    --destdir ${OUTPUT_DIR}

cp ${INPUT_DIR}/TRAIN.err ${OUTPUT_DIR}/train.src-com.err
cp ${INPUT_DIR}/DEV.err ${OUTPUT_DIR}/valid.src-com.err
