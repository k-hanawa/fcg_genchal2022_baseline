# Baseline system for the feedback comment generation@GenChal2022 
This is a baseline system for the feedback comment generation@GenChal2022
This system is an encoder-decoder with a copy mechanism based on a pointer generator netowork.
The implementation is based on [fairseq](https://github.com/pytorch/fairseq).

## Requirements

- Python 3.7+
- Install the required libraries:
```bash
$ pip install -r requirements.txt
```

## Dataset

Download FCG dataset from https://fcg.sharedtask.org/ and unzip to `data/train_dev`.

## Usage

### Preprocess

Convert to fairseq compatible format using the following shell script.
```bash
$ src/preprocess_train_dev.sh -i data/train_dev -o data/fcg-bin
```

### Train

Train a model based on a pointer generator netowork.
```bash
$ fairseq-train \
    data/fcg-bin --task fcg --arch fcg \
    --optimizer adam --lr 0.001 \
    --max-tokens 1024 --max-epoch 50 \
    --eval-bleu --eval-bleu --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric\
    --user-dir src \
    --save-dir results/fcg_baseline
```

### Generate
First, output in the form of one tokenized comment per line by `fairseq-generate`.
```bash
$ fairseq-generate \
    data/fcg-bin --task fcg \
    --path results/fcg_baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --user-dir src --gen-subset valid | grep '^H' | LC_ALL=C sort -V | cut -f3 > results/fcg_baseline/DEV.prep_feedback_comment.out
```
Then, convert to a submission format.
```bash
$ python src/postprocess.py -i data/train_dev/DEV.prep_feedback_comment.public.tsv -s results/fcg_baseline/DEV.prep_feedback_comment.out -m data/train_dev/spm.model -o results/fcg_baseline/DEV.prep_feedback_comment.out.tsv
```

### Evaluate
Compute precision, recall and f1 score based on sentence BLEU using [SacreBLEU](https://github.com/mjpost/sacrebleu).
You should provide **detokenized** system outputs file.
```bash
$ python src/evaluate_bleu.py -i results/fcg_baseline/DEV.prep_feedback_comment.out.tsv -r data/train_dev/DEV.prep_feedback_comment.public.tsv
BLEU precision: 46.341357634752534
BLEU recall: 46.341357634752534
BLEU F1: 46.341357634752534
```
`--verbose/-v` option shows the system output length, the reference length and the bleu score of each sentence.
```bash
$ python src/evaluate_bleu.py -i results/fcg_baseline/DEV.prep_feedback_comment.out.tsv -r data/train_dev/DEV.prep_feedback_comment.public.tsv -v
...
Input: So restaurants divide the area [[in to]] two sections .
System: Choose a <preposition> that indicates the beneficiary of an action instead of the <preposition> <<of>>.
Reference: It seems to be a careless mistake, but use the <preposition> that expresses the <prepositions> <<in>> and <<to>> in one word.
BLEU: 14.889568593912923

System length: 170
Reference length: 170

BLEU precision: 46.341357634752534
BLEU recall: 46.341357634752534
BLEU F1: 46.341357634752534
```
