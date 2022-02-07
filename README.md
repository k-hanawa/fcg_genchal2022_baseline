# Baseline system for the feedback comment generation@GenChal2022 
This is a baseline system for the feedback comment generation@GenChal2022
This system is an encoder-decoder with a copy mechanism based on a pointer generator netowork.
The implementation is based on fairseq.

## Requirements

- Python 3.7+
- Install the required libraries:
```bash
pip install -r requirements.txt
```

## Resources

TBW

## Usage

### Preprocess

```bash
for split (TRAIN DEV); do python src/convert_data.py -i data/train_dev/${split}.prep_feedback_comment.public.tsv -o data/fcg/${split}; done
python src/convert_data.py -i data/train_dev/TEST.prep_feedback_comment.public.tsv -o data/fcg/TEST --test
```

```bash
fairseq-preprocess \
    --trainpref data/fcg/TRAIN --validpref data/fcg/DEV --testpref data/fcg/TEST \
    --source-lang src --target-lang com --joined-dictionary \
    --destdir data/fcg-bin
```

```bash
cp data/fcg/TRAIN.err data/fcg-bin/train.src-com.err
cp data/fcg/DEV.err data/fcg-bin/valid.src-com.err
cp data/fcg/TEST.err data/fcg-bin/test.src-com.err
```

### Train
```bash
fairseq-train \
    data/fcg-bin --task fcg --arch fcg \
    --optimizer adam --lr 0.001 \
    --max-tokens 1024 --max-epoch 50 \
    --eval-bleu --eval-bleu --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric\
    --user-dir src \
    --save-dir results/fcg_baseline
```

### Generate
```bash
fairseq-generate \
    data/fcg-bin --task fcg \
    --path results/fcg_baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --user-dir src --gen-subset test | grep '^H' | LC_ALL=C sort -V | cut -f3 > results/fcg_baseline/generation_resutls.test
```
