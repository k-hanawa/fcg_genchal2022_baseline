# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import sacrebleu
import openpyxl

no_comment = '<NO_COMMENT>'


def manual_score(df):
    scores = df.iloc[:, 3].to_list()

    assert all(s in (0, 1, -1) for s in scores)

    ref_len = len(scores)
    hyp_len = len([s for s in scores if s != -1])
    tp = len([s for s in scores if s == 1])

    if hyp_len == 0:
        precion = 0
    else:
        precion = tp / hyp_len

    if ref_len == 0:
        recall = 0
    else:
        recall = tp / ref_len

    if precion + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precion * recall / (precion + recall)

    return precion, recall, f1, tp, hyp_len, ref_len


def bleu_score(df):
    hyps = df.iloc[:, 2].to_list()
    refs = df.iloc[:, 1].to_list()
    bleu_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp == no_comment:
            continue
        bleu = sacrebleu.sentence_bleu(hyp, [ref])
        bleu_list.append(bleu.score)

    bleu_sum = sum(bleu_list)
    hyp_len = len(bleu_list)
    ref_len = len(refs)

    if hyp_len == 0:
        bleu_precision = 0
    else:
        bleu_precision = bleu_sum / hyp_len / 100

    if ref_len == 0:
        bleu_recall = 0
    else:
        bleu_recall = bleu_sum / ref_len / 100

    if bleu_precision + bleu_recall == 0:
        bleu_f1 = 0
    else:
        bleu_f1 = 2 * bleu_precision * bleu_recall / (bleu_precision + bleu_recall)

    return bleu_precision, bleu_recall, bleu_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')

    args = parser.parse_args()

    df = pd.read_excel(args.input, index_col=0, engine='openpyxl')

    manual_precion, manual_recall, manual_f1, tp, hyp_len, ref_len = manual_score(df)
    bleu_precion, bleu_recall, bleu_f1 = bleu_score(df)

    print('-------')
    print('Basic stats')
    print('-------')
    print('Input file:', args.input)
    print('Num of reference feedback comments:', ref_len)
    print('Num of generated feedback comments:', hyp_len)
    print('Num of generated <NO_COMMENT>:', ref_len - hyp_len)
    print('-------')
    print('Manual evaluation')
    print('-------')
    print('Precision: {:.3f} ({} / {})'.format(manual_precion, tp, hyp_len))
    print('Recall: {:.3f} ({} / {})'.format(manual_recall, tp, ref_len))
    print('F1: {:.3f}'.format(manual_f1))
    print('-------')
    print('BLEU')
    print('-------')
    print('Precision: {:.3f}'.format(bleu_precion))
    print('Recall: {:.3f}'.format(bleu_recall))
    print('F1: {:.3f}'.format(bleu_f1))
    print('-------')


if __name__ == "__main__":
    main()
