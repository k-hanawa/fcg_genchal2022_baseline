# -*- coding: utf-8 -*-

import argparse
import sacrebleu

no_comment = '<NO_COMMENT>'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--reference', '-r', type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    hyps = []
    with open(args.input) as fi:
        for l in fi:
            _, _, hyp = l.rstrip('\n').split('\t')
            hyps.append(hyp)
    refs = []
    inputs = []
    with open(args.reference) as fr:
        for l in fr:
            cols = l.rstrip('\n').split('\t')
            if len(cols) == 3:
                source, offset, ref = cols
            else:
                source, offset = cols
                ref = '<NO_REFERENCE>'
            si, ei = offset.split(':')
            si, ei = int(si), int(ei)
            inputs.append('{}[[{}]]{}'.format(source[:si], source[si:ei], source[ei:]))
            refs.append(ref)
    assert len(hyps) == len(refs) == len(inputs)

    n_skip = 0
    bleu_list = []
    for hyp, ref in zip(hyps, refs):
        if hyp == no_comment:
            n_skip += 1
            continue
        bleu = sacrebleu.sentence_bleu(hyp, [ref])
        bleu_list.append(bleu.score)

    bleu_sum = sum(bleu_list)
    hyp_len = len(hyps) - n_skip
    ref_len = len(refs)

    if hyp_len == 0:
        bleu_prec = 0
    else:
        bleu_prec = bleu_sum / hyp_len

    if ref_len == 0:
        bleu_rec = 0
    else:
        bleu_rec = bleu_sum / ref_len

    if bleu_prec + bleu_rec == 0:
        bleu_f1 = 0
    else:
        bleu_f1 = 2 * bleu_prec * bleu_rec / (bleu_prec + bleu_rec)

    if args.verbose:
        for hyp, ref, inp, bleu in zip(hyps, refs, inputs, bleu_list):
            print('Input:', inp)
            print('System:', hyp)
            print('Reference:', ref)
            print('BLEU:', bleu)
            print()
        print('System length:', hyp_len)
        print('Reference length:', ref_len)
        print()

    print('BLEU precision:', bleu_prec)
    print('BLEU recall:', bleu_rec)
    print('BLEU F1:', bleu_f1)

if __name__ == "__main__":
    main()
