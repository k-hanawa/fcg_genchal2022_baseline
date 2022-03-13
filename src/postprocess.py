# -*- coding: utf-8 -*-

import argparse
import sentencepiece as spm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system-input', '-i', type=str)
    parser.add_argument('--system-output', '-s', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--spm-model', '-m', type=str)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    with open(args.system_input) as fsi:
        lines_si = fsi.readlines()
    with open(args.system_output) as fso:
        lines_so = fso.readlines()
    assert len(lines_si) == len(lines_so)

    with open(args.output, 'w') as fo:
        for lsi, lso in zip(lines_si, lines_so):
            source, offset, *_ = lsi.rstrip('\n').split('\t')
            output = sp.decode(lso.rstrip('\n').split(' '))
            fo.write('{}\t{}\t{}\n'.format(source, offset, output))


if __name__ == "__main__":
    main()
