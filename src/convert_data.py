# -*- coding: utf-8 -*-

import argparse
import bisect
from itertools import accumulate
from tqdm import tqdm
import sentencepiece as spm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output-prefix', '-o', type=str)
    parser.add_argument('--spm-model', '-m', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    with open(args.input) as fi,\
         open(args.output_prefix + '.src', 'w') as fo_src,\
         open(args.output_prefix + '.com', 'w') as fo_com,\
         open(args.output_prefix + '.err', 'w') as fo_err:
        for i, line in tqdm(enumerate(fi)):
            if args.test:
                source, offset = line.rstrip('\n').split('\t')
                comment = 'dummy'
            else:
                source, offset, comment = line.rstrip('\n').split('\t')
            source_toks = source.split(' ')
            comment_toks = sp.encode(comment, out_type=str)

            si, ei = offset.split(':')
            si, ei = si.strip(), ei.strip()
            si, ei = int(si), int(ei)
            source_tok_lens = [len(t) for t in source_toks]
            idxs = [j + sl for j, sl in enumerate(accumulate([0] + source_tok_lens))]
            siw = bisect.bisect_left(idxs, si)
            eiw = bisect.bisect_left(idxs, ei)

            fo_src.write('{}\n'.format(' '.join(t for t in source_toks)))
            fo_com.write('{}\n'.format(' '.join(t for t in comment_toks)))
            fo_err.write('{} {}\n'.format(siw, eiw))


if __name__ == "__main__":
    main()
