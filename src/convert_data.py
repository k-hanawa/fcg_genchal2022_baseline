# -*- coding: utf-8 -*-

import argparse
import spacy
import bisect
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output-prefix', '-o', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")

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
            doc_s = nlp(source)
            doc_c = nlp(comment)

            si, ei = offset.split(':')
            si, ei = si.strip(), ei.strip()
            si, ei = int(si), int(ei)
            idxs = [t.idx for t in doc_s]
            siw = bisect.bisect_left(idxs, si)
            eiw = bisect.bisect_left(idxs, ei)

            fo_src.write('{}\n'.format(' '.join(t.lower_ for t in doc_s)))
            fo_com.write('{}\n'.format(' '.join(t.lower_ for t in doc_c)))
            fo_err.write('{} {}\n'.format(siw, eiw))


if __name__ == "__main__":
    main()
