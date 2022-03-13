# -*- coding: utf-8 -*-

import argparse
import sentencepiece as spm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--model-prefix', '-m', type=str)
    parser.add_argument('--vocab-size', '-v', type=int, default=2000)
    args = parser.parse_args()

    comments = []
    with open(args.input) as fi:
        for i, line in enumerate(fi):
            source, offset, comment = line.rstrip('\n').split('\t')
            comments.append(comment)

    spm.SentencePieceTrainer.train(sentence_iterator=iter(comments), model_prefix=args.model_prefix, vocab_size=args.vocab_size)

if __name__ == "__main__":
    main()
