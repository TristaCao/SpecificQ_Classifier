import argparse
import sys
import pickle as p
import numpy as np

def main(args):
    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.array(word_embeddings)
    word2index = p.load(open(args.vocab, 'rb'))
    output_file = open(args.output_filename, 'w')
    for word, index in word2index.items():
        output_file.write('%s %s\n' % (word, ' '.join(str(v) for v in word_embeddings[index])))
    output_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--vocab", type = str)
    argparser.add_argument("--word_embeddings", type = str)
    argparser.add_argument("--output_filename", type = str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)

