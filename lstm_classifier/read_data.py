import argparse
import csv
from nltk import word_tokenize

def write_tsv_file(input_csv_filename, output_tsv_filename):
    input_text = []
    output_label = []
    with open(input_csv_filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            #import pdb
            #pdb.set_trace()
            _, title, context, question, level, label = row
            sentence = ' '.join(word_tokenize(question.lower()))
            if label == 's':
                input_text.append(sentence)
                output_label.append(1)
            elif label == 'g':
                input_text.append(sentence)
                output_label.append(0)

    with open(output_tsv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        for i in range(len(input_text)):
            csv_writer.writerow([input_text[i], output_label[i]])

def main(args):
    write_tsv_file(args.train_csv_file, args.output_train_tsv_file)
    write_tsv_file(args.val_csv_file, args.output_val_tsv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_file", type=str)
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--output_train_tsv_file", type=str)
    parser.add_argument("--output_val_tsv_file", type=str)
    args = parser.parse_args()
    main(args)
