import argparse
import csv
from nltk import word_tokenize
from torch.utils.data import WeightedRandomSampler

def write_tsv_file(input_csv_filename, output_tsv_filename, balance_data=False):
    input_text = []
    output_label = []
    with open(input_csv_filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            #import pdb
            #pdb.set_trace()
            _, title, context, question, level, label = row
            #sentence = ' '.join(word_tokenize(question.lower())) + ' ' + ' '.join(word_tokenize(title.lower())) + ' ' + ' '.join(word_tokenize(context.lower()))
            sentence = ' '.join(word_tokenize(question.lower())) + ' ' + ' '.join(word_tokenize(title.lower()))
            if label == 's':
                input_text.append(sentence)
                output_label.append(1)
            elif label == 'g':
                input_text.append(sentence)
                output_label.append(0)
    if balance_data:
        data_weights = []
        for i in range(len(input_text)):
            if output_label[i] == 1:
                data_weights.append(1.0)
            else:
                data_weights.append(2.0)
        data_indices = list(WeightedRandomSampler(data_weights, num_samples=int(len(input_text)*1.5), replacement=True))
    else:
        data_indices = range(len(input_text))

    with open(output_tsv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        i = 0
        for index in data_indices:
            csv_writer.writerow(['a', i, output_label[index], input_text[index]])
            i += 1

def main(args):
    write_tsv_file(args.train_csv_file, args.output_train_tsv_file, balance_data=True)
    write_tsv_file(args.val_csv_file, args.output_val_tsv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_file", type=str)
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--output_train_tsv_file", type=str)
    parser.add_argument("--output_val_tsv_file", type=str)
    args = parser.parse_args()
    main(args)
