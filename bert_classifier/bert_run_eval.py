import torch
import numpy as np
import pickle
from collections import defaultdict

from sklearn.metrics import matthews_corrcoef, confusion_matrix

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tools import *
from multiprocessing import Pool, cpu_count
import convert_examples_to_features
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "genspec_question_generation/genspec-data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'genspec-classifier-epoch3.tar.gz'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'genspec-classifier'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = 'genspec_question_generation/outputs/%s/' % TASK_NAME

# The directory where the evaluation reports will be written to.
REPORTS_DIR = 'genspec_question_generation/reports/%s_evaluation_report/' % TASK_NAME

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'genspec_question_generation/cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
#MAX_SEQ_LENGTH = 128
MAX_SEQ_LENGTH = 32

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += '/report_%d' % (len(os.listdir(REPORTS_DIR)))
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += '/report_%d' % (len(os.listdir(REPORTS_DIR)))
    os.makedirs(REPORTS_DIR)

def get_eval_report(task_name, labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    print('F1-score: %.4f Precision: %.4f Recall: %.4f Accuracy: %.4f' % (f1_score(labels, preds, average='micro'), precision_score(labels, preds, average='micro'), recall_score(labels, preds, average='micro'), accuracy_score(labels, preds)))
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return {
        "task": task_name,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(task_name, labels, preds)


def bert_evaluate(args):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

    processor = BinaryClassificationProcessor()
    eval_examples = processor.get_dev_examples(DATA_DIR)
    label_list = processor.get_labels() # [0, 1] for binary classification
    num_labels = len(label_list)
    eval_examples_len = len(eval_examples)

    label_map = {label: i for i, label in enumerate(label_list)}
    eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

    eval_features = []
    for eval_example in eval_examples_for_processing:
        eval_features.append(convert_examples_to_features.convert_example_to_feature(eval_example))

    all_guids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_guids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR,
                                                          num_labels=len(label_list))
    model.to(device)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    print_guids = []
    guid_preds_dict = defaultdict(None)
    for guids, input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
        print_guids += [guid for guid in guids.numpy()]

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)

    result['eval_loss'] = eval_loss

    output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in (result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    for i in range(len(preds)):
        guid_preds_dict[print_guids[i]] = preds[i]

    dev_tsv_file = open(os.path.join(DATA_DIR, 'dev.tsv'), 'r')
    dev_tsv_preds_file = open(os.path.join(DATA_DIR, 'dev.preds.tsv'), 'w')
    for line in dev_tsv_file.readlines():
       _, curr_guid, true_label, sentence = line.strip('\n').split('\t')  
       dev_tsv_preds_file.write('%s\t%s\t%s\t%s\n' % (curr_guid, true_label, guid_preds_dict[int(curr_guid)], sentence))

if __name__ == "__main__":
    bert_evaluate(None)
