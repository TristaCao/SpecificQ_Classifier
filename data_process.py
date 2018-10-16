import sys
import csv
from csv import reader, writer
import torch
import nltk
from nltk.corpus import stopwords
from random import shuffle

def main(args):
    q_context = {}
    q_score = {}
    with open("baby_train_question.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            question_a = row[10]
            question_b = row[11]
            score = float(row[6])
            context = row[8]
            choice = row[5]
            
            # add question -> context to q_context
            if question_a not in q_context:
                q_context[question_a] = context
            if question_b not in q_context:
                q_context[question_b] = context
                
            # calculate score
            score_a = 0
            socre_b = 0
            if choice == "Question A is more specific":
                score_a = score
                score_b = 1-score
            elif choice == "Question B is more specific":
                score_b = score
                score_a = 1 - score
            elif choice == "Both are at the same level of specificity":
                score_a = 0.5
                score_b = 0.5
            
            # assign question->score to q_score
            if question_a in q_score:
                q_score[question_a] = (q_score[question_a]+score_a)*0.5
            else:
                q_score[question_a] = score_a
                
            if question_b in q_score:
                q_score[question_b] = (q_score[question_b]+score_b)*0.5
            else:
                q_score[question_b] = score_b
    
    outputs = []
    for question in q_context:
        output = []
        output.append(question)
        output.append(q_context[question])
        output.append(q_score[question])
        outputs.append(output)
    
    random.shuffle(outputs)
    with open("processed_data.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["question", "context", "score"])
        for r in outputs:
            w.writerow(r)
            
           

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))