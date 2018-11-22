import sys
import csv
from csv import reader, writer
import torch
import nltk
from nltk.corpus import stopwords
import random

def extract_score(trust, choice, context, question_a, question_b, context_qs, q_score_count):
    if context not in context_qs:
        context_qs[context] = [question_a, question_b]
    else:
        if question_a not in context_qs[context]:
            context_qs[context].append(question_a)
        if question_b not in context_qs[context]:
            context_qs[context].append(question_b)

    # calculate cumulated score
    score_a = 0
    score_b = 0
    if choice == "Question A is more specific":
        score_a = trust
    elif choice == "Question B is more specific":
        score_b = trust
    elif choice == "Both are at the same level of specificity":
        score_a = trust*0.5
        score_b = trust*0.5


    # assign question->score to q_score
    if question_a in q_score_count:
        count = q_score_count[question_a][1]
        q_score_count[question_a] = (q_score_count[question_a][0]+score_a, count+1)
    else:
        q_score_count[question_a] = (score_a,1)

    if question_b in q_score_count:
        count = q_score_count[question_b][1]
        q_score_count[question_b] = (q_score_count[question_b][0]+score_b, count+1)
    else:
        q_score_count[question_b] = (score_b,1)


def main(args):
    context_qs = {}
    q_score_count = {}
    q_score = {}
    with open("f1307714.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            golden = row[2]
            tainted = row[6]
            trust = float(row[8])
            choice = row[14]
            context = row[16]
            question_a = row[18]
            question_b = row[19]

            if golden=="ture" or tainted=="true":
                continue

            extract_score(trust, choice, context, question_a, question_b, \
                          context_qs, q_score_count)

    with open("f1316545.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            golden = row[2]
            tainted = row[6]
            trust = float(row[8])
            choice = row[14]
            context = row[16]
            question_a = row[18]
            question_b = row[19]

            if golden=="ture" or tainted=="true":
                continue

            extract_score(trust, choice, context, question_a, question_b, \
                          context_qs, q_score_count)

    with open("f1316768.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            golden = row[2]
            tainted = row[6]
            trust = float(row[8])
            choice = row[14]
            context = row[16]
            question_a = row[19]
            question_b = row[20]

            if golden=="ture" or tainted=="true":
                continue

            extract_score(trust, choice, context, question_a, question_b, \
                          context_qs, q_score_count)

    with open("f1312216.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            golden = row[2]
            tainted = row[6]
            trust = float(row[8])
            choice = row[14]
            context = row[16]
            question_a = row[19]
            question_b = row[20]

            if golden=="ture" or tainted=="true":
                continue

            extract_score(trust, choice, context, question_a, question_b, \
                          context_qs, q_score_count)
    with open("f1317791.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            golden = row[2]
            tainted = row[6]
            trust = float(row[8])
            choice = row[14]
            context = row[16]
            question_a = row[19]
            question_b = row[20]

            if golden=="ture" or tainted=="true":
                continue

            extract_score(trust, choice, context, question_a, question_b, \
                          context_qs, q_score_count)


    for q in q_score_count:
        scr, cot = q_score_count[q]
        q_score[q] = scr/cot



    with open("processed_data.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["context", "question", "score", "specific_or_not"])
        for c in context_qs:
            qs = context_qs[c]
            for q in qs:
                s = 0
                if q_score[q] >= 0.5:
                    s = 1
                row = [c, q, q_score[q], s]
                w.writerow(row)

    with open("processed_data_lstm.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for c in context_qs:
            qs = context_qs[c]
            for q in qs:
                s = 0
                if q_score[q] >= 0.5:
                    s = 1
                row = [q, s]
                w.writerow(row)




#    context_qs = {}
#    q_score_count = {}
#    q_score = {}
#
#    with open("f1312216.csv") as file:
#        f = reader(file, delimiter = ',')
#        next(f)
#        for row in f:
#            golden = row[2]
#            tainted = row[6]
#            trust = float(row[8])
#            choice = row[14]
#            context = row[16]
#            question_a = row[18]
#            question_b = row[19]
#
#            if golden=="ture" or tainted=="true":
#                continue
#
#            extract_score(trust, choice, context, question_a, question_b, \
#                          context_qs, q_score_count)
#
#    for q in q_score_count:
#        scr, cot = q_score_count[q]
#        q_score[q] = scr/cot
#
#
#
#    with open("processed_data_test.csv", mode = 'w') as file:
#        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        w.writerow(["context", "question", "score", "specific_or_not"])
#        for c in context_qs:
#            qs = context_qs[c]
#            for q in qs:
#                s = 0
#                if q_score[q] >= 0.5:
#                    s = 1
#                row = [c, q, q_score[q], s]
#                w.writerow(row)
#
#




if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
