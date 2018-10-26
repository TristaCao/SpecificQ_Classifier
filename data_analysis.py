import sys
import csv
from csv import reader, writer
import torch
import nltk
from nltk.corpus import stopwords
import random


def cal_score(q1,q2, qs_cs):
    if (q1,q2) in qs_cs:
        choice, score = qs_cs[(q1,q2)]
        if choice == "Question A is more specific":
            return score
        elif choice == "Question B is more specific":
            return -score
        elif choice == "Both are at the same level of specificity":
            return 0
    if (q2,q1) in qs_cs:
        choice, score = qs_cs[(q2,q1)]
        if choice == "Question A is more specific":
            return -score
        elif choice == "Question B is more specific":
            return score
        elif choice == "Both are at the same level of specificity":
            return 0

def main(args):
    raw = {}
    context_q = {}
    q_context = {}
    q_score_count = {}
    q_score = {}
    q_pair_score = {}
    with open("a1312216.csv") as file:
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
                
            if context not in context_q:
                context_q[context] = []
                
            if question_a not in context_q[context]:
                context_q[context].append(question_a)
            if question_b not in context_q[context]:
                context_q[context].append(question_b)
                
            if context not in raw:
                raw[context] = {}
            raw[context][(question_a, question_b)] = (choice, score)
                
            # calculate cumulated score
            score_a = 0
            socre_b = 0
            if choice == "Question A is more specific":
                score_a = score
                score_b = -score
            elif choice == "Question B is more specific":
                score_b = score
                score_a = - score
            elif choice == "Both are at the same level of specificity":
                score_a = 0
                score_b = 0

            
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
                
            for q in q_score_count:
                scr, cot = q_score_count[q]
                q_score[q] = scr/cot
                
            # calculate pair score
            for con in context_q:
                qs = context_q[con]
                scr = 0
                for index in range(len(qs)):
                    if index == len(qs)-1:
                        scr = cal_score(qs[index], qs[0], raw[con])
                    else:
                        scr = cal_score(qs[index], qs[index+1], raw[con])
                    scr2 = cal_score(qs[index], qs[index-1], raw[con])
                    if scr == None or scr2 == None:
                        continue
                    scr += scr2
                    q_pair_score[qs[index]] = scr*0.5
    
    
    
    with open("q_score.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["context", "question", "score", "pair_score", "difference"])
        for q in q_score:
            if q not in q_pair_score:
                continue
            row = [q_context[q], q, q_score[q], q_pair_score[q], \
                   abs(q_score[q]-q_pair_score[q])]
            w.writerow(row)
    
   
    
            
           

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
