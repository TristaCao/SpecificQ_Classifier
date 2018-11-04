import sys
import io
import re
import math
from csv import reader, writer
from random import shuffle
import nltk
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import wordnet as wn
import string
import collections
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NUM_TRAIN = 1500 

def sent_len(q_tokens):
    return len(q_tokens)

# return num_of_nouns, num_of_plural_nouns, num_of_numbers, 
# num_of_propernames
def ne_cd(q_tokens):
    tags = nltk.pos_tag(q_tokens)
    counts = Counter(tag for word,tag in tags)
    nouns = counts['NN'] + counts['NNS'] + counts['NNP'] + counts['NNPS']
    pnouns = counts['NNS'] + counts['NNPS']
    nnums = counts['CD']
    npnames = counts['NNP'] + counts['NNPS']
    adj = counts['JJ'] + counts['JJR'] + counts['JJS']
    return nouns, pnouns, nnums, npnames, adj

# given a single training question 
# return a vector with the same size as word_vector
# each entry represents the number of appearance of word type in this single q
def word_count(q_tokens, word_type):
    feature_vector = np.zeros(len(word_type))
    for token in q_tokens:
        if token in word_type:
            index = word_type.index(token)
            feature_vector[index] += 1
    return feature_vector

# return the average, max, min of idfs 
def idf(q_tokens, idf_dic):
    average = 0.0
    maximum = 0
    minimum = 100
    for token in q_tokens:
        idf = math.log1p((NUM_TRAIN+1)/1-1)+1
        if token in idf_dic:
            idf = idf_dic[token]
        average += idf
        maximum = max(maximum, idf)
        minimum = min(minimum, idf)
    if len(q_tokens) == 0:
        return 0,0,0
        
    return average/len(q_tokens), maximum, minimum


    
# read in LIWC lexicon -- from Hal
def read_liwc():
    liwc_dict = {}
    liwc_prefix = {}
    liwc_cat = {}
    for l in io.open('LIWC2007.dic', 'rb').readlines():
        l = l.decode('windows-1252').strip()
        if l == '' or l.startswith('%'): continue
        m = re.fullmatch('([0-9]+)\t(.+)', l)
        if m is not None:
            liwc_cat[int(m.group(1))] = m.group(2)
        else:
            a = l.split()
            assert len(a) >= 2, l
            term = a[0]
            if term == 'kind': continue # skip the kind <of> instance...
            cats = [liwc_cat[int(s)] for s in a[1:]]
            if len(cats) == 0:
                continue
            if not term.endswith('*'): # just a term
                liwc_dict[term] = cats
            else: # prefix
                liwc_dict[term[:-1]] = cats # shortcut for exact match
                if term[0] not in liwc_prefix:
                    liwc_prefix[term[0]] = []
                liwc_prefix[term[0]].append((term[:-1], cats))
    return liwc_dict, liwc_prefix



def match_liwc(w, liwc_dict, liwc_prefix):
    w = w.lower()
    if w in liwc_dict:
        return liwc_dict[w]
    if w[0] in liwc_prefix:
        for prefix, cats in liwc_prefix[w[0]]:
            if w.startswith(prefix):
                return cats
    return set()  
    
def polarity(q_tokens, liwc_dict, liwc_prefix):
    # doing for only insight, Discrepancy, Certainty	
    insight_ct = 0
    discrep_ct = 0
    certain_ct = 0
    cause_ct = 0
    tentat_ct = 0
    differ_ct = 0
    for token in q_tokens:
        cat = match_liwc(token, liwc_dict, liwc_prefix)
        if cat != set():
            if 'insight' in cat:
                insight_ct += 1
            if 'discrep' in cat:
                discrep_ct += 1
            if 'certain' in cat:
                certain_ct += 1
            if 'cause' in cat:
                cause_ct += 1
            if 'tentat' in cat:
                tentat_ct += 1
            if 'differ' in cat:
                differ_ct += 1
    return insight_ct, discrep_ct, certain_ct, cause_ct, tentat_ct, differ_ct
    
def token_hypernym(token):
    syn = wn.synsets(token)
    hypernyms = []
    count = 0
    while syn != []:
        syn = wn.synset(syn[0].name()).hypernyms()
        count += 1
        if syn != []:
            hypernyms.append(syn[0])
    return (count), hypernyms

def num_hypernyms(q_tokens):
    tags = nltk.pos_tag(q_tokens)
    average = 0
    count = 0
    minimum = 50
    maximum = 0
    for token,pos in tags:
        if pos[0] == 'N' or pos[0] == 'V':
            num_hyper, _ = token_hypernym(token)
            count += 1
            average += num_hyper
            minimum = min(minimum, num_hyper)
            maximum = max(maximum, num_hyper)
    if count == 0:
        return 0,0,0
    return average/count, maximum, minimum

# return how many words in the 
def hypernym_match(q_tokens, c_tokens):
    count = 0
    # construct a list of nouns and verbs in the context
    cont_words = []
    tags = nltk.pos_tag(c_tokens)
    for token, pos in tags:
        if pos[0] == 'N' or pos[0] == 'V':
            if token not in cont_words:
                cont_words.append(token)
    # construct a list of nouns and verbs in the question
    q_words_hyps = []         
    tags = nltk.pos_tag(q_tokens)
    for token, pos in tags:
        if pos[0] == 'N' or pos[0] == 'V':
            if token not in q_words_hyps:
                num_hyp, hypernyms  = token_hypernym(token)
                q_words_hyps.append(hypernyms)
    
    # for each word type in question, get its list of hypernyms 
    # (only consider the most popular hypernym)
    for hyps in q_words_hyps:
        break_loop = False
        # check with each word in the context
        # see if the context word is a hypernym of the question word
        # check every meaning of the context word
        for c_word in cont_words:
            c_word = c_word.lower()
            for syn in wn.synsets(c_word):
                if syn in hyps:
                    count += 1
                    break_loop = True
                    break
            if break_loop:
                break
    return count

# return the cosine similarity between the average BOE's for context and q
def similarity(q_tokens, c_tokens, boe):
    q_embedding = np.zeros(200)
    c_embedding = np.zeros(200)
    q_count = 0
    c_count = 0
    for qt in q_tokens:
        qt = qt.lower()
        if qt not in boe:
            continue
        q_count += 1
        q_embedding += boe[qt]
    
    for ct in c_tokens:
        
        ct = ct.lower()
        if ct not in boe:
            continue
        c_count += 1
        c_embedding += boe[ct]
        
    if q_count == 0 or c_count == 0:
        return 0, q_embedding, c_embedding
    
    q_embedding = q_embedding/q_count
    c_embedding = c_embedding/c_count
    
#    print(cosine_similarity(q_embedding, c_embedding))
    return cosine_similarity(q_embedding.reshape(1,-1),c_embedding.reshape(1,-1))[0][0], q_embedding, c_embedding
 
#    return cosine_similarity([q_embedding],[c_embedding])[0][0], q_embedding, c_embedding
    

def main(args):
    
    # readin questions
    count = 0
    word_type = [] # a list of word types appeared in all training questions
    qs = []  # a list of questions  
    with open("processed_data.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            if count >= 400 and count <500:
                count += 1
                continue
            if count >500:
                break
            count += 1
            question = row[1]
            no_punc = question.translate(str.maketrans('','',string.punctuation))
            qs.append(no_punc)
            tokens = word_tokenize(no_punc)
#           # get rid off the stop words
#           filtered = [w for w in tokens if not w in stopwords.words('english')]
            for token in tokens:
                if token not in word_type:
                    word_type.append(token)
    
    # import trained word embeddings
    embedding_vectors = open("vectors.txt")
    boe = {}
    for v in embedding_vectors:
        l = v.split()
        lst = l[1:]
        boe[l[0]] = [float(i) for i in lst]

        
    
    # LIWC
    liwc_dict, liwc_prefix = read_liwc()
    
    # pretrain idf based on all training questions (corpus)
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
    X = vectorizer.fit_transform(qs)
    idf_vec = vectorizer.idf_
    idf_dic = dict(zip(vectorizer.get_feature_names(), idf_vec))
    
    
    label_features = []
    with open("processed_data.csv") as file:
        f = reader(file, delimiter = ',')
        next(f)
        for row in f:
            feature = []
            q = row[1]
            q_no_punc = q.translate(str.maketrans('','',string.punctuation))
            q_tokens = word_tokenize(q_no_punc)
            c = row[0]
            c_no_punc = c.translate(str.maketrans('','',string.punctuation))
            c_tokens = word_tokenize(c_no_punc)
            f0 = sent_len(q_tokens)
            f1, f2, f3, f4, f5 = ne_cd(q_tokens)
            f6, f7, f8 = idf(q_tokens,idf_dic)
            f9, f10, f11, f12, f13, f14 = polarity(q_tokens, liwc_dict, liwc_prefix)
            f15, f16, f17 = num_hypernyms(q_tokens)
            f18 = hypernym_match(q_tokens,c_tokens)
            f19, f20, f21 = similarity(q_tokens,c_tokens, boe)
            f22 = word_count(q_tokens, word_type).tolist()
            

            
            feature.append(row[3])
            feature.append(f0)
            feature.append(f1)
            feature.append(f2)
            feature.append(f3)
            feature.append(f4)
            feature.append(f5)
            feature.append(f6)
            feature.append(f7)
            feature.append(f8)
            feature.append(f9)
            feature.append(f10)
            feature.append(f11)
            feature.append(f12)
            feature.append(f13)
            feature.append(f14)
            feature.append(f15)
            feature.append(f16)
            feature.append(f17)
            feature.append(f18)
            feature.append(f19)
            feature += f20.tolist() # embedding for q
            feature += f21.tolist() # embedding for c
            feature += f22 #bag of words
            label_features.append(feature)
            
    with open("word_feature_weights.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(word_type)
      
    with open("labels_features.csv", mode = 'w') as file:
        w = writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for r in label_features:
            w.writerow(r)
            

            

       

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
