import argparse

import gzip
import json
import gensim.downloader
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings

from baseline import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet',help="wordnet",action="store_true")
    parser.add_argument('--w2v',help="word2vec",action="store_true")
    args=parser.parse_args()
    return args.wordnet, args.w2v

def generate_production(wordnet,w2v):

    df_swords=pd.read_csv('data/swords-v1.1-dev.csv',sep=",",quotechar='"',encoding="utf-8")
    if w2v:
        glove_vectors = gensim.downloader.load('word2vec-google-news-300')
        #glove_vectors = gensim.downloader.load('glove-twitter-25')
        knearest=train_knearest(glove_vectors,10,0.4)

    result = {'substitutes_lemmatized': True, 'substitutes': {}}

    for target_id, df_substitues in df_swords.groupby('target_id'):
        target= df_substitues['target'].values[0]
        pos = df_substitues['pos'].values[0]

        if w2v:
            result['substitutes'][target_id] = generate_knearest(target,glove_vectors,knearest)
        else:
            result['substitutes'][target_id] = generate_wordnet(target,pos)

    with open('data/predictions.json', 'w') as f:
        f.write(json.dumps(result))

wordnet,w2v= parse_arguments()
generate_production(wordnet,w2v)