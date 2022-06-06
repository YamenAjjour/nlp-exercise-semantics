import argparse

import gzip
import json
import gensim.downloader
import numpy as np
from tqdm import tqdm

import warnings

from baseline import generate_most_similar,generate_knearest,train_knearest

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius',help="radius to keighbour",type=float)
    args=parser.parse_args()
    return args.radius

def generate_production(radius):
    print(f"trying radius {radius}")
    with gzip.open('data/swords-v1.1_dev.json.gz', 'r') as f:
        swords = json.load(f)
    glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    #glove_vectors = gensim.downloader.load('glove-twitter-25')
    knearest=train_knearest(glove_vectors,10,radius)

    result = {'substitutes_lemmatized': True, 'substitutes': {}}
    errors = 0

    for tid, target in tqdm(swords['targets'].items()):
        context = swords['contexts'][target['context_id']]
        result['substitutes'][tid] = generate_knearest(
        context['context'],
        target['target'],
        target['offset'],w2v=glove_vectors,knearest=knearest,target_pos=target.get('pos'))

    if errors > 0:
        warnings.warn(f'{errors} targets were not evaluated due to errors')

    with open('data/swords-v1.1_dev_mygenerator.lsr.json', 'w') as f:
        f.write(json.dumps(result))

radius= parse_arguments()
generate_production(radius)