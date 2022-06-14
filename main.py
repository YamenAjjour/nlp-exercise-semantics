import argparse
import csv
import gzip
import json
import gensim.downloader
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from baseline import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet',help="use wordnet synonyms as lexical substitutes",action="store_true")
    parser.add_argument('--w2v',help="use word2vec-based similarity metrics to find subsitutes",action="store_true")
    parser.add_argument('--distillbert',help="use distillbert to find subsitutes",action="store_true")
    parser.add_argument("-i", "--input", help="path to the input of a lexical subsitution system")
    parser.add_argument("-o", "--output", help="path to the output of the lexical substitution system  ")
    args=parser.parse_args()
    return args.wordnet, args.w2v, args.input ,args.output

def generate_substitutes(wordnet, w2v, distillbert, path_input, path_output):
    """
    Generate substitutes using word2vec or wordnet for the targets provided in the inputs. The output is stored in the
    following format

    target_id, substitute


    :param wordnet: boolean for using wordnet synonyms
    :param w2v: boolean for using word2vec semantically similar words
    :param path_input: path to the input dataset which contains the "target","pos","target_id","context"
    :param path_output: a csv file with target ids and the produced substitutes
    :return:
    """
    df_swords=pd.read_csv(path_input,sep=",",quotechar='"',encoding="utf-8")

    if w2v:
        #glove_vectors = gensim.downloader.load('word2vec-google-news-300')
        glove_vectors = gensim.downloader.load('glove-twitter-25')
        knearest=train_knearest(glove_vectors,10,0.4)
    elif distillbert:
        model_checkpoint = "distilbert-base-uncased"
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    target_ids=[]
    all_substitutes=[]
    all_scores=[]
    for target_index, target_record in tqdm(df_swords.iterrows()):

        target= target_record['target']
        pos = target_record['pos']
        target_id = target_record['target_id']
        context = target_record["context"]
        if w2v:
            subsitutes, scores= generate_knearest(target,glove_vectors,knearest)
        elif distillbert:
            subsitutes, scores = generate_distillbert(target,context,model,tokenizer)
        else:
            subsitutes, scores = generate_wordnet(target,pos)
        all_substitutes.extend(subsitutes)
        all_scores.extend(scores)
        target_ids.extend([target_id for _ in subsitutes])

    df=pd.DataFrame({'target_id':target_ids,'substitute':all_substitutes,'score':all_scores})

    df.to_csv(path_output,sep=",",quotechar='"',encoding="utf-8",quoting=csv.QUOTE_ALL,index=False,columns=['target_id','substitute','score'])

if __name__ == "__main__":
    wordnet,w2v, path_input, path_output = parse_arguments()
    generate_substitutes(wordnet, w2v, path_input, path_output)