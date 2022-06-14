import unittest

import pandas as pd
from evaluator import *
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from main import *
# class TestW2vBaseline(unittest.TestCase):
#
#     def test_w2v_substitution(self):
#         glove_file = datapath("/home/ajjour/Downloads/glove.840B.300d.txt")
#         tmp_file = get_tmpfile("test_word2vec.txt")
#         _ = glove2word2vec(glove_file, tmp_file)
#         vecs = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
#         tmp_file = get_tmpfile("test_word2vec.txt")
#         substitutes = generate_most_similar(context="women sing and men bring",target_offset=7,target="spain",vecs=vecs)
#         print(substitutes)


class TestEvaluation(unittest.TestCase):
    def create_dataset(self):
        path = "data/swords-v1.1-dev.csv"
        df_dataset = pd.read_csv(path,sep=",",quotechar='"')
        target_id = 't:b2759a395bae8d9d501e991417c77f46cc4a1e9d'
        df_sample = df_dataset[df_dataset['target_id']==target_id]
        self.path_groundtruth_sample= 'data/sample.csv'
        self.path_predictions = 'data/predictions-sample.json'
        df_sample.to_csv('data/sample.csv',sep=",",quotechar='"',encoding="utf-8",index=False)
    def predict_substitutes(self):

        generate_production(True,False,self.path_groundtruth_sample,self.path_predictions)
        predictions = parse_predictions(self.path_predictions)
        groundtruth = parse_groundtruth(self.path_groundtruth_sample,is_acceptable=False)
        evaluation = eval(predictions,groundtruth,True)
        evaluation_strict = eval(predictions,groundtruth,False)
        evaluation.update(evaluation_strict)
        return evaluation
    def test_evaluation(self):
        self.create_dataset()
        evaluation = self.predict_substitutes()
        print(evaluation)
        self.assertTrue(evaluation['strict_f1c']==0.23529411764705882)
        self.assertTrue(evaluation['lenient_f1c']==0.19047619047619047)