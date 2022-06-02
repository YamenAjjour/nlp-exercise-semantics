import unittest
from baseline import *
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
class TestW2vBaseline(unittest.TestCase):
    def test_check_and_add(self):
        fruits={"apple":0,"orange":10,"banana":20,"kiwi":30,"ananas":40,"grapfruit":50,"apricots":60,
         "lemons":70,"avocdo":80,"carrots":90,"tomatos":100
         }
        substitutes=[("",0) for i in range(10)]
        print(substitutes)
        for key in fruits:
            check_and_add(substitutes,key,fruits[key])
        print(substitutes)
        check_and_add(substitutes,"mandarins",40)
        print(substitutes)
        self.assertEqual(substitutes[5][0],"apricots")

    def test_w2v_substitution(self):
        glove_file = datapath("/home/ajjour/Downloads/glove.840B.300d.txt")
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        vecs = KeyedVectors.load_word2vec_format(tmp_file, binary=False)
        tmp_file = get_tmpfile("test_word2vec.txt")
        substitutes = generate_w2v(context="women sing and men bring",target_offset=7,target="spain",vecs=vecs)
        print(substitutes)