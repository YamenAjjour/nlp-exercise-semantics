import random
import gensim

from gensim.test.utils import datapath
from sklearn.neighbors import NearestNeighbors
from scipy import spatial

def generate_most_similar(context, target, target_offset, glove_vectors,target_pos=None):
    """Produces _substitutes_ for _target_ span within _context_

    Args:
      context: A text context, e.g. "My favorite thing about her is her straightforward honesty.".
      target: The target word, e.g. "straightforward"
      target_offset: The character offset of the target word in context, e.g. 35
      target_pos: The UD part-of-speech (https://universaldependencies.org/u/pos/) for the target, e.g. "ADJ"

    Returns:
      A list of substitutes and scores e.g. [(sincere, 80.), (genuine, 80.), (frank, 70), ...]
    """
    # TODO: Your method here; placeholder outputs 10 common verbs
    if target in glove_vectors.index_to_key:
        results = glove_vectors.most_similar(target)
    else:
        results=[]
    scores = [pair[1]*100 for pair in results]
    substitutes = [pair[0] for pair in results]
    return list(zip(substitutes, scores))

# NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "run") or False if your method produces wordforms (e.g. "ran")

def get_vectors(w2v):
    all_vec=[]
    for key in w2v.key_to_index:
        all_vec.append(w2v[key])
    return all_vec

def train_knearest(w2v,k,radius):
    """
    Train a knearest model starting from a gensim word2vec model and k
    :param w2v: word2vec model
    :param k: count of neighbours to find
    :return:
    """
    print("training")
    all_vect=get_vectors(w2v)

    knearest = NearestNeighbors(n_neighbors=k, radius=radius)
    knearest.fit(all_vect)
    return knearest

def generate_knearest(context, target, target_offset, w2v, knearest, target_pos=None):
    if target in w2v.index_to_key:
        target_vector = w2v[target]
        distances, indices = knearest.kneighbors([target_vector])
        max_distance =max(distances[0])
        normalized_distances = [distance/max_distance for distance in distances[0]]
        similarities=[]
        for distance in normalized_distances:
            similarities.append((1 - distance) * 100)
        tokens = [w2v.index_to_key[index] for index in indices[0]]
    else:
        return list(zip([],[]))
    return list(zip(tokens,similarities))

