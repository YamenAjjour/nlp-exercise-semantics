import gensim
import random
import torch
from gensim.test.utils import datapath
from transformers import AutoTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from nltk.corpus import wordnet as wn


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
    :return: a pair of two lists: subsitutes and scores
    """
    print("training")
    all_vect=get_vectors(w2v)

    knearest = NearestNeighbors(n_neighbors=k, radius=radius)
    knearest.fit(all_vect)
    return knearest

def generate_knearest( target, w2v, knearest, ):
    """
    Generate substitutes for a target using a knearest neighbors model
    :param target: target word
    :param w2v: a gensim word2vec model
    :param knearest: a trainied knearest model on the same word2vec model
    :return: a pair of two lists: subsitutes and scores
    """
    if target in w2v.index_to_key:
        target_vector = w2v[target]
        distances, indices = knearest.kneighbors([target_vector])
        max_distance =max(distances[0])
        normalized_distances = [distance/max_distance for distance in distances[0]]
        similarities=[]
        for distance in normalized_distances:
            similarities.append((1 - distance) * 100)
        subsitutes = [w2v.index_to_key[index] for index in indices[0]]
    else:
        return [],[]
    return subsitutes,similarities


def generate_most_similar( target, glove_vectors):
    """Produces _substitutes_ for target span within using standard gensim method

    Args:

      target: The target word, e.g. "straightforward"
      glove_vectors: is a word2vec model loaded by gensim
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
    return substitutes, scores

def get_wordnet_pos(pos_tag):
    """
    Map a part of speech tag to wordnet format
    :param pos_tag:
    :return: wordnet compliant part of speech tag
    """
    if pos_tag.startswith('a'):
        return wn.ADJ
    elif pos_tag.startswith('v'):
        return wn.VERB
    elif pos_tag.startswith('n'):
        return wn.NOUN
    elif pos_tag.startswith('r'):
        return wn.ADV
    else:
        return ''


def generate_wordnet(target,pos_tag):
    """
    Generate synoyms using wordnet by getting all synsets of the target and producing all words of these synsets
    :param word: the target word to generate substitutes for
    :param pos_tag: the target part of speech tag 'a' for adjective, 'v' for verb, 'n' for noun, 'r' for adeverb
    :return: subsitutes and scores as lists
    """
    all_lemmas=[]
    wordnet_pos = get_wordnet_pos(pos_tag)
    for synset in wn.synsets(target,wordnet_pos):
        synonyms = [token for token in synset.lemma_names() if token != target]
        synonyms = [synonym.replace("_"," ") for synonym in synonyms]
        all_lemmas.extend(synonyms)
    return all_lemmas,[1 for _ in all_lemmas]

def generate_distillbert(target, context,model,tokenizer):
    context=context.replace(target,"[MASK]")
    inputs = tokenizer(context, return_tensors="pt")
    token_logits = model(**inputs).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_10_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
    top_10_tokens = [tokenizer.decode([token]) for token in top_10_tokens]
    return top_10_tokens, torch.topk(mask_token_logits, 10, dim=1).values[0].tolist()