import random
import gensim

from gensim.test.utils import datapath
from scipy import spatial
def generate(
        context,
        target,
        target_offset,
        glove_vectors,
        target_pos=None):
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
    results = glove_vectors.most_similar(target)

    scores = [pair[1]*100 for pair in results]
    substitutes = [pair[0] for pair in results]
    return list(zip(substitutes, scores))

# NOTE: 'substitutes_lemmatized' should be True if your method produces lemmas (e.g. "run") or False if your method produces wordforms (e.g. "ran")

def check_and_add(substitutes, word, score):
    i = 0
    while i <len(substitutes) and score >= substitutes[i][1] :

        if i >0:
            substitutes[i-1]=substitutes[i]
        i = i + 1
    if i > 0:
        substitutes[i-1]=(word,score)

def generate_w2v(
    context,
    target,
    target_offset,vecs,
    target_pos=None,k=10):
    substitutes=[("",0) for i in range(k)]
    for word in vecs.vocab:
        score  = (1 - spatial.distance.cosine(vecs[word],vecs[target])) * 100
        check_and_add(substitutes,word,score)
    return substitutes

