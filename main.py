import gzip
import json
from baseline import generate,generate_w2v
import warnings
from tqdm import tqdm
import gensim.downloader
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
with gzip.open('data/swords-v1.1_dev.json.gz', 'r') as f:
    swords = json.load(f)
glove_vectors = gensim.downloader.load('glove-twitter-25')
glove_file = datapath("/home/ajjour/Downloads/glove.840B.300d.txt")
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
vecs = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

result = {'substitutes_lemmatized': True, 'substitutes': {}}
errors = 0
for tid, target in tqdm(swords['targets'].items()):
    context = swords['contexts'][target['context_id']]
    try:
        result['substitutes'][tid] = generate_w2v(
            context['context'],
            target['target'],
            target['offset'],vecs=vecs,
            target_pos=target.get('pos'))
    except:
        errors += 1
        continue

if errors > 0:
    warnings.warn(f'{errors} targets were not evaluated due to errors')

with open('data/swords-v1.1_dev_mygenerator.lsr.json', 'w') as f:
    f.write(json.dumps(result))