import gzip
import json
from baseline import generate
import warnings
from tqdm import tqdm
with gzip.open('swords-v1.1_dev.json.gz', 'r') as f:
    swords = json.load(f)

result = {'substitutes_lemmatized': True, 'substitutes': {}}
errors = 0
for tid, target in tqdm(swords['targets'].items()):
    context = swords['contexts'][target['context_id']]
    try:
        result['substitutes'][tid] = generate(
            context['context'],
            target['target'],
            target['offset'],
            target_pos=target.get('pos'))
    except:
        errors += 1
        continue

if errors > 0:
    warnings.warn(f'{errors} targets were not evaluated due to errors')

with open('swords-v1.1_dev_mygenerator.lsr.json', 'w') as f:
    f.write(json.dumps(result))