import csv
from pathlib import Path
from argparse import ArgumentParser
import gzip
from tqdm import tqdm
import json
import pandas as pd
import re
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="path to the original dataset")
    parser.add_argument("-o", "--output", help="path to the preprocessed dataset")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    return input_path,output_path

def clean_context(context,double_quotes=True):
    context=re.sub('\\t+',' ',context)
    context=re.sub('\\n+',' ',context)
    context=re.sub('\s+',' ',context)
    if double_quotes:
        context=context.replace('"',"'")
    return context

def get_pos(pos):

    return  {
            'VERB': 'v',
            'NOUN': 'n',
            'ADJ': 'a',
            'ADV': 'r'
        }.get(pos, 'n')

def simplify_dataset(input_path,output_path):
    targets = []
    contexts =[]
    o_contexts =[]
    subsitutes = []
    part_of_speeches= []
    score = []
    target_ids =[]

    with gzip.open(input_path, 'r') as f:
        swords = json.load(f)
        for sid,subsitute in tqdm(swords['substitutes'].items()):
            target_id = subsitute['target_id']
            target_ids.append(target_id)
            target = swords['targets'][subsitute['target_id']]
            context = swords['contexts'][target['context_id']]
            labels = swords['substitute_labels'][sid]

            targets.append(target['target'])
            subsitutes.append(subsitute['substitute'])
            contexts.append(clean_context(context['context']))
            o_contexts.append(clean_context(context['context'],False))
            part_of_speeches.append(get_pos(target['pos']))
            score.append(len([label for label in labels if (label=="TRUE" or label =="UNSURE")])/len(labels))

        df_simplified_dataset = pd.DataFrame({"target":targets,"target_id":target_ids,"context":contexts, "substitute":subsitutes, "pos":part_of_speeches, "score":score})
        df_simplified_dataset.sort_values(['target','context'],inplace=True)
        df_simplified_dataset.to_csv(output_path,sep=",",quotechar='"',quoting=csv.QUOTE_ALL,encoding="utf-8",columns=['target','pos','substitute','score','target_id','context'],index=False)
if __name__ == "__main__":
    input_path, output_path = parse_args()
    simplify_dataset(input_path,output_path)



