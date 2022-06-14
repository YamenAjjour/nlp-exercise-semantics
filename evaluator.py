from pathlib import Path
from argparse import ArgumentParser
from statistics import mean
import pandas as pd
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(token,pos):
    return lemmatizer.lemmatize(token, pos=pos).lower().strip()

def parse_groundtruth(groundtruth_path,is_acceptable):
    groundtruth = {}
    df_groundtruth = pd.read_csv(groundtruth_path,sep=",",quotechar='"',encoding="utf-8")
    for target_id, df_substitutes in df_groundtruth.groupby('target_id'):
        pos = df_substitutes['pos'].values[0]
        target = df_substitutes['target'].values[0]
        target_lemma = lemmatize(target.lower(),pos)
        if is_acceptable:
            threshold = 0.5+1e-4
        else:
            threshold = 0.1
        all_substitutes = df_substitutes['substitute'].values
        df_substitutes = df_substitutes[df_substitutes['score']>=threshold]
        substitutes = df_substitutes['substitute'].values

        substitutes = [lemmatize(substitute_lemma,pos) for substitute_lemma in substitutes]
        groundtruth[target_id]={'substitutes':substitutes,'all_substitutes':all_substitutes,'pos':pos,'target_lemma':target_lemma}

    return groundtruth


def parse_predictions(predictions_path):
    predictions = {}
    with open(predictions_path, 'r') as predictions_file:
        results = json.load(predictions_file)
        for target_id in results['substitutes']:
            substitutes = results['substitutes'][target_id]
            predictions[target_id]=substitutes
    return predictions


def eval(predictions,groundtruth,is_lenient=False):
    p_denominator= 0
    numerator = 0
    r_denominator = 0

    for target_id in groundtruth:
        all_substitutes = set(groundtruth[target_id]['all_substitutes'])
        true_substitutes = set(groundtruth[target_id]['substitutes'])
        target_pos = groundtruth[target_id]['pos']
        target_lemma = groundtruth[target_id]['target_lemma']
        predicted_substitutes = predictions[target_id]
        sorted_substitutes = sorted(predicted_substitutes,key= lambda x: -x[1])
        lemmatized_substitutes = [lemmatize(x[0],target_pos) for x in  sorted_substitutes]
        substitutes = [substitute_lemma for substitute_lemma  in lemmatized_substitutes if substitute_lemma!=target_lemma]

        if is_lenient:
            filtred_substitutes=[substitute_lemma for substitute_lemma  in substitutes if substitute_lemma in all_substitutes]
            substitutes = set(filtred_substitutes[:10])
        else:
            substitutes  = set(substitutes[:10])

        numerator += len([x for x in substitutes if x in true_substitutes])
        p_denominator += len(substitutes)
        r_denominator += min(len(true_substitutes),10)

    p = numerator/p_denominator
    r = numerator/r_denominator
    f = (2*p*r)/(p+r)
    if is_lenient:

        return {'lenient_f1c':f * 100}
    else:
        return {"strict_f1c": f * 100}


def parse_input():
    """
    read the files given as parameters and load the newline-delimited stings
    :return: a tripe with   [0] a list of predicted labels. 1 if Y and 0 if N
                            [1] a list of true labels. 1 if Y and 0 if N
                            [3] the output Path
    """
    parser = ArgumentParser()
    parser.add_argument("-p", "--predictions", help="path to the dir holding the predicted predictions.json")
    parser.add_argument("-t", "--truth", help="path to the dir holding the swords-v1.1-dev.csv")
    parser.add_argument("-o", "--output", help="path to the dir to write the results to")
    args = parser.parse_args()

    groundtruth_path = Path(args.truth) / 'swords-v1.1-dev.csv'
    predictions_path = Path(args.predictions) / 'predictions.json'
    output_path = Path(args.output) / 'evaluation.protext'



    return (predictions_path,
            groundtruth_path,
            output_path)

def to_tira_measure(measure):
    measure_str = ""
    for metric in measure:
        measure_str = measure_str + f'measure {{\n key: "{metric}"\n value: "{measure[metric]}"\n}}\n'


    return measure_str

if __name__ == "__main__":
    predictions_path, groundtruth_path, output_path = parse_input()
    predictions = parse_predictions(predictions_path)
    groundtruth = parse_groundtruth(groundtruth_path,is_acceptable=False)
    evaluation_strict = eval(predictions,groundtruth,False)
    evaluation = eval(predictions,groundtruth,True)
    evaluation.update(evaluation_strict)
    measure_str = to_tira_measure(evaluation)
    with open(output_path , 'w') as of:
        of.write(measure_str)
