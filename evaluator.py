from pathlib import Path
from argparse import ArgumentParser
from statistics import mean
import pandas as pd
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(token,pos):
    return lemmatizer.lemmatize(token, pos=pos)

def parse_groundtruth(groundtruth_path,is_acceptable):
    groundtruth = {}
    df_groundtruth = pd.read_csv(groundtruth_path,sep=",",quotechar='"',encoding="utf-8")
    for target_id, df_substitutes in df_groundtruth.groupby('target_id'):
        pos = df_substitutes['pos'].values[0]

        if is_acceptable:
            threshold = 0.5+1e-4
        else:
            threshold = 0.1
        df_substitutes = df_substitutes[df_substitutes['score']>=threshold]
        substitutes = df_substitutes['substitute'].values
        groundtruth[target_id]={'substitutes':substitutes,'pos':pos}
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
        true_substitutes = set(groundtruth[target_id]['substitutes'])
        target_pos = groundtruth[target_id]['pos']
        predicted_substitutes = predictions[target_id]
        sorted_substitutes = sorted(predicted_substitutes,key= lambda x: -x[1])


        substitutes = set([lemmatize(x[0],target_pos) for x in  sorted_substitutes[:10]])
        numerator += len([x for x in substitutes if x in true_substitutes])
        p_denominator += len(substitutes)
        if is_lenient:
            r_denominator += min(len(true_substitutes),10)
        else:
            r_denominator += len(true_substitutes)


    p = numerator/p_denominator
    r = numerator/r_denominator
    f = (2*p*r)/(p+r)

    if is_lenient:
        return {'lenient_f1c':f}
    else:
        return {"strict_f1c":f}


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

    predictions = parse_predictions(predictions_path)
    groundtruth = parse_groundtruth(groundtruth_path,is_acceptable=True)

    return (predictions,
            groundtruth,
            output_path)

def to_tira_measure(measure):
    measure_str = ""
    for metric in measure:
        measure_str = measure_str + f'measure {{\n key: "{metric}"\n value: "{measure[metric]}"\n}}\n'


    return measure_str

if __name__ == "__main__":
    predictions, groundtruth, output_path = parse_input()
    evaluation = eval(predictions,groundtruth,True)
    evaluation_strict = eval(predictions,groundtruth,False)
    evaluation.update(evaluation_strict)
    measure_str = to_tira_measure(evaluation)
    with open(output_path , 'w') as of:
        of.write(measure_str)
