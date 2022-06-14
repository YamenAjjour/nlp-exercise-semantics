[Swords](https://github.com/p-lambda/swords#evaluating-new-lexical-substitution-methods-on-swords) evaluation reimplementation
## Running Baseline
To generate lexical substitutes using a baseline
```
   python main.py --wordnet --input data/dataset/lexical-substitution-data.csv --output data/wordnet/predictions.csv
   python main.py --w2v --input data/dataset/lexical-substitution-data.csv --output data/word2vec/predictions.csv
```
   * *--wordnet*       to use wordnet synonyms as lexical substitutes
   * *--w2v*           to use word2vec-based similarity metrics to find subsitutes
   * *--distillbert*   to use distillbert to generate substitute as a fill-in-the-gap task
   * *--input*         path to the input of a lexical subsitution system 
   * *--ouptut*        path to the output of the lexical substitution system  

## Evaluation
 To evaluate the output of a system. 
   
```
    python evaluator.py --predictions data/wordnet --truth data/dataset/ --output output/
``` 
* *--truth* a folder where the dataset **swords-v1.1-dev.csv** resides
* *--output* a folder to write the evaluation metrics to 
* *--predction* a csv file containing the substitutes for each target id as follows

```
target_id,substitute,score
"t:0bc0ac48c9799fcae60eb1b1f6bc63a29ccc4d53","come","1"
"t:0bc0ac48c9799fcae60eb1b1f6bc63a29ccc4d53","come up","1"
"t:0bc0ac48c9799fcae60eb1b1f6bc63a29ccc4d53","arrive","1"

```
The evaluation script produces lenient f1 score and strict f1 score for **conceivable** substitutes, i.e., score > 0.1

## Data Preprocessing
The evaluation script has been tested on the development dataset of swords. To use the script, the dataset must be 
fist converted to csv format using the following command

```
   python --input data/swords-v1.1_dev.json.gz --output data/dataset 
```


