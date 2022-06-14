1. To setup the project run ./setup.sh
2. python dataset.py --input data/swords-v1.1_dev.json.gz --output data
3. python main.py --wordnet --input data/swords-v1.1-dev.csv --output data/predictions.json
4. ./evaluate.sh
5. python evaluator.py -p data -t data -o data