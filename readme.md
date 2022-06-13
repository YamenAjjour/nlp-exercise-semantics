1. To setup the project run ./setup.sh
2. python dataset.py --input data/swords-v1.1_dev.json.gz --output data/swords-v1.1-dev.csv
3. ./evaluate.sh
4. python evaluator.py -p data -t data -o dat