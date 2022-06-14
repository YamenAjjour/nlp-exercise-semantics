#python main.py --wordnet --input data/swords-v1.1-dev.csv --output data/predictions.json
cd ../swords

python -m swords.cli eval swords-v1.1_dev --result_json_fp ../nlp-exercise-semantics/data/predictions.json --output_metrics_json_fp notebooks/mygenerator.metrics.json
cd -
