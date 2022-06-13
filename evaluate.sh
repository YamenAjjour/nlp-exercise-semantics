python main.py --wordnet
cd ../swords

python -m swords.cli eval swords-v1.1_dev --result_json_fp ../nlp-exercise-semantics/data/predictions.json
cd -
