for i in $(LC_ALL=C seq 0.0001 0.0001 1);do
  echo "$i"
  python main.py --radius "$i"
  cd ../swords

  python -m swords.cli eval swords-v1.1_dev --result_json_fp ../nlp-exercise-semantics/data/swords-v1.1_dev_mygenerator.lsr.json
  cd -
done