
python main.py
cd ../swords
download()
{
virtualenv venv
source venv/bin/active
pip install nltk==3.5
python -m nltk.downloader wordnet
pip install requests==2.25.1
pip install numpy==1.19.5
pip install -e .

}
python -m swords.cli eval swords-v1.1_dev --result_json_fp ../nlp-exercise-semantics/swords-v1.1_dev_mygenerator.lsr.json