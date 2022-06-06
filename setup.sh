cd ..
git clone git@github.com:p-lambda/swords.git
cd -
pip install nltk==3.5
python -m nltk.downloader wordnet
pip install requests==2.25.1
pip install numpy==1.19.5
pip install -e .