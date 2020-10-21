#!/bin/sh
sudo rm -r bert-transformer elmo-allen_nlp
pip3 install virtualenv

virtualenv bert-transformer
source bert-transformer/bin/activate
pip3 install -r bert-transformer-requirements.txt
deactivate

virtualenv elmo-allen_nlp
source elmo-allen_nlp/bin/activate
pip3 install -r allen-nlp-requirements.txt
deactivate