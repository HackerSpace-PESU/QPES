#!/bin/sh
sudo rm -r env
pip3 install virtualenv

sudo mkdir env
cd env

virtualenv bert-transformer
source bert-transformer/bin/activate
pip3 install -r ../bert-transformer-requirements.txt
deactivate

virtualenv elmo-allen_nlp
source elmo-allen_nlp/bin/activate
pip3 install -r ../allen-nlp-requirements.txt
deactivate