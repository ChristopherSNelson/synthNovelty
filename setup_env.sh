#!/bin/bash

pip install numpy==1.26.4 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.38.2
pip install simpletransformers==0.61.13
pip install rxnfp --no-deps #avoid ancient dependencies
pip install rdkit-pypi
pip install numpy pandas tqdm scikit-learn
pip install matplotlib
pip install datasets

