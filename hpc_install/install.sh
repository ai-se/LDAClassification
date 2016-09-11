#!/usr/bin/env bash

#cd /share2/aagrawa8/

#wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh

#bash Miniconda2-latest-Linux-x86_64.sh # take care of the installation directory

#cd
conda remove mkl mkl-service -y
conda install pip numpy scipy scikit-learn matplotlib -y
conda install nomkl -y
#conda install mkl -y
pip install -U lda
pip install -U nltk

#python
#import nltk
#nltk.download

