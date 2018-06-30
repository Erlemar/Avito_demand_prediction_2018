# Avito demand prediction

[Kaggle](https://www.kaggle.com/c/avito-demand-prediction)
This is a first serious machine learning competition on Kaggle in which I took part.

There were 4 people in my team and we won't be successfull if we tried alone. We got 131th place (top-7%) and bronze medal.

This repository contains some of the scripts and jupyter notebooks which I used while working on this competition.

## Files description

* Text embeddings and image keypoints.ipynb - getting text embeddings and extracting keypoints from images;
* text2nan.ipynb - filling missing values with neural net predictions;
* NN_latest.ipynb - my attemps ar deep learning;
* LGB.ipynb, Avito.ipynb - notebooks with modelling, a bit messy;
* evaluate  .py - modified scripts to extract NIMA features (originally from [this repo](https://github.com/titu1994/neural-image-assessment).);
* dataprocesser.py - my main script for data processing, has a lot of parameters, which can be changed;
* fasttext.py, word2vec.py - scripts to train embeddings from scratch using all available files;
* features.py - processing text and creating meta-features;
* singlemodel.py - an attempt to use script to load data and train models. Didn't really use it;
* nn_functions.py - helper functions for neural nets.

## Solutions by teammates:

* [Nikita](https://github.com/ML-Person/My-solution-to-Avito-Challenge-2018)
* [Diyago](https://github.com/Diyago/Machine-Learning-scripts/tree/master/DL/Kaggle:%20Avito%20Demand%20Prediction%20Challenge%20(bronze%20solution))