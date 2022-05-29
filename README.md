## Repo summary/presentation (UNDER CONSTRUCTION - to be completed by 5 June, 2022)

This repo explores the use of convolutional autoencoders as feature extraction tools compared to hand-crafted features obtained by the pyAudioAnalysis Python library.
For the purposes of the experiments we consider the valence class data of the [MSP Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) dataset.

In short, the repo consists of the following files :
- main_ntbk.ipynb : The main repo (Jupyter) notebook that contains sections describing each experiment step
- autoencoders.py : Contains hand-written symmetric convolutional autoencoder architectures. They take as input fixed size spectograms of (74,200) size and are parametrized with respect to the latent (bottleneck) dimension of the autoencoder.  
- autoencoder_training_and_tuning.ipynb : Colab notebook that contains the training and tuning of the architectures included in the autoencoders.py file.
- "files" folder : Contains training/test data and spectograms pickle files and the final (tuned) encoder model that we use for feature extraction.

==========================================================================================
==========================================================================================

Below one may find a detailed presentation of the main points and results to consider while reading the "main_ntbk.ipynb" file. 

From now one we refer to the hand-crafted features of pyAudioAnalysis as "high-level features" and the features extracted at the central layer of the autoencoder as "code features".

The "main_ntbk.ipynb" notebook is structured as follows :

## 1) MSP Podcast

### 2) Exploring the signal size of the audio files - Elimination of short ones

3) Unlabelled data : Spectograms

   (spectograms_of_unlabelled_data.pickle)

4) Convolutional Autoencoder model

5) Training labelled data : High Level Features, Spectograms & Code Features

   (explain how we save one pickle data for each)
   
   training_data_hlf.pickle // training_data_spectograms.pickle // training_data_cf.pickle

6) The SVM classifier (Tuning C)

7) Test labelled data : High Level Features, Spectograms & Code Features

   (explain how we save one pickle data for each)
   
   test_data_hlf.pickle // test_data_spectograms.pickle // test_data_cf.pickle

8) The experiment
