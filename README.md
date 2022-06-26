# Repo Summary

This repo explores the use of convolutional autoencoders as feature extraction tools when compared to hand-crafted features obtained by the pyAudioAnalysis Python library, in a scenario where the available training dataset consists of "a few" labelled points and "many" unlabelled points (of the same data source).

The unlabelled points are used via their spectograms (min_signal = 30K, spectogram size = (74,200)) to train a symmetric convolutional autoencoder which learns feature representations of the data. In each iteration of the experiment we consider potions (percentages) of the labelled training data to train two classifiers (we consider many different types) on their hand-crafted and autoencoder features respectively. Finally, the learned classifiers are tested on hand-crafted and code representations of a test dataset, where we keep the weighted F1 scores.

For the purposes of our experiments we consider data from the valence class of the [MSP Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) dataset. They are divided into "negative", "neutral" and "positive" (sub)classes.

In short, this repo consists of the following :
- main_ntbk.ipynb : This is the main notebook that one should read to go through the experiment steps and results.

Important note: In order to run the full experiment again, the user needs the pickle files of the "files" repo folder and to re-run the code of section 7 of the "main_ntbk.ipynb" notebook.
- autoencoders.py : Contains hand-written symmetric convolutional autoencoder architectures. They take as input fixed size spectograms of (74,200) size and are parametrized with respect to the latent (bottleneck) dimension of the autoencoder.  
- autoencoder_training_tuning.ipynb : Colab notebook that contains the training and tuning of the architectures included in the autoencoders.py file.
- Report.md : This file is a detailed report of the "main_ntbk.ipynb"
- Presentation.pdf : A short presentation of the repo
- "files" folder : Contains training and test data features pickle files and the final encoder model that we use for code feature extraction.
- another_approach.md : This file contains only the results obtained when performing above experiment with min_signal = 20K (equivalent to spectogram size = (49,200)).
Note that ALL files included in this repo refer to min_signal = 30K. The 20K expreriment graphs are presented for informational purposes.
