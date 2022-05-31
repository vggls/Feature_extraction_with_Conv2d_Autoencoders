## Repo Summary / Presentation (UNDER CONSTRUCTION - to be completed by 5 June, 2022)

This repo explores the use of convolutional autoencoders as feature extraction tools when compared to hand-crafted features obtained by the pyAudioAnalysis Python library, in a scenario where the available training dataset consists of "a few" labelled points and "many" unlabelled points (of the same data source).

The unlabelled points are used via their spectograms to train a symmetric convolutional autoencoder which learns feature representations of the data. In each iteration of the experiment we consider potions (percentages) of the labelled training data to train two SVM classifiers ('rbf' kernel and C=10) on their hand-crafted and autoencoder features respectively. Finally, the learned classifiers are tested on hand-crafted and code representations of a test dataset, where we keep the weighted F1 scores.

For the purposes of our experiments we consider data from the valence class of the [MSP Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) dataset. They are divided into "negative", "neutral" and "positive" (sub)classes.

In short, this repo consists of the following :
- main_ntbk.ipynb : The main repo (Jupyter) notebook that contains sections describing each experiment step.
- autoencoders.py : Contains hand-written symmetric convolutional autoencoder architectures. They take as input fixed size spectograms of (74,200) size and are parametrized with respect to the latent (bottleneck) dimension of the autoencoder.  
- autoencoder_training_tuning.ipynb : Colab notebook that contains the training and tuning of the architectures included in the autoencoders.py file.
- "files" folder : Contains training and test data features pickle files and the final encoder model that we use for code feature extraction.

## **********************************************************************************************

Below one may find a detailed presentation of the main points and results to consider while reading the "main_ntbk.ipynb" file. 

From now one we refer to the hand-crafted features of pyAudioAnalysis as "high-level features" and the features extracted at the central layer of the autoencoder as "code features".

The "main_ntbk.ipynb" notebook is structured as follows :

#### 1) MSP Podcast
   In this study we divide the data into 3 speaker independent sets; named Training (labelled) set, Unlabelled set and Test (labelled) set. This section is dedicated into properly constructing them.
   
   - In the first place we build a "training_folder" divided into 3 subfolders of 400 audio points each (one subfolder for each valence class). The purpose of this set is to represent the "small" group of labelled training data.
   
   - Similarly, we build an "unlabelled_folder" divided into 3 subfolders of 3300 points each. These points will be used later without their labels and represent the "large" group of unlabelled training data. Note that only for this section we considered their labels as well in an attempt to have a "balanced" split of the data.
    
   - Finally, we build the "test_folder" divided into 3 subfolders of 500 points each. These points are used for testing via their high-level and code representations.
    
#### 2) Exploring the signal size of the audio files - Elimination of short audio files
   The motivation for this section comes from the observation that the data of all 3 sets will be fed, via their spectograms, into the convolutional autoencoder. More specifically the unlabelled data will train the autoencoder while the training and test data will get their predicted representations from the learned encoder. 
   
   However, due to the different duration of the data we will also get spectograms of different x-axis shape and thus it is not possible to feed them directly to the network, as it requires fixed input sizes. 
   
   So, based on the above, in this section we study the files' duration a bit deeper and attempt to fix a reasonable minimum signal size per file such that files of smaller sizes can be ignored and files of larger signal sizes will be treated via multiple minimum signal size subparts. At the same time we monitor how many points are eliminated by each set and class and by how much class balance is disturbed in each set. 
   
   For these purposes, we construct the signal boxplots of the unlabelled and training sets and conclude that the threshold of 30K as minimum signal size could reasonably satisfy the above constraints. This threshold corresponds to about 3.67secs (sampled at frequency 8000).
   
   <p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/171047834-f0ffdb08-3605-4cbe-bddb-b658f0287c71.png" height="250" width="500" />
     <img src="https://user-images.githubusercontent.com/55101427/171047881-5d08b548-3d7b-4404-b8df-21b02ce7c1b5.png" height="250" width="500" />
   </p>
   
   By eliminating signals of size less than 30K samples, the number of the experiment contributing points is re-arranged as follows :
   - training set: 350 "negative", 362 "neutral" and 319 "positive" points (1031 in total)
   - unlabelled set: 2781 "negative", 2674 "neutral" and 2622 "positive" points (8077 in total)
   - test set: 432 "negative", 421 "neutral" and 383 "positive" points (1236 in total)
     (the test set numbers are computed in detail in section 7)

#### 3) Unlabelled data : Spectograms
   In this section the spectograms of the unlabelled data are constructed, at levels s_win=s_step=0.05 as per pyAudioAnalysis "spectogram" method. As explained above, we focus only in data with signal size of at least 30K samples and cut the signal in 30K size parts. This results in spectograms of fixed shape (74, 200). Note that the tail subparts of each signal, sized below 30K, are also eliminated by the process. 
   
   In order to get a better view of the above consider the following examples:
   When we run the command "fs, signal = aIO.read_audio_file(wav_path)", the following cases might take place :
   - If signal.shape[0]<30000, then the file is eliminated
   - If signal.shape[0]>=30000, consider wlog file with signal.shape[0]=70000. Then the parts [0,30000] and [30000, 60000] are kept and result in two spectograms of shape (74, 200) while the remaining (60001, 70000] sample range is eliminated.
   
   Eventually, we get 11300 spectograms corresponding to the 8077 unlabelled data. They are saved in the "spectograms_of_unlabelled_data.pickle" file.

#### 4) Convolutional Autoencoder model
   At the beginning of this section the reader is referred to two external sources, the "autoencoders.py" file and the Colab "autoencoder_training_and_tuning.ipynb" notebook; whose short summary will be given here.
   
   The first one, includes multiple hand-written symmetric convolutional autoencoder architectures parametrized with respect to their latent dimension size (i.e. the size of the "bottleneck" layer at the center of the symmetric autoencoder). The architectures are build considering different number of convolutional layers, kernels and strides.
   
   On the other hand, the Colab notebook is used for tuning and training the architectures of autoencoders.py with the spectograms of section 3. It consists of the "tune_autoencoder" function which runs 8 (2^3) training experiments per autoencoder architecture for different latent dimensions (bottleneck dimension), Adam learning rates and training batch sizes. Based on further experiments conducted we stick to the "Binary Crossentropy" loss and the "Adam" optimizer, because choices such as "MSE" loss and "SGD" optimizer did not give robust structures under many different tuning combinations. The function eventually plots the loss history of each hyperparameter combination and stores the respective encoder model. We highlight that due to Colab's limited RAM sources it was very time consuming possible to tune for more hyper-values. In the end of the notebook, we choose the best autoencoder of each architecture and compare them based on their loss values and their complexity. The chosen model is saved in the "files" folder and its summary and loss history are as follows :
   
   <p align="center">
     <img src="https://user-images.githubusercontent.com/55101427/171223701-a75e6824-df41-4908-bcaf-3ee06e2e57ba.png" height="220" width="250" />
     <img src="https://user-images.githubusercontent.com/55101427/171220266-a2f8ba4b-fdc7-4265-b114-209961575d38.png" height="330" width="300" />
   </p>
   
   Returning back to "main_ntbk.ipynb", in the short section no. 4, we load the "optimal" encoder (over all tuned above) in order to use it as code feature extractor for the training and test data. 
  
#### 5) Training labelled data : High Level Features, Spectograms & Code Features
   As the title suggests this section is concerned with calculating the training data features at three different levels.
   
   We deal with the signal size exactly as in section 3, and we get the following :
   
   - For each training data point a 136-dim high level features vector via the "mid_feature_extraction" method of pyAudioAnalysis. The calculations are based
   m_win=m_step=1 and s_win=s_step=0.05. 
   
     The points (along with their labels) are saved in the "training_data_hlf.pickle" file. (see "files" folder)

   - For each data point at least one (or more) spectogram(s) of size (74, 200) via the "spectogram" method of pyAudioAnalysis. We note that spectograms that correspond to the same data point have the same label. Eventually, the 1031 data points yield 1491 spectograms.
     
     In the code, we also introduce the "var" variable which keeps track of the data point (one among the 1031) that the spectogram/label pair comes from. This is useful in order to correctly collect the training datapoints in percentages per class in the experiment of section 8. This information is stored in the "track" list and is transferred to the code features below as well.
     
     The spectograms (along with their labels and "track" list) are saved in the "training_data_spectograms.pickle" file.
          
   - For each spectogram we get its code features via the learned encoder of section 4 (use of encoder.predict() method). We have 1491 64-dim code feature vectors with the same labels as the spectograms.

     The points (along with their labels and "track" list) are saved in the "training_data_cf.pickle" file. (see "files" folder)

#### 6) The SVM classifier (Tuning C)
   In this section we decide on the type of classifier that we will use for the experiment. We use the entire training dataset, through its high level feature representations, to perform cross-validation and tune the C parameter. Eventually, we get C=10. 

#### 7) Test labelled data : High Level Features, Spectograms & Code Features
   This section is quite similar to section no.5 with the training data; with the difference that there is no need to introduce the tracking "var" variable as the test set is used in its wholeness (and not in percentages) in the experiment. Eventually, we get:
   
   - For each test data point a 136-dim high level features vector via the "mid_feature_extraction" method of pyAudioAnalysis. The calculations are based
   m_win=m_step=1 and s_win=s_step=0.05.
   
       The points (along with their labels) are saved in the "test_data_hlf.pickle" file. (see "files" folder)
   
   - For each test point at least one (or more) spectogram(s) of size (74, 200) via the "spectogram" method of pyAudioAnalysis.  Eventually, the 1236 test points yield 1696 spectograms.
   
       The spectograms (along with their labels) are saved in the "test_data_spectograms.pickle" file.
   
   - For each spectogram we get its code features via the learned encoder of section 4 (use of encoder.predict() method). We have 1696 64-dim code feature vectors with the same labels as the spectograms.

        The points (along with their labels) are saved in the "test_data_cf.pickle" file. (see "files" folder)
   
#### 8) The experiment
   In this final section, via the "experiment" function we can perform at a fixed "training percentage" level, the following comparison :
   - Fix training percentage level and get "part" of the total training data. We get the same "percentage" of data per class.
   - For this part of training data, train a section 6 classifier and test it on the whole test data (for their high level representations). From this get weighted F1 score. (This step needs the training_data_hlf.pickle and test_data_hlf.pickle files)
   - Apply the above step but with the code feature representations instead. Again get a weighted F1 score. (This step needs the training_data_cf.pickle and test_data_cf.pickle files)

   By applying the above procedure for multiple percentage levels (5%, 10%,.. 95% and 100%), we eventually reach the following graph which describes the respective weighted F1 scores for pyAudioAnalysis and code features as the percentage of training data gradually increases by 5%.

(insert graph here)

#### Variations to check in future experiments
- change min signal size
- handle differently signals of smaller size (for instance introduce padding)
- tune for more hyper values
- what if tune the SVM with the training code features instead ?
