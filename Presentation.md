## Repo Report

Below one may find a detailed report of the main points and results to consider while reading the "main_ntbk.ipynb" file. 

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
     (the test set numbers are computed in detail in section 6)

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
     <img src="https://user-images.githubusercontent.com/55101427/171587494-bc27adfc-611d-46fe-a218-ccfd936dc095.png" height="220" width="250" />
     <img src="https://user-images.githubusercontent.com/55101427/171587732-a31e0a52-748d-40ec-88ac-b0cdeca4872e.png" height="330" width="300" />
   </p>
   
   Returning back to "main_ntbk.ipynb", in the short section no. 4, we load the "optimal" encoder (over all tuned above) in order to use it as code feature extractor for the training and test data. 
  
#### 5) Training labelled data : High Level Features, Spectograms & Code Features
   As the title suggests this section is concerned with calculating the training data features at three different levels.
   
   We deal with the signal size exactly as in section 3, and we get the following :
   
   - For each training data point a 136-dim high level features vector via the "mid_feature_extraction" method of pyAudioAnalysis. The calculations are based
   m_win=m_step=1 and s_win=s_step=0.05. 
   
     The points (along with their labels) are saved in the "training_data_hlf.pickle" file. (see "files" folder)

   - For each data point at least one (or more) spectogram(s) of size (74, 200) via the "spectogram" method of pyAudioAnalysis. We note that spectograms that correspond to the same data point have the same label. Eventually, the 1031 data points yield 1491 spectograms.
     
     In the code, we also introduce the "var" variable which keeps track of the data point (one among the 1031) that the spectogram/label pair comes from. This is useful in order to correctly collect the training datapoints in percentages per class in the experiment of section 7. This information is stored in the "track" list and is transferred to the code features below as well.
     
     The spectograms (along with their labels and "track" list) are saved in the "training_data_spectograms.pickle" file.
          
   - For each spectogram we get its code features via the learned encoder of section 4 (use of encoder.predict() method). We have 1491 64-dim code feature vectors with the same labels as the spectograms.

     The points (along with their labels and "track" list) are saved in the "training_data_cf.pickle" file. (see "files" folder)

#### 6) Test labelled data : High Level Features, Spectograms & Code Features
   This section is quite similar to section no.5 with the training data; with the difference that there is no need to introduce the tracking "var" variable as the test set is used in its wholeness (and not in percentages) in the experiment. Eventually, we get:
   
   - For each test data point a 136-dim high level features vector via the "mid_feature_extraction" method of pyAudioAnalysis. The calculations are based
   m_win=m_step=1 and s_win=s_step=0.05.
   
       The points (along with their labels) are saved in the "test_data_hlf.pickle" file. (see "files" folder)
   
   - For each test point at least one (or more) spectogram(s) of size (74, 200) via the "spectogram" method of pyAudioAnalysis.  Eventually, the 1236 test points yield 1696 spectograms.
   
       The spectograms (along with their labels) are saved in the "test_data_spectograms.pickle" file.
   
   - For each spectogram we get its code features via the learned encoder of section 4 (use of encoder.predict() method). We have 1696 64-dim code feature vectors with the same labels as the spectograms.

        The points (along with their labels) are saved in the "test_data_cf.pickle" file. (see "files" folder)
   
#### 7) The experiment
   In this final section, we use the "experiment" function, for SVM, kNN, DecisionTree, RandomForest and AdaBoost classifiers, to perform multiple iterations of the following procedure:
   
   Per iteration :
   
   - We fix a training percentage level and get "part" of the total training data. We get the same "percentage" of data per class.
   - For this part of training data, we consider their high level representations and train a classifier. Then we test it on the entire test dataset considering the high level representations of the data. From this get a weighted F1 score. (This step needs the training_data_hlf.pickle and test_data_hlf.pickle files)
   - Apply the above step but with the code feature representations instead. Again get a weighted F1 score. (This step needs the training_data_cf.pickle and test_data_cf.pickle files)

   By applying the above iteration procedure for multiple percentage levels, 2%, 4%, 6%.., 98% and 100% (50 iterations in total), we eventually reach the following graphs (per classifier) which describe the respective weighted F1 scores for pyAudioAnalysis and code features, as the percentage of training data gradually increases by 2%.
      
<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/172442132-d96c3f75-ccec-4cde-b0a5-b2a031063c42.png" height="220" width="310" />
     <img src="https://user-images.githubusercontent.com/55101427/172442246-36200781-8710-4fef-9de7-6e4e755a3207.png" height="220" width="310" />
     <img src="https://user-images.githubusercontent.com/55101427/172442351-a9beea69-76da-4331-b118-beceb6ed1221.png" height="220" width="370" />
   </p>
   
<p float="left">
     <img src="https://user-images.githubusercontent.com/55101427/172442458-b9883f36-6782-4e46-99bd-8f41bc9a125d.png" height="220" width="330" />
     <img src="https://user-images.githubusercontent.com/55101427/172442572-817141ca-9bad-4a1e-acdd-b52a7f3a9c91.png" height="220" width="330" />
     <img src="https://user-images.githubusercontent.com/55101427/172442681-e95ff73d-033c-4aed-b9d7-aa3caceb83c1.png" height="220" width="330" />
   </p>

At this final point it is essential to highlight the graph of the Random Forest classifier. We observe that as long as we consider training batches of up to 56% of the original dataset the autoencoder features outperform the hand-crafted ones.

## **********************************************************************************************

#### Variations to check in future experiments
- Change min signal size (above was 30K samples)
- Handle differently signals of smaller size (for instance introduce 0-padding)
- Tune autoencoder for more hyper values
- Consider more hyper-parameter values for the classifiers (apart from the default ones used here)
