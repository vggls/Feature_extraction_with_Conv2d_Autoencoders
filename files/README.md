hlf = high level features = pyAudioAnalysis features

cf = code features = Autoencoder features

The "..hlf.pickle" files are encoder independent.

The "..cf.pickle" files are computed from "encoder3-latent_dim_64-l_rate_0.001-batch_128.h5" encoder, as this is the final encoder that we use to perform the experiment. The "encoder2-latent_dim_64-l_rate_0.005-batch_128.h5" encoder is saved here because is shows good learning ability as well. One might verify this in the "autoencoder_training_and_tuning.ipynb" file.
