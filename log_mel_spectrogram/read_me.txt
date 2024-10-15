This directory contains implementation of MOS prediction model. We modify
the proposed model in 'Intrusive and non-intrusive perceptual speech quality 
assessment using a convolutional neural network' at WASPAA'19, Microsoft by 
replacing manual feature extraction with feature learning.

In our model, we compute magnitude spectogram of clean & noisy speech, then
splice it to form context vectors. We expect our CNN model to learn relevant
features from context vectors itself that are suited to MOS prediction task. 

The dataset used is NISQA corpus. For training, both simulated and live utterances
have been used. For testing, both simulated and live utterances of validation data
have been used due to ref-deg file size mismatch in the testing dataset. 

The model is trained with the original architecture using the 'train_model' script.
Our idea involves exploring the effect of manual feature extraction vs auto feature
extraction on speech assessment.

The model is tested using the 'test_model' script and results tabulated and shown
graphically using scatter plots. 

Comments: 