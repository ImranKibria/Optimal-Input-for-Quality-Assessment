# Optimal-Input-for-Quality-Assessment
Exploring a variety of inputs to understand impact on modeling speech quality.

Analyzing performance of a convolution-based quality assessment model when it takes the following inputs:
1) hand crafted features: pitch, voice activity, frame energy, and 26 Mel-frequency coefficients and their deltas.
2) learnable features: model takes audio waveform as input and learns features optimal for quality prediction.
3) mel-spectrogram.
4) log-mel-spectrogram.

This project is an extension to the work of Gamper, Hannes, et al. proposed in "Intrusive and non-intrusive perceptual speech quality assessment using a convolutional neural network." published at IEEE WASPAA'19 workshop.
