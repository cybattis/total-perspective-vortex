# total-perspective-vortex

This subject aims to create a brain computer interface based on electroencephalographic
data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG
reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B
in a t0 to tn timeframe.

## Goals

- Process EEG datas (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline object from scikit-learn
- Classify a data stream in "real time"

## Tools

- MNE a library specialized in EEG data processing
- Scikit-learn a library specialized in machine learning.

## Data

The data was mesured during a motor imagery experiment, where people had to
do or imagine a hand or feet movement. Those people were told to think or do a move-
ment corresponding to a symbol displayed on screen. The results are cerebral signals
with labels indicating moments where the subject had to perform a certain task.

## Application steps

1. Preprocessing, parsing and formating
    - Script to parse and visualize raw data fron physionet
    - Filter out unwanted frequency
    - Visualize results

    > [!TIP]
    > One example is to use the power of the signal by frequency and by channel to the pipeline’s input.
    > Most of the algorithms linked to filtering and obtaining the signal’s specter use fourier transform or wavelet transform (cf. bonus).

2. Treatment pipeline
    - Dimensionality reduction algorithm (ie : PCA, ICA, CSP, CSSP...).
    - Classification algorithm, there is plenty of choice among those available in sklearn, to output the decision of what data chunk correspond to what kind of motion.
    - "Playback" reading on the file to simulate a data stream.<br/>
    <br/>
   1. Use sklearn and MNE algorithms for testing
   2. Implementation dimensionality reduction algorithm.

3. Train, Validation and Test
