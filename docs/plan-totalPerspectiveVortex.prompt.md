## Plan: Total Perspective Vortex Implementation

This project aims to build a BCI pipeline for coding motor imagery signals. We will unify existing data handling, 
implement a custom Common Spatial Patterns (CSP) algorithm for dimensionality reduction, wrap it in a `scikit-learn` pipeline, 
and simulate real-time classification.

### Steps
1.  **Data handling unification**: Update `src/egg.py` to handle MNE dataset fetching (Motor Imagery tasks),
filtering (Band-pass 8-30Hz), and epoching (using annotations/events).
    - [x] write a script to visualize raw data then filter it to keep only useful frequency bands, and visualize again after this preprocessing.
2.  **Custom CSP Transformer**: Create `src/csp.py` implementing a class with `fit` and `transform` methods using 
covariance matrices and eigenvalue decomposition (simulating `mne.decoding.CSP` logic).
3.  **Pipeline Construction**: In `main.py`, build a `scikit-learn` Pipeline combining the Custom CSP and a classifier
(e.g., LDA (`LinearDiscriminantAnalysis`) or SVM) and train it on a subset of the data.
4.  **Real-time Simulation**: Implement a "playback" class/generator in `src/egg.py` that yields small chunks of data
(e.g., 1s windows) from a test file to simulate a live stream.
5.  **Integration & Loop**: Create the main loop in `main.py` that consumes the simulated stream, applies the trained
pipeline, and outputs predictions (A vs B) in real-time.

### Further Considerations
1.  **Algorithm Selection**: CSP is the gold standard for Motor Imagery. PCA is easier but performs poorly on EEG spatial filtering.
I recommend implementing generic CSP.
2.  **Event Handling**: We will classify "Hands" vs "Feet" (or similar MI tasks) based on the PhysioNet dataset codes (T1 vs T2).
3.  **Validation**: A simple confusion matrix on a hold-out set should be printed before starting the real-time simulation.

