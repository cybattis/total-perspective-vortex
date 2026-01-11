"""
============================================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
============================================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP. A LinearDiscriminantAnalysis
classifier is used to classify left vs right hand movement from MEG data. This is based on the MNE
example: https://mne.tools/1.4/auto_examples/decoding/decoding_csp_eeg.html
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

RUN_TASKS = {
    1: [3, 7, 11],  # Real open and close fist
    2: [4, 8, 12],  # Imagine Open and close fist
    3: [5, 9, 13],  # Real open and close fist and feets
    4: [6, 10, 14]  # Imagine open and close fist and feet
}

ANNOTATIONS = {
    "T0": "Resting state",
    "T1": "Left hand, eyes open",
    "T2": "Right hand, eyes closed"
}

class EEGClassifier:
    def __init__(self, subjects, tasks, dataset_path, show_montage=False):
        self.subjects = subjects
        self.runs = RUN_TASKS[tasks]

        raw_fnames = eegbci.load_data(self.subjects, self.runs, path=dataset_path)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(self.raw)  # set channel names

        # Set montage
        montage = make_standard_montage("standard_1005")
        self.raw.set_montage(montage)

        if show_montage:
            self.raw.plot_sensors(show_names=True, sphere="eeglab")
            print(montage)


    def plot_raw_and_filtered(self):
        """
        Génère et affiche côte à côte les densités spectrales (PSD) des données brutes et filtrées.
        """
        # Création d'une figure avec 2 sous-graphiques (1 ligne, 2 colonnes)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Spectre des données brutes
        # picks="data" sélectionne automatiquement les bons canaux (ici EEG)
        spectrum_raw = self.raw.compute_psd()
        spectrum_raw.plot(average=True, picks="data", exclude="bads", amplitude=False, axes=ax1, show=False)
        ax1.set_title("Spectre : Données Brutes (Raw)")

        # 2. Spectre des données filtrées
        # Important : On utilise .copy() pour ne pas altérer self.raw en place
        raw_filtered = self.raw.copy()
        raw_filtered.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False)

        spectrum_filtered = raw_filtered.compute_psd()
        spectrum_filtered.plot(average=True, picks="data", exclude="bads", amplitude=False, axes=ax2, show=False)
        ax2.set_title("Spectre : Données Filtrées (Filtered)")

        plt.tight_layout()
        plt.show()



    def run(self):
        """
        Main function to run the EEG classification pipeline.
        """
        # avoid classification of evoked responses by using epochs that start 1s after cue onset.
        tmin, tmax = -1.0, 4.0
        event_id = dict(hands=2, feet=3)

        raw = self.raw.copy().crop(tmin=tmin, tmax=tmax)

        # Apply band-pass filter
        raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

        # #############################################################################
        # Read epochs

        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
        labels = epochs.events[:, -1] - 2

        # #############################################################################
        # # Classification with CSP + LDA

        # Define a monte-carlo cross-validation generator (reduce variance):
        epochs_data = epochs.get_data()
        epochs_data_train = epochs_train.get_data()
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        cv_split = cv.split(epochs_data_train)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([("CSP", csp), ("LDA", lda)], memory=None)
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

        # Printing the results
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1.0 - class_balance)
        print(
            "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
        )

        # plot CSP patterns estimated on full data for visualization
        csp.fit_transform(epochs_data_train, labels)

        # Plot CSP patterns (only first 4 components to avoid the warning)
        fig_patterns = csp.plot_patterns(
            epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5,
            components=np.arange(4), show=False
        )

        # Handle both single figure and list of figures
        if isinstance(fig_patterns, list):
            for idx, fig in enumerate(fig_patterns):
                fig.savefig(f"csp_patterns_{idx}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
            print(f"CSP patterns saved to csp_patterns_0.png to csp_patterns_{len(fig_patterns)-1}.png")
        else:
            fig_patterns.savefig("csp_patterns.png", dpi=100, bbox_inches='tight')
            print("CSP patterns saved to csp_patterns.png")
            plt.close(fig_patterns)

        # Plot CSP filters (only first 4 components)
        fig_filters = csp.plot_filters(
            epochs.info, ch_type="eeg", units="Filters (AU)", size=1.5,
            components=np.arange(4), show=False
        )

        # Handle both single figure and list of figures
        if isinstance(fig_filters, list):
            for idx, fig in enumerate(fig_filters):
                fig.savefig(f"csp_filters_{idx}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
            print(f"CSP filters saved to csp_filters_0.png to csp_filters_{len(fig_filters)-1}.png")
        else:
            fig_filters.savefig("csp_filters.png", dpi=100, bbox_inches='tight')
            print("CSP filters saved to csp_filters.png")
            plt.close(fig_filters)

        # #############################################################################
        # # Classification with cross-validation

        # Running classifier: test classifier on sliding window
        sfreq = raw.info["sfreq"]
        w_length = int(sfreq * 0.5)  # running classifier: window length
        w_step = int(sfreq * 0.1)  # running classifier: window step size
        w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

        scores_windows = []

        for train_idx, test_idx in cv_split:
            y_train, y_test = labels[train_idx], labels[test_idx]

            X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
            X_test = csp.transform(epochs_data_train[test_idx])

            # fit classifier
            lda.fit(X_train, y_train)

            # running classifier: test classifier on sliding window
            score_this_window = []
            for n in w_start:
                X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
                score_this_window.append(lda.score(X_test, y_test))
            scores_windows.append(score_this_window)

        # Plot scores over time
        w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

        # #############################################################################
        # Visualize classification over time

        fig = plt.figure(figsize=(8, 6))
        plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
        plt.axvline(0, linestyle="--", color="k", label="Onset")
        plt.axhline(0.5, linestyle="-", color="k", label="Chance")
        plt.xlabel("time (s)")
        plt.ylabel("classification accuracy")
        plt.title("Classification score over time")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("classification_over_time.png", dpi=100, bbox_inches='tight')
        print("Classification over time plot saved to classification_over_time.png")
        plt.close(fig)

        print("\nAll analysis complete! Check the generated PNG files for visualizations.")
