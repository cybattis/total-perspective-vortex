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
from mne.decoding import CSP, get_spatial_filter_from_estimator

from settings import Settings

RUN_TASKS = {
    1: [3, 7, 11],  # Real open and close fist
    2: [4, 8, 12],  # Imagine Open and close fist
    3: [5, 9, 13],  # Real open and close fist and feets
    4: [6, 10, 14]  # Imagine open and close fist and feet
}

class EEGClassifier:
    def __init__(self, settings: Settings, dataset_path: str):
        self.show_plots = not settings.no_plot
        self.subjects = settings.subjects
        self.runs = RUN_TASKS[settings.task]

        raw_fnames = eegbci.load_data(self.subjects, self.runs, path=dataset_path)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(self.raw)  # set channel names
        self.raw.annotations.rename(dict(T1="hands", T2="feet"))  # as documented on PhysioNet
        self.raw.set_eeg_reference(projection=True)

        # Set montage
        montage = make_standard_montage("standard_1005")
        self.raw.set_montage(montage)

        if settings.show_montage:
            self.raw.plot_sensors(show_names=True, sphere="eeglab")
            print(montage)

        # Filtering
        self.raw.notch_filter(60, fir_design='firwin')
        self.raw_filtered = self.raw.copy().filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False)


    def plot_raw_and_filtered(self):
        """
        Plot raw and filtered EEG data for comparison.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

        # Raw data
        times = self.raw.times
        data_raw = self.raw.get_data(picks="eeg", units="uV")
        # Filtered data
        data_filtered = self.raw_filtered.get_data(picks="eeg", units="uV")

        if self.show_plots:
            ax1.plot(times, data_raw.T, color='k', linewidth=0.2, alpha=0.5)
            ax1.set_title("Données Brutes (Raw)")
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel("Amplitude (µV)")

            ax2.plot(times, data_filtered.T, color='k', linewidth=0.2, alpha=0.5)
            ax2.set_title("Données Filtrées (7-30 Hz)")
            ax2.set_xlabel("Temps (s)")

            plt.suptitle("Comparaison Temporelle (Tous canaux)")
            plt.tight_layout()
            plt.show()


    def run(self):
        """
        Main function to run the EEG classification pipeline.
        """
        self.plot_raw_and_filtered()

        # Avoid classification of evoked responses by using epochs that start 1s after cue onset.
        tmin, tmax = -1.0, 4.0

        # #############################################################################
        # Read epochs
        picks = pick_types(self.raw_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(
            self.raw_filtered,
            event_id=["hands", "feet"],
            tmin=tmin,
            tmax=tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)
        labels = epochs.events[:, -1] - 2

        # #############################################################################
        # # Classification with CSP + LDA

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        epochs_data = epochs.get_data(copy=False)
        epochs_data_train = epochs_train.get_data(copy=False)
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        cv_split = cv.split(epochs_data_train)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=8, reg='ledoit_wolf', log=True, norm_trace=False)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([("CSP", csp), ("LDA", lda)], memory=None)
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

        # Printing the results
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1.0 - class_balance)
        print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

        # plot eigenvalues and patterns estimated on full data for visualization
        csp.fit_transform(epochs_data, labels)

        if self.show_plots:
            spf = get_spatial_filter_from_estimator(csp, info=epochs.info)
            spf.plot_scree()
            spf.plot_patterns(components=np.arange(4))

        # #############################################################################
        # # Classification with cross-validation

        sfreq = self.raw_filtered.info["sfreq"]
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
                X_test = csp.transform(epochs_data[test_idx][:, :, n: (n + w_length)])
                score_this_window.append(lda.score(X_test, y_test))
            scores_windows.append(score_this_window)

        # Plot scores over time
        w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

        if self.show_plots:
            plt.figure()
            plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
            plt.axvline(0, linestyle="--", color="k", label="Onset")
            plt.axhline(0.5, linestyle="-", color="k", label="Chance")
            plt.xlabel("time (s)")
            plt.ylabel("classification accuracy")
            plt.title("Classification score over time")
            plt.legend(loc="lower right")
            plt.show()
            plt.close()

        # Calcul des statistiques résumées
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_scores_time = np.mean(scores_windows, 0)
        max_score_time = np.max(mean_scores_time)

        print("\n" + "=" * 60)
        print(f"{'RÉSUMÉ DES RÉSULTATS DE CLASSIFICATION':^60}")
        print("=" * 60)
        print(f"{'Précision Moyenne (Cross-Val)':<30} : {mean_score:.2%} (+/- {std_score:.2%})")
        print(f"{'Niveau de Chance':<30} : {class_balance:.2%}")
        print("-" * 60)
        print("Analyse Temporelle :")
        print(f"{'  Score Max Atteint':<30} : {max_score_time:.2%}")
        print(f"{'  Écart-type sur le temps':<30} : {np.std(mean_scores_time):.4f}")
        print("-" * 60)

        # Configuration de l'affichage numpy pour le tableau de scores (plus compact)
        with np.printoptions(formatter={'float': lambda x: f"{x: 0.2f}"}, linewidth=60):
            print("Profil des scores (fenêtre glissante) :")
            print(mean_scores_time)
        print("=" * 60 + "\n")

        print("\nAll analysis complete! Check the generated PNG files for visualizations.")
