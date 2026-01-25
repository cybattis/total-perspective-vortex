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
from matplotlib.gridspec import GridSpec
from mne.viz import plot_topomap

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from settings import Settings

RUN_TASKS = {
    1: [3, 7, 11],  # Real open and close fist
    2: [4, 8, 12],  # Imagine Open and close fist
    3: [5, 9, 13],  # Real open and close fist and feets
    4: [6, 10, 14]  # Imagine open and close fist and feet
}

class EEGClassifier:
    def __init__(self, subject: int, task: int, settings: Settings):
        self.show_plots = not settings.no_plot
        self.subject = subject
        self.runs = RUN_TASKS[task]

        raw_fnames = eegbci.load_data(self.subject, self.runs, path=settings.dataset_path, verbose=False)
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
        self.raw.notch_filter(60, fir_design='firwin', skip_by_annotation='edge')
        self.raw_filtered = self.raw.copy().filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")


    def _plot_results(self, w_times, mean_scores, csp, epochs, stats):
        """
        Combine all visualizations into a single final figure.
        Adds textual statistics to the right of the temporal plot.
        """
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        # Grid: 3 rows, 4 columns
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])

        # 1. Raw Signals
        times = self.raw.times
        data_raw = self.raw.get_data(picks="eeg", units="uV")
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(times, data_raw.T, color='k', linewidth=0.2, alpha=0.5)
        ax1.set_title("Raw Data")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µV)")

        # 2. Filtered Signals
        data_filtered = self.raw_filtered.get_data(picks="eeg", units="uV")
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(times, data_filtered.T, color='k', linewidth=0.2, alpha=0.5)
        ax2.set_title("Filtered Data (7-30 Hz)")
        ax2.set_xlabel("Time (s)")

        # 3. Temporal Classification Score
        ax3 = fig.add_subplot(gs[1, :3])
        ax3.plot(w_times, mean_scores, label="Mean Score", color='#1f77b4', linewidth=1.5)
        ax3.axvline(0, linestyle="--", color="k", label="Onset")
        ax3.axhline(0.5, linestyle="-", color="r", label="Chance (50%)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Performance Over Time", fontweight='bold')
        ax3.legend(loc="lower right", framealpha=0.9)
        ax3.grid(True, linestyle=':', alpha=0.6)

        # 4. Textual Statistics
        ax_text = fig.add_subplot(gs[1, 3])
        ax_text.axis("off")

        stats_content = (
            f"Mean (CV)     : {stats['mean']:.1%}\n"
            f"Std Dev       : ±{stats['std']:.1%}\n"
            f"Chance Level  : {stats['balance']:.1%}\n"
            f"Max Score     : {stats['max']:.1%}"
        )
        text_str = f"RESULTS\n{'─' * 18}\n{stats_content}"

        # Style box to prettify the text
        props = dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='#bdbdbd', alpha=0.9)
        ax_text.text(0.05, 0.5, text_str, transform=ax_text.transAxes,
                     va='center', ha='left', fontsize=11, family='monospace',
                     bbox=props, linespacing=1.6)

        # 5. CSP Patterns (Top 4 components) - full bottom row
        for i in range(min(4, len(csp.patterns_))):
            ax_pattern = fig.add_subplot(gs[2, i])
            plot_topomap(csp.patterns_[i], epochs.info, axes=ax_pattern, show=False)
            ax_pattern.set_title(f"CSP Pattern {i+1}")

        plt.suptitle(f"EEG Analysis Summary - Subject(s): {self.subject}", fontsize=16)
        plt.show()


    def run(self):
        """
        Main function to run the EEG classification pipeline.
        """
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

        # plot eigenvalues and patterns estimated on full data for visualization
        csp.fit(epochs_data, labels)

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
            csp.transform(epochs_data_train[test_idx])

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

        # Calcul des statistiques résumées
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1.0 - class_balance)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_scores_time = np.mean(scores_windows, 0)
        max_score_time = np.max(mean_scores_time)

        # Build stats dict for plotting
        stats = {
            "mean": mean_score,
            "std": std_score,
            "balance": class_balance,
            "max": max_score_time
        }

        if not self.show_plots:
            print("\n" + "=" * 60)
            print(f"{'CLASSIFICATION RESULTS SUMMARY':^60}")
            print("=" * 60)
            print(f"{'Average Accuracy (Cross-Val)':<30} : {mean_score:.2%} (+/- {std_score:.2%})")
            print(f"{'Chance Level':<30} : {class_balance:.2%}")
            print("-" * 60)
            print("Temporal Analysis:")
            print(f"{'  Max Score Achieved':<30} : {max_score_time:.2%}")
            print(f"{'  Std Dev over time':<30} : {np.std(mean_scores_time):.4f}")
            print("-" * 60)

            # # Configure numpy display for the scores array (more compact)
            # with np.printoptions(formatter={'float': lambda x: f"{x: 0.2f}"}, linewidth=60):
            #     print("Score profile (sliding window):")
            #     print(mean_scores_time)
            # print("=" * 60 + "\n")
        else:
            self._plot_results(w_times, mean_scores_time, csp, epochs, stats)

        return mean_score
