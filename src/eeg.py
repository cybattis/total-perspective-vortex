"""
============================================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
============================================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP. A LinearDiscriminantAnalysis
classifier is used to classify left vs right hand movement from MEG data. This is based on the MNE
example: https://mne.tools/1.4/auto_examples/decoding/decoding_csp_eeg.html
"""
import matplotlib.pyplot as plt
import mne
from matplotlib.gridspec import GridSpec
from mne.viz import plot_topomap
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from settings import Settings

experiments = [
    {
        "runs": [3, 7, 11],
        "mapping": {'T0': "rest", 'T1': "left/fist", 'T2': "right/fist"},
        "event_id": {"left/fist": 1, "right/fist": 2},
    },
    {
        "runs": [4, 8, 12],
        "mapping": {'T0': "rest", 'T1': "left/imaginefist", 'T2': "right/imaginefist"},
        "event_id": {"left/imaginefist":1, "right/imaginefist":2},
    },
    {
        "runs": [5, 9, 13],
        "mapping": {'T0': "rest", 'T1': "top/fists", 'T2': "bottom/feets"},
        "event_id": {"top/fists": 1, "bottom/feets": 2},
    },
    {
        "runs": [6, 10, 14],
        "mapping": {'T0': "rest", 'T1': "top/imaginefists", 'T2': "top/imaginefeets"},
        "event_id": {"top/imaginefists": 1, "top/imaginefeets": 2},
    }
]

crop_train = True

class EEGClassifier:
    def __init__(self, subject: int, task: int, settings: Settings):
        mne.set_log_level('WARNING')

        self.settings = settings
        self.subject = subject
        self.task = task
        self.runs = experiments[task-1].get("runs")
        self.mapping = experiments[task-1].get("mapping")
        self.event_id = experiments[task-1].get("event_id")

        # Load data
        raw_fnames = eegbci.load_data(self.subject, self.runs, path=settings.dataset_path, verbose=False)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

        self.raw.annotations.rename(self.mapping)
        self.raw.set_eeg_reference(projection=True)


    def _preprocess(self):
        eegbci.standardize(self.raw)  # set channel names
        montage = make_standard_montage("biosemi64")
        self.raw.set_montage(montage, on_missing='ignore')

        if self.settings.show_montage:
            self.raw.plot_sensors(show_names=True, sphere="eeglab")
            print(montage)

        # Select channels
        channels = self.raw.info["ch_names"]
        good_channels = [
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "Fpz",
        ]
        bad_channels = [x for x in channels if x not in good_channels]
        self.raw.drop_channels(bad_channels)

        # Filter
        raw_filtered = self.raw.copy()
        raw_filtered.notch_filter(60, method="iir")
        raw_filtered.filter(7.0, 32.0, fir_design="firwin", skip_by_annotation="edge") # 8Hz-40Hz for motor imagery

        return raw_filtered

    def _create_model(self):
        # Decomposer
        csp = CSP(n_components=16)
        # Classifier l1_ratio
        logr = LogisticRegression(solver='liblinear', verbose=False)
        # Pipeline
        clf = Pipeline([("CSP", csp), ("LogisticRegression", logr)], memory=None, verbose=False)
        return clf

    def run(self):
        """
        Main function to run the EEG classification pipeline.
        """
        raw_filtered = self._preprocess()

        # Avoid classification of evoked responses by using epochs that start 1s after cue onset.
        tmin, tmax = -1.0, 2.0

        events, _ = events_from_annotations(raw_filtered, event_id=experiments[self.task-1]['event_id'], verbose=False)
        picks = pick_types(raw_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = Epochs(raw_filtered, events, experiments[self.task-1]["event_id"], tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=False)
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1]

        # monte-carlo cross-validation generator:
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # Display accuracy
        model = self._create_model()
        epochs_train = epochs.copy()
        if crop_train:
            epochs_train = epochs_train.crop(1, 2)
        score = cross_val_score(model, epochs_train.get_data(), labels, cv=cv, verbose=False)
        model.fit(epochs_data, labels)
        accuracy = model.score(epochs_data, labels)

        return accuracy, score

    def _plot_results(self, w_times, mean_scores, csp, epochs, stats, raw_filtered):
        """
        Combine all visualizations into a single final figure.
        Adds textual statistics to the right of the temporal plot.
        """
        x_label = "Time (s)"

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        # Grid: 3 rows, 4 columns
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])

        # 1. Raw Signals
        times = self.raw.times
        data_raw = self.raw.get_data(picks="eeg", units="uV")
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(times, data_raw.T, color='k', linewidth=0.2, alpha=0.5)
        ax1.set_title("Raw Data")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Amplitude (µV)")

        # 2. Filtered Signals
        data_filtered = raw_filtered.get_data(picks="eeg", units="uV")
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(times, data_filtered.T, color='k', linewidth=0.2, alpha=0.5)
        ax2.set_title("Filtered Data (7-30 Hz)")
        ax2.set_xlabel(x_label)

        # 3. Temporal Classification Score
        ax3 = fig.add_subplot(gs[1, :3])
        ax3.plot(w_times, mean_scores, label="Mean Score", color='#1f77b4', linewidth=1.5)
        ax3.axvline(0, linestyle="--", color="k", label="Onset")
        ax3.axhline(0.5, linestyle="-", color="r", label="Chance (50%)")
        ax3.set_xlabel(x_label)
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
                     va='center', ha='left', fontsize=10, family='monospace',
                     bbox=props, linespacing=1.6)

        # 5. CSP Patterns (Top 4 components) - full bottom row
        for i in range(min(4, len(csp.patterns_))):
            ax_pattern = fig.add_subplot(gs[2, i])
            plot_topomap(csp.patterns_[i], epochs.info, axes=ax_pattern, show=False)
            ax_pattern.set_title(f"CSP Pattern {i+1}")

        plt.suptitle(f"EEG Analysis Summary - Subject(s): {self.subject}", fontsize=12)
        plt.show()