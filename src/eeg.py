"""
============================================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
============================================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP. A LinearDiscriminantAnalysis
classifier is used to classify left vs right hand movement from MEG data. This is based on the MNE
example: https://mne.tools/1.4/auto_examples/decoding/decoding_csp_eeg.html
"""
import mne

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from settings import Settings

experiments = [
    {
        "runs": [3, 7, 11],
        "mapping": {'T0': "rest", 'T1': "left_fist", 'T2': "right_fist"},
        "event_id": {"left_fist": 1, "right_fist": 2},
    },
    {
        "runs": [4, 8, 12],
        "mapping": {'T0': "rest", 'T1': "left_fist_imagine", 'T2': "right_fist_imagine"},
        "event_id": {"left_fist_imagine":1, "right_fist_imagine":2},
    },
    {
        "runs": [5, 9, 13],
        "mapping": {'T0': "rest", 'T1': "fists", 'T2': "feets"},
        "event_id": {"fists": 1, "feets": 2},
    },
    {
        "runs": [6, 10, 14],
        "mapping": {'T0': "rest", 'T1': "imagine_fists", 'T2': "imagine_feets"},
        "event_id": {"imagine_fists": 1, "imagine_feets": 2},
    }
]

crop_train = True


class EEGClassifier:
    def __init__(self, subject: int, task: int, settings: Settings):
        mne.set_log_level('CRITICAL')
        self.settings = settings
        self.subject = subject
        self.task = task

        self.runs = experiments[task-1].get("runs")
        self.mapping = experiments[task-1].get("mapping")
        self.event_id = experiments[task-1].get("event_id")

        self.raw = None


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
            "FC3", "FC1", "FCz", "FC2", "FC4",
            "C3", "C1", "Cz", "C2", "C4",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "Fpz",
        ]
        bad_channels = [x for x in channels if x not in good_channels]
        self.raw.drop_channels(bad_channels)

        # Filter
        raw_filtered = self.raw.copy()
        raw_filtered.notch_filter(60, method="iir")
        raw_filtered.filter(8.0, 40.0, fir_design="firwin", skip_by_annotation="edge") # 8Hz-40Hz for motor imagery

        return raw_filtered

    def run(self):
        """
        Main function to run the EEG classification pipeline.
        """
        # Load data
        raw_fnames = eegbci.load_data(self.subject, self.runs, path=self.settings.dataset_path, verbose=False)
        self.raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        self.raw.annotations.rename(self.mapping)
        self.raw.set_eeg_reference(projection=True)


        raw_filtered = self._preprocess()

        # Avoid classification of evoked responses by using epochs that start 1s after cue onset.
        tmin, tmax = -1.0, 2.0

        events, _ = events_from_annotations(raw_filtered, event_id=experiments[self.task-1]['event_id'], verbose=False)
        picks = pick_types(raw_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = Epochs(raw_filtered, events,
                        experiments[self.task-1]["event_id"],
                        tmin, tmax,
                        proj=True, picks=picks,
                        baseline=None, preload=True,
                        verbose=False)
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1]

        epochs_train = epochs.copy()
        if crop_train:
            epochs_train = epochs_train.crop(1, 2)

        return epochs_train, epochs_data, labels


class EEGStream:
    """
    Simulates a real-time EEG stream by playing back a recorded file.
    """
    def __init__(self, subject: int, task: int, settings: Settings, chunk_duration: float = 1.0):
        self.subject = subject
        self.task = task
        self.settings = settings
        self.chunk_duration = chunk_duration

        # Determine runs for the task (taking the last run as test data for simulation)
        # Note: In a real scenario, we might want to separate train/test data cleanly.
        # Here we pick one run to simulate streaming.
        self.runs = experiments[task-1].get("runs")
        self.test_run = [self.runs[-1]] # Use the last run for simulation

        self.raw = None
        self.sfreq = None

    def load_data(self):
        """Loads and prepares the raw data for streaming."""
        mne.set_log_level('CRITICAL')

        # Load data
        raw_fnames = eegbci.load_data(self.subject, self.test_run, path=self.settings.dataset_path, verbose=False)
        self.raw = read_raw_edf(raw_fnames[0], preload=True)

        # Standard preprocessing must match training
        eegbci.standardize(self.raw)
        montage = make_standard_montage("biosemi64")
        self.raw.set_montage(montage, on_missing='ignore')

        # Select channels (same as EEGClassifier)
        channels = self.raw.info["ch_names"]
        good_channels = [
            "FC3", "FC1", "FCz", "FC2", "FC4",
            "C3", "C1", "Cz", "C2", "C4",
            "CP3", "CP1", "CPz", "CP2", "CP4",
            "Fpz",
        ]
        bad_channels = [x for x in channels if x not in good_channels]
        self.raw.drop_channels(bad_channels)

        # Apply filters (simplification for simulation: pre-filter the whole file)
        # In a strict real-time system, we would filter buffer by buffer.
        self.raw.notch_filter(60, method="iir")
        self.raw.filter(8.0, 40.0, fir_design="firwin", skip_by_annotation="edge")

        self.sfreq = self.raw.info['sfreq']

    def stream(self):
        """
        Yields chunks of data.
        """
        if self.raw is None:
            self.load_data()

        data = self.raw.get_data() # shape (n_channels, n_times)
        n_samples = data.shape[1]
        chunk_size = int(self.sfreq * self.chunk_duration)

        for start in range(0, n_samples, chunk_size):
            end = start + chunk_size
            if end > n_samples:
                break

            chunk = data[:, start:end]
            # Yield (data_chunk, metadata/info) if needed, here just data
            yield chunk


