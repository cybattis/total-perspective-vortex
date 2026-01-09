import mne
import argparse

import numpy as np
from matplotlib import pyplot as plt

root = "./datasets"

def raw_info(_raw):
    """Print raw info summary."""
    n_time_samps = _raw.n_times
    time_secs = _raw.times
    ch_names = _raw.ch_names
    n_chan = len(ch_names)  # note: there is no _raw.n_channels attribute
    print(
        f"the (cropped) sample data object has {n_time_samps} time samples and "
        f"{n_chan} channels."
    )
    print(f"The last time sample is at {time_secs[-1]} seconds.")
    print("The first few channel names are {}.".format(", ".join(ch_names[:3])))
    print()  # insert a blank line in the output

    # some examples of _raw.info:
    print("bad channels:", _raw.info["bads"])  # chs marked "bad" during acquisition
    print(_raw.info["sfreq"], "Hz")  # sampling frequency
    print(_raw.info["description"], "\n")  # miscellaneous acquisition info

    print(_raw.info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EDF and event files for a given subject and task.")
    parser.add_argument("-s", "--subject", type=int, default=1, help="Subject ID (e.g., 1)")
    parser.add_argument("-t", "--task", type=str, default="base1", help="Task ID (e.g., 1)")
    parser.add_argument("-sm", "--show-montage", action="store_true", help="Show montage plot")
    parser.add_argument("-r", "--report", action="store_true", help="Generate report page")
    args = parser.parse_args()

    subject = args.subject

    TASK_MAPPING = {
        'base1': [1],
        'base2': [2],
        'motor1':   [3, 7, 11],
        'imagery1': [4, 8, 12],
        'motor2':   [5, 9, 13],
        'imagery2': [6, 10, 14]
    }
    task = TASK_MAPPING.get(args.task, 0)

    print(f"Subject: {subject}, Task: {task}")
    print("Loading data...")

    raw_file = mne.datasets.eegbci.load_data(args.subject, task, path=root, force_update=False)

    # Load raw EDF file
    raw = mne.io.read_raw_edf(raw_file[0], preload=False)
    mne.datasets.eegbci.standardize(raw)

    # Set montage
    BCI2000 = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(BCI2000)
    if args.show_montage:
        raw.plot_sensors(show_names=True, sphere="eeglab")
        print(BCI2000)

    raw.load_data()
    # Set EEG reference to average of T9 and T10
    raw.set_eeg_reference(ref_channels=['T9', 'T10'])
    raw_info(raw)

    # Notch filter at 60 Hz to remove power line noise
    raw.notch_filter(freqs=60.0)

    raw.plot()

    if args.report:
        report = mne.Report(title="EGG report")
        report.add_raw(raw=raw, title="EGG raw report", psd=False)  # omit PSD plot
        report.save("report_raw.html", overwrite=True, open_browser=False)

    # update psd after adding bad channels
    print("Bad channels selected:", raw.info["bads"])

    # Plot power spectral density
    spectrum = raw.compute_psd()
    spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False)

    raw2 = raw.copy()
    raw2.info["bads"] = []
    events, event_id = mne.events_from_annotations(raw2)
    epochs = mne.Epochs(raw2, events=events, event_id=event_id)
    epochs.average().plot()

    plt.show(block=True)
    raw.close()
    print("Done.")