import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import argparse

root = "./datasets"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EDF and event files for a given subject and task.")
    parser.add_argument("--subject", type=str, default="S001", help="Subject ID (e.g., S001)")
    parser.add_argument("--task", type=str, default="1", help="Task ID (e.g., 1)")
    args = parser.parse_args()

    subject = args.subject
    task = args.task

    if len(task) == 1:
        task = "0" + task  # pad task with leading zero if necessary

    # raw_file = root + f"/{subject}/{subject}R{task}.edf"

    raw_file = mne.datasets.eegbci.load_data(1, [1], path=root, force_update=False)
    print(raw_file)

    raw = mne.io.read_raw_edf(raw_file[0], preload=False)
    events = raw.annotations

    raw.crop(tmax=60)
    raw.load_data()

    print(raw)
    print(events)

    builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
    for montage_name, montage_description in builtin_montages:
        print(f"{montage_name}: {montage_description}")

    # BCI2000 = mne.channels.make_standard_montage("standard_1020")
    # raw.set_montage(BCI2000)

