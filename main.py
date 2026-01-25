import argparse
import sys

from pathlib import Path

import mne
from mne.datasets.eegbci import eegbci

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from eeg import EEGClassifier
from settings import Settings

dataset_path = "./datasets"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EDF and event files for a given subject and task.")
    parser.add_argument("-d", "--download", action="store_true", help="Download all dataset")
    parser.add_argument("-m", "--montage", action="store_true", help="Show montage plot")
    parser.add_argument("-nop", "--no-plot", action="store_true", help="Do not show plots")
    parser.add_argument("-s", "--subject", type=int, default=1, help="Subject ID (e.g., 1)")
    parser.add_argument("-t", "--task", type=int, default=1, help="Task ID (e.g., 1)")
    parser.add_argument("-A", "--all", action="store_false", help="Run all tasks for all subjects")

    args = parser.parse_args()
    settings = Settings(args)
    settings.dataset_path = dataset_path

    if settings.download:
        print("Téléchargement du dataset EEG Motor Movement/Imagery complet...")
        print("Cela peut prendre plusieurs minutes...")

        subjects = list(range(1, 110))  # Sujets 1 to 109
        runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All tasks
        eegbci.load_data(subjects=subjects, runs=runs, path=dataset_path, update_path=False)

        print(f"Dataset téléchargé dans : {dataset_path}")

    mne.set_log_level("WARNING")

    if not args.all:
        print(f"Subject {args.subject} - Task {args.task}")
        egg = EEGClassifier(args.subject, args.task, settings)
        egg.run()
    else:
        subjects = list(range(1, 10))
        mean_overall = []
        for t in range(1, 5):
            mean_task = []
            for subject in subjects:
                task = t
                print(f"Subject {subject} - Task {task}")
                egg = EEGClassifier(subject, task, settings)
                mean = egg.run()
                mean_task.append(mean)
            mean_overall.append(sum(mean_task)/len(mean_task))

        print("\n")
        print("Task averages:")
        print(f"Task 1: {mean_overall[0]:.2%}")
        print(f"Task 2: {mean_overall[1]:.2%}")
        print(f"Task 3: {mean_overall[2]:.2%}")
        print(f"Task 4: {mean_overall[3]:.2%}")
        print("\n")
        print(f"Overall mean accuracy = {sum(mean_overall)/len(mean_overall):.2%}")

    print("\nAll analysis complete! Check the generated PNG files for visualizations.")