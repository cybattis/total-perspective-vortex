import argparse
import sys

from pathlib import Path
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
    parser.add_argument("-s", "--subject", type=int, default=1, help="Subject ID (e.g., 1)")
    parser.add_argument("-t", "--task", type=int, default=1, help="Task ID (e.g., 1)")
    parser.add_argument("-nop", "--no-plot", action="store_true", help="Do not show plots")

    settings = Settings(parser.parse_args())

    if settings.download:
        print("Téléchargement du dataset EEG Motor Movement/Imagery complet...")
        print("Cela peut prendre plusieurs minutes...")

        subjects = list(range(1, 110))  # Sujets 1 to 109
        runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All tasks
        eegbci.load_data(subjects=subjects, runs=runs, path=dataset_path, update_path=False)

        print(f"Dataset téléchargé dans : {dataset_path}")

    egg = EEGClassifier(settings, dataset_path)
    egg.run()
