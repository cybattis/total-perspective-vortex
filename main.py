import argparse
import sys
from pathlib import Path
import mne
from mne.datasets.eegbci import eegbci
from sklearn.metrics import accuracy_score

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from eeg import EEGClassifier
from settings import Settings

dataset_path = "./datasets"

def download_dataset() -> None:
    print("Téléchargement du dataset EEG Motor Movement/Imagery complet...")
    print("Cela peut prendre plusieurs minutes...")

    _subjects = list(range(1, 110))  # Sujets 1 to 109
    _runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All tasks
    eegbci.load_data(subjects=_subjects, runs=_runs, path=dataset_path, update_path=False)

    print(f"Dataset téléchargé dans : {dataset_path}")

def print_verbose_results(_subject: int, _task: int, accuracy, score) -> None:
    print(f"S{_subject:03d} T{_task} | Accuracy: {accuracy:.2%} (score: {score.mean():.2} ~{score.std():.2})")

if __name__ == "__main__":
    mne.set_log_level("CRITICAL")

    parser = argparse.ArgumentParser(description="Process EDF and event files for a given subject and task.")

    parser.add_argument("-A", "--all", action="store_true", help="Run all tasks for all subjects")
    parser.add_argument("-d", "--download", action="store_true", help="Download complete dataset")
    parser.add_argument("-m", "--montage", action="store_true", help="Show montage plot")
    parser.add_argument("-p", "--plot", action="store_true", help="Show plots")
    parser.add_argument("-s", "--subject", type=int, nargs='+', default=[1], choices=range(1, 110), metavar="SUBJECT", help="Subject ID(s) (e.g., 1 2)")
    parser.add_argument("-t", "--task", type=int, nargs='+', default=[1], choices=range(1, 5), help="Task ID(s) (e.g., 1 2)")
    parser.add_argument("-T", "--all-task", action="store_true", help="Run all tasks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose result output for each subject and task")

    args = parser.parse_args()
    settings = Settings(args, dataset_path)

    if args.download:
        download_dataset()
        exit(0)

    subjects = args.subject
    tasks = args.task

    if args.all:
        subjects = range(1, 10)
        tasks = range(1, 5)
    elif args.all_task:
        tasks = range(1, 5)

    tasks_result = [[], [], [], []]  # Initialize lists for each task
    for subject in subjects:
        for task in tasks:
            egg = EEGClassifier(subject, task, settings)
            accuracy, score = egg.run()

            if args.verbose:
                print_verbose_results(subject, task, accuracy, score)

            tasks_result[task - 1].append(accuracy)

    print("\nTask averages:")
    mean_acc = [sum(r) / len(r) for r in tasks_result]
    for i, r in enumerate(tasks_result):
        if len(r) == 0: continue
        print(f"Task {i+1}: {sum(r) / len(r):.2%}")

    task_mean = sum(mean_acc) / len(mean_acc)
    print(f"\nMean accuracy across all tasks: {task_mean:.2%}")
    print("\nAll analysis complete!")
