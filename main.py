import argparse
import sys
import os

# Set environment variables to limit the number of threads used by libraries to 1 to avoid issues with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import concurrent.futures
from pathlib import Path
from mne.datasets.eegbci import eegbci

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from eeg import EEGClassifier
from settings import Settings
from train import train

dataset_path = "./datasets"


def download_dataset() -> None:
    print("Téléchargement du dataset EEG Motor Movement/Imagery complet...")
    print("Cela peut prendre plusieurs minutes...")

    _subjects = list(range(1, 110))  # Sujets 1 to 109
    _runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All tasks
    eegbci.load_data(subjects=_subjects, runs=_runs, path=dataset_path, update_path=False)

    print(f"Dataset téléchargé dans : {dataset_path}")


def print_verbose_results(_subject: int, _task: int, _accuracy, _score) -> None:
    print(f"S{_subject:03d} T{_task} | Accuracy: {_accuracy:.2%} (score: {_score.mean():.2} ~{_score.std():.2})")


def process_task(subject, task, _settings) -> tuple:
    egg = EEGClassifier(subject, task, _settings)
    epochs_train, epochs_data, labels = egg.run()
    _model, _accuracy, _score = train(epochs_train, epochs_data, labels)
    return subject, task, _model, _accuracy, _score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EDF and event files for a given subject and task.")

    parser.add_argument("-A", "--all", action="store_true", help="Run all tasks for all subjects")
    parser.add_argument("-d", "--download", action="store_true", help="Download complete dataset")
    parser.add_argument("-m", "--montage", action="store_true", help="Show montage plot")
    parser.add_argument("-p", "--plot", action="store_true", help="Show plots")
    parser.add_argument("-s", "--subject", type=int, nargs='+', default=[1], choices=range(1, 110), metavar="SUBJECT", help="Subject ID(s) (e.g., 1 2)")
    parser.add_argument("-t", "--task", type=int, nargs='+', default=[1], choices=range(1, 5), help="Task ID(s) (e.g., 1 2)")
    parser.add_argument("-T", "--all-task", action="store_true", help="Run all tasks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose result output for each subject and task")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel jobs to run (default: number of CPUs)")

    args = parser.parse_args()
    settings = Settings(args, dataset_path)

    if args.download:
        download_dataset()
        exit(0)

    subjects = args.subject
    tasks = args.task

    if args.all:
        print("Running all tasks for all subjects...")
        subjects = range(1, 110)
        tasks = range(1, 5)
    elif args.all_task:
        print(f"Running all tasks for subject(s) {', '.join(str(s) for s in subjects)}...")
        tasks = range(1, 5)

    tasks_result = [[], [], [], []]  # Initialize lists for each task

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # List of tasks to be executed
        futures = [
            executor.submit(process_task, subject, task, settings)
            for subject in subjects
            for task in tasks
        ]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                subject, task, model, accuracy, score = future.result()

                if args.verbose:
                    print_verbose_results(subject, task, accuracy, score)
                else:
                    print(f"S{subject:03d} T{task} | Accuracy: {accuracy:.2%}")

                tasks_result[task - 1].append(accuracy)
            except Exception as e:
                print(f"Une erreur est survenue lors du traitement: {e}")

    print("\nTask averages:")
    mean_acc = [sum(r) / len(r) for r in tasks_result]
    for i, r in enumerate(tasks_result):
        if len(r) == 0: continue
        print(f"Task {i+1}: {sum(r) / len(r):.2%}")

    task_mean = sum(mean_acc) / len(mean_acc)
    print(f"\nMean accuracy across all tasks: {task_mean:.2%}")
    print("\nAll analysis complete!")
