import argparse


class Settings:
    def __init__(self, argument: argparse.Namespace, dataset_path: str = ""):
        self.dataset_path = dataset_path
        self.show_montage = argument.montage
        self.plot = argument.plot
        self.verbose = argument.verbose
