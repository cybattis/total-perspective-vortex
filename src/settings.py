import argparse


class Settings:
    def __init__(self, argument: argparse.Namespace):
        self.download = argument.download
        self.show_montage = argument.montage
        self.subjects = argument.subject
        self.task = argument.task
        self.no_plot = argument.no_plot