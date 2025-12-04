import logging
import os


class Utils:

    @staticmethod
    def get_logger(name=''):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to self.logger
        logger.addHandler(ch)
        return logger

    @classmethod
    def get_root_folder_path(cls):
        return os.path.dirname(os.path.abspath(__file__))