# main.py - Application entry point
import tkinter as tk
import sv_ttk
import json, os

from settings import Settings
from src.app_model import AppModel
from src.main_view import MainView
from src.main_controller import MainController
from utils import Utils

log = Utils.get_logger('Application')


def load_config():
    try:
        cfg_path = os.path.join(Utils.get_root_folder_path(), "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        log.info("Loaded config: %s", cfg_path)
        return Settings(config)
    except Exception as e:
        log.error("Could not load config.json: %s", e)
        exit(1)


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.settings = load_config()

        sv_ttk.set_theme(self.settings.get('theme'))

        # Initialize MVC components
        self.model = AppModel(self.settings)
        self.view = MainView(self.root)
        self.controller = MainController(self.model, self.view)

        self.print_info()

    def run(self):
        log.info("Running Application...")
        self.root.mainloop()

    def print_info(self):
        log.info("Application: %s", self.root.title())
        log.info("Version tk: %s", tk.TkVersion)
        log.info("Theme: %s", sv_ttk.get_theme())
        log.info("Root window size: %sx%s", self.root.winfo_width(), self.root.winfo_height())
        log.info("Root path: %s", Utils.get_root_folder_path())


if __name__ == "__main__":
    app = Application()
    app.run()
