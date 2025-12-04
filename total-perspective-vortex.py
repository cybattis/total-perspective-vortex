# main.py - Application entry point
import tkinter as tk
import sv_ttk

from src.app_model import AppModel
from src.main_view import MainView
from src.main_controller import MainController
from utils import Utils

log = Utils.get_logger('Application')

class Application:
    def __init__(self):
        self.root = tk.Tk()
        sv_ttk.set_theme("dark")

        # Initialize MVC components
        self.model = AppModel()
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
