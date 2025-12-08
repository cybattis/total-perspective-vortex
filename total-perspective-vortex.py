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


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.settings = Settings()

        sv_ttk.set_theme(self.settings.get('theme'))

        # Global key bindings
        self.root.bind("<Key>", self.key_handler)

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

    def key_handler(self, event):
        # Binding is Control-r, so this handler is called only when Ctrl+R is pressed.
        print("event:", event)
        if event.keysym.lower() == 'r' and (event.state & 0x4):
            log.info("Ctrl+R detected - Restarting application...")
            self.controller.handle_restart()

        if event.keysym.lower() == 'q' and (event.state & 0x4):
            log.info("Ctrl+Q detected - Exiting application...")
            self.controller.handle_exit()

        if event.keysym.lower() == 't' and (event.state & 0x4):
            log.info("Ctrl+T detected - Toggling theme...")
            self.controller.handle_theme_toggle()


if __name__ == "__main__":
    app = Application()
    app.run()
