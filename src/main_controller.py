# controllers/main_controller.py
import os
import sys
from tkinter import filedialog

import mne
import sv_ttk

from src.egg import Egg
from utils import Utils

log = Utils.get_logger("MainController")

class MainController:
    """Handles user interactions and coordinates between model and view"""
    def __init__(self, model, view):
        self.egg = None
        self.model = model
        self.view = view
        self.model.add_observer(self)
        self._setup_event_handlers()


    def _setup_event_handlers(self):
        # Connect view events to controller methods
        self.view.bind_events({
            'select_file': self.handle_file_selection,
            'toggle_theme': self.handle_theme_toggle,
            'restart_app': self.handle_restart,
            'exit_app': self.handle_exit,
            'visualize_dataset': self.handle_visualization,
            'load_next_egg_data': self.handle_next_egg_data,
            'load_previous_egg_data': self.handle_prev_egg_data,
        })
        if self.model.current_file:
            self.handle_file_selection(autoload=True)


    def handle_file_selection(self, autoload=False):

        if autoload and self.model.current_file:
            file_path = self.model.current_file
        else:
            file_path = filedialog.askopenfilename(
                initialdir=Utils.get_root_folder_path() + "/datasets",
                title="Select Dataset",
                filetypes=(("EDF files", "*.edf"), ("All files", "*.*"))
            )

        if file_path:
            try:
                # Load EEG data using MNE
                raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                self.egg = Egg(raw_data, self.view)
                self.model.load_dataset(file_path)
                if not autoload:
                    self.view.show_message(
                        f"Dataset loaded successfully!\nChannels: {len(self.egg.raw_data.ch_names)}\n"
                        f"Duration: {self.egg.raw_data.times[-1]:.1f}s")
            except Exception as e:
                self.view.show_message(f"Error loading dataset: {str(e)}", "error")
                self.egg = None


    def handle_visualization(self):
        self.view.clear_display_frame()

        if self.egg is not None:
            try:
                self.egg.visualize()
            except Exception as e:
                self.view.show_message(f"Error visualizing dataset: {str(e)}", "error")
        else:
            self.view.show_message("Please load a dataset first")

    def handle_next_egg_data(self):
        self.view.clear_display_frame()

        if self.egg is not None:
            try:
                self.egg.visualize()
            except Exception as e:
                self.view.show_message(f"Error visualizing dataset: {str(e)}", "error")
        else:
            self.view.show_message("Please load a dataset first")

    def handle_prev_egg_data(self):
        self.view.clear_display_frame()

        if self.egg is not None:
            try:
                self.egg.visualize()
            except Exception as e:
                self.view.show_message(f"Error visualizing dataset: {str(e)}", "error")
        else:
            self.view.show_message("Please load a dataset first")


    def handle_theme_toggle(self):
        self.model.toggle_theme()
        sv_ttk.toggle_theme()


    def handle_exit(self):
        log.info("Exit application")
        try:
            # Close all matplotlib figures
            import matplotlib.pyplot as plt
            plt.close('all')

            # Properly destroy the tkinter window
            self.view.root.quit()
            self.view.root.destroy()
        except Exception as e:
            print(f"Error during exit: {e}")
            # Force exit if normal cleanup fails
            import sys
            sys.exit(0)


    def handle_restart(self):
        """Close resources, destroy the UI and re-exec the Python process to fully reload the app."""
        log.info("Restart application")
        try:
            # Close matplotlib figures
            import matplotlib.pyplot as plt
            plt.close('all')

            # Close MNE raw if open
            if self.raw_data is not None:
                self.raw_data.close()

            # Properly destroy the tkinter window
            self.view.root.quit()
            self.view.root.update_idletasks()
            self.view.root.destroy()

            # Replace the current process with a new Python interpreter running the same script/args
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"Error during restart: {e}")
            # Force exit if restart fails
            os._exit(1)


    def on_model_change(self, event_type, data):
        """Observer pattern - react to model changes"""
        if event_type == "dataset_loaded":
            filename = data.split("/")[-1]
            self.view.update_file_label(filename)
        elif event_type == "theme_changed":
            self.view.update_theme_indicator(data)