# controllers/main_controller.py
from tkinter import filedialog

import mne
import sv_ttk

from src.egg import create_eeg_visualization
from utils import Utils

log = Utils.get_logger("MainController")

class MainController:
    """Handles user interactions and coordinates between model and view"""
    def __init__(self, model, view):
        self.raw_data = None
        self.model = model
        self.view = view
        self.model.add_observer(self)
        self._setup_event_handlers()


    def _setup_event_handlers(self):
        # Connect view events to controller methods
        self.view.bind_events({
            'select_file': self.handle_file_selection,
            'visualize_dataset': self.handle_visualization,
            'toggle_theme': self.handle_theme_toggle,
            'exit_app': self.handle_exit
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
                self.raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                self.model.load_dataset(file_path)
                if not autoload:
                    self.view.show_message(
                        f"Dataset loaded successfully!\nChannels: {len(self.raw_data.ch_names)}\nDuration: {self.raw_data.times[-1]:.1f}s")
            except Exception as e:
                self.view.show_message(f"Error loading dataset: {str(e)}", "error")
                self.raw_data = None


    def handle_visualization(self):
        self.view.clear_display_frame()

        if self.raw_data is not None:
            try:
                create_eeg_visualization(self.raw_data, self.view)
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


    def on_model_change(self, event_type, data):
        """Observer pattern - react to model changes"""
        if event_type == "dataset_loaded":
            filename = data.split("/")[-1]
            self.view.update_file_label(filename)
        elif event_type == "theme_changed":
            self.view.update_theme_indicator(data)