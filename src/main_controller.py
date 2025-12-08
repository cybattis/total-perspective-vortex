# controllers/main_controller.py
import os
import sys
from tkinter import filedialog

import sv_ttk
from matplotlib import pyplot as plt

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
            'toggle_theme': self.handle_theme_toggle,
            'restart_app': self.handle_restart,
            'exit_app': self.handle_exit,
            'load_dataset': self.handle_load_dataset,
            'visualize_dataset': self.handle_visualization,
            'reset_dataset': self.handle_reset_dataset,
            'apply_preprocessing': self.handle_preprocessing,
        })
        if self.model.current_file:
            self.handle_load_dataset(autoload=True)

    def handle_load_dataset(self, file_path=None, autoload=False):
        if autoload and self.model.current_file:
            file_path = self.model.current_file

        if file_path:
            try:
                # Load EEG data using MNE
                self.egg = Egg(file_path)
                self.model.set_dataset(file_path)

                if not autoload:
                    self.view.show_message(
                        f"Dataset loaded successfully!\nChannels: {len(self.egg.raw_data.ch_names)}\n"
                        f"Duration: {self.egg.raw_data.times[-1]:.1f}s")
            except Exception as e:
                self.view.show_message(f"Error loading dataset: {str(e)}", "error")
                self.egg = None

    def handle_reset_dataset(self):
        self.egg.preprocess_data = self.egg.raw_data.copy()
        self.clear_and_close_figures()

    def clear_and_close_figures(self):
        """Helper to clear the view's display frames and close the figures."""
        fig1 = self.view.clear_display_frame()
        fig2 = self.view.clear_psd_frame()
        if fig1:
            plt.close(fig1)
        if fig2:
            plt.close(fig2)

    def handle_visualization(self):
        self.clear_and_close_figures()

        if self.egg is not None:
            try:
                # Raw data plot
                fig_raw = self.egg.visualize()
                self.view.embed_figure_in_display_frame(fig_raw)

                # PSD plot
                fig_psd = self.egg.visualize_psd()
                self.view.embed_psd_figure(fig_psd)

                self.view.add_info_label(self.egg.detail)
            except Exception as e:
                self.view.show_message(f"Error visualizing dataset: {str(e)}", "error")
        else:
            self.view.show_message("Please load a dataset first")

    def handle_preprocessing(self):
        if self.egg is not None:
            try:
                self.egg.preprocess()
                self.handle_visualization()
            except Exception as e:
                self.view.show_message(f"Error during preprocessing: {str(e)}", "error")
        else:
            self.view.show_message("Please load a dataset first")

    # Application control handlers
    # ============================
    def on_model_change(self, event_type, data):
        """Observer pattern - react to model changes"""
        if event_type == "dataset_loaded":
            filename = data.split("/")[-1]
            self.view.update_file_label(filename)
        elif event_type == "theme_changed":
            self.view.update_theme_indicator(data)

    def handle_theme_toggle(self):
        self.model.toggle_theme()
        sv_ttk.toggle_theme()

    def handle_exit(self):
        log.info("Exit application")
        try:
            # Close domain resources
            plt.close('all')
            self.egg.close()
            # --------------------------------

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
            # Close domain resources
            plt.close('all')
            self.egg.close()
            # --------------------------------

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