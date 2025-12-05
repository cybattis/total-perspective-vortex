# views/main_view.py
from tkinter import ttk, messagebox
import tkinter as tk

from utils import Utils


class MainView:
    """Handles UI layout and display"""
    def __init__(self, root):
        self.root = root
        self.root_folder = Utils.get_root_folder_path()
        self.event_handlers = {}

        # UI Components
        self.file_label = None
        self.canvas = None
        self.detail_panel = None

        # Initialize UI
        self._setup_window()
        self._setup_menu()
        self._setup_main_layout()


    def _setup_window(self):
        self.root.title("Total Perspective Vortex")
        self.root.geometry("1280x720")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        # Bind window close event to exit handler
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._trigger_event('exit_app'))


    def _setup_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load dataset", command=lambda: self._trigger_event('select_file'))
        file_menu.add_separator()
        file_menu.add_command(label="Restart", command=lambda: self._trigger_event('restart_app'))
        file_menu.add_command(label="Exit", command=lambda: self._trigger_event('exit_app'))
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Toggle Theme", command=lambda: self._trigger_event('toggle_theme'))
        menubar.add_cascade(label="Settings", menu=settings_menu)

        self.root.config(menu=menubar)


    def _setup_main_layout(self):
        # Configuration frame
        side_panel = ttk.Frame(self.root)
        side_panel.grid(row=0, column=0, sticky="nsew", padx=5)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)
        side_panel.columnconfigure(0, weight=1)

        # Configuration Frame
        upper_frame = ttk.Frame(side_panel, padding=10, borderwidth=2, relief="ridge")
        upper_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        ttk.Label(upper_frame, text="Configuration").pack(anchor="w")

        # Load dataset button
        file_frame = ttk.Frame(upper_frame)
        file_frame.pack(anchor="w", pady=10)
        ttk.Button(
            file_frame,
            text="Load Dataset",
            command=lambda: self._trigger_event('select_file')
        ).pack(side='left')
        self.file_label = ttk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.pack(side='left', padx=8)

        # Visualize button
        visu_frame = ttk.Frame(upper_frame)
        visu_frame.pack(anchor="w", pady=10)
        ttk.Button(
            visu_frame,
            text="Visualize",
            command=lambda: self._trigger_event('visualize_dataset')
        ).pack(side='left')

        # Clear button
        ttk.Button(
            visu_frame,
            text="X",
            command=self.clear_display_frame
        ).pack(side='left', padx=8)

        # control buttons
        # Backward, forward, play, pause can be added here in future
        control_frame = ttk.Frame(upper_frame)
        control_frame.pack(anchor="w", pady=10)
        # Placeholder for future control buttons
        ttk.Button(control_frame, text="<<", command=lambda: self._trigger_event('load_next_egg_data')).pack(side='left')
        ttk.Button(control_frame, text="").pack(side='left', padx=5)
        ttk.Button(control_frame, text=">>", command=lambda: self._trigger_event('load_prev_egg_data')).pack(side='left')

        # Details frames
        self.detail_panel = ttk.Frame(side_panel, padding=10, borderwidth=2, relief="ridge")
        self.detail_panel.grid(row=1, column=0, sticky="nsew")

        ttk.Label(self.detail_panel, text="Details", anchor="w").pack(padx=5)

        # Visual Display Frame
        self.display_frame = ttk.Frame(self.root, padding=20, borderwidth=2, relief="ridge")
        self.display_frame.grid(row=0, column=1, sticky="nsew")
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)

        self.default_label = ttk.Label(self.display_frame, text="Visual Display Area")
        self.default_label.pack(anchor="center", expand=True)


    def _trigger_event(self, event_name):
        """Trigger an event handler if it exists"""
        if event_name in self.event_handlers:
            self.event_handlers[event_name]()


    # Public Methods
    # =================================================
    def clear_display_frame(self):
        """Clear all widgets from display frame"""
        # Clean up matplotlib canvas if it exists
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
            import matplotlib.pyplot as plt
            plt.close(self.canvas.figure)
            self.canvas = None

        # Clear all other widgets
        for widget in self.display_frame.winfo_children():
            widget.destroy()

    def add_info_label(self, text):
        """Add information label to display frame"""
        info_label = ttk.Label(self.detail_panel, text=text)
        info_label.pack(side='top', padx=5, pady=5, anchor='w')

    def bind_events(self, handlers):
        """Bind event handlers from controller"""
        self.event_handlers = handlers

    def update_file_label(self, filename):
        """Update the file label with new filename"""
        self.file_label.config(text=filename)


    @staticmethod
    def update_theme_indicator(theme):
        """Update UI based on theme change"""
        print(f"Theme updated to: {theme}")

    @staticmethod
    def show_message(message, mess_type="info"):
        """Show message to user"""
        if mess_type == "info":
            messagebox.showinfo("Info", message)
        elif mess_type == "error":
            messagebox.showerror("Error", message)