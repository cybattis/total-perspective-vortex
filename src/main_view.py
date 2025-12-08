# views/main_view.py
from tkinter import ttk, messagebox, filedialog
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.psd_canvas = None
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
        self.root.columnconfigure(1, weight=5)
        # Bind window close event to exit handler
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._trigger_event('exit_app'))


    def _setup_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load dataset", command=lambda: self._trigger_event('load_dataset',
                                                self.show_file_dialog(self.root_folder + "/datasets")))
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
        side_panel.rowconfigure(0, weight=2)
        side_panel.rowconfigure(1, weight=1)
        side_panel.columnconfigure(0, weight=1)

        # Configuration Frame
        upper_frame = ttk.Frame(side_panel, padding=10, borderwidth=2, relief="ridge")
        upper_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        ttk.Label(upper_frame, text="Configuration").pack(pady=5, side='top')

        # Dataset label
        file_frame = ttk.Frame(upper_frame)
        file_frame.pack(anchor="w", pady=10)
        ttk.Label(file_frame, text="Dataset: ").pack(side='left')
        self.file_label = ttk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.pack(side='left', padx=8)

        # Visualization
        visu_frame = ttk.Frame(upper_frame)
        visu_frame.pack(anchor="w")
        # Visualize button
        ttk.Button(
            visu_frame, text="Visualize",
            command=lambda: self._trigger_event('visualize_dataset')
        ).pack(side='left')
        # Clear button
        ttk.Button(
            visu_frame, text="X",
            command=lambda: self._trigger_event('reset_dataset')
        ).pack(side='left', padx=8)

        # Preprocessing
        preprocess_frame = ttk.Frame(upper_frame)
        preprocess_frame.pack(pady=10, fill='x', expand=True, side='top', anchor='n')
        ttk.Label(preprocess_frame, text="Preprocessing", padding=5).pack(anchor='n')

        # Filter cutoff scale
        cutoff_frame = ttk.Frame(preprocess_frame)
        cutoff_frame.pack(side='top', fill='x', pady=2)
        ttk.Label(cutoff_frame, text="Low pass filter (Hz):").pack(side='top', anchor='w')
        cutoff_var = tk.DoubleVar(value=1.0)
        cutoff_scale = ttk.Scale(
            cutoff_frame,
            from_=0.5,
            to=100.0,
            orient='horizontal',
            variable=cutoff_var,
            command=lambda v: cutoff_value_label.config(text=f"{float(v):.1f} Hz")
        )
        cutoff_scale.pack(side='left', fill='x', expand=True, padx=5)
        cutoff_value_label = ttk.Label(cutoff_frame, text=f"{cutoff_var.get():.1f} Hz")
        cutoff_value_label.pack(side='top')

        ttk.Button(
            preprocess_frame,
            text="Notch Filter (50 Hz)",
            command=lambda: self._trigger_event('preprocess_dataset', 'notch')
        ).pack(side='top', fill='x', pady=2)
        ttk.Button(
            preprocess_frame,
            text="Resample (100 Hz)",
            command=lambda: self._trigger_event('preprocess_dataset', 'resample')
        ).pack(side='top', fill='x', pady=2)
        # ttk.Button(
        #     preprocess_frame,
        #     text="Remove EOG Channels",
        #     command=lambda: self._trigger_event('preprocess_dataset', 'remove_eog')
        # ).pack(side='top', fill='x', pady=2)
        # ttk.Button(
        #     preprocess_frame,
        #     text="Remove ECG Channels",
        #     command=lambda: self._trigger_event('preprocess_dataset', 'remove_ecg')
        # ).pack(side='top', fill='x', pady=2)
        # ttk.Button(
        #     preprocess_frame,
        #     text="ICA Artifact Removal",
        #     command=lambda: self._trigger_event('preprocess_dataset', 'ica')
        # ).pack(side='top', fill='x', pady=2)
        # ttk.Button(
        #     preprocess_frame,
        #     text="Re-reference to Average",
        #     command=lambda: self._trigger_event('preprocess_dataset', 'rereference')
        # ).pack(side='top', fill='x', pady=2)

        # Refresh button
        ttk.Button(
            preprocess_frame,
            text="Apply",
            command=lambda: self._trigger_event('apply_preprocessing')
        ).pack(side='left', pady=10)

        # Details frames
        self.detail_panel = ttk.Frame(side_panel, padding=10, borderwidth=2, relief="ridge")
        self.detail_panel.grid(row=1, column=0, sticky="nsew")

        ttk.Label(self.detail_panel, text="Details", anchor="w").pack(padx=5)

        # Tabbed view for main content
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=1, sticky="nsew")

        # Processing Tab
        self.display_frame = ttk.Frame(notebook, borderwidth=2, relief="ridge")
        self.display_frame.pack(fill='both', expand=True)
        notebook.add(self.display_frame, text="Processing")
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        ttk.Label(self.display_frame, text="Visual Display Area").pack(anchor="center", expand=True)

        # PSD Tab
        self.psd_frame = ttk.Frame(notebook, borderwidth=2, relief="ridge")
        self.psd_frame.pack(fill='both', expand=True)
        notebook.add(self.psd_frame, text="Power Spectral Density")
        self.psd_frame.rowconfigure(0, weight=1)
        self.psd_frame.columnconfigure(0, weight=1)
        ttk.Label(self.psd_frame, text="PSD Area").pack(anchor="center", expand=True)

        # Training and Validation Tab
        training_tab = ttk.Frame(notebook, borderwidth=2, relief="ridge")
        training_tab.pack(fill='both', expand=True)
        notebook.add(training_tab, text="Training and Validation")
        ttk.Label(training_tab, text="Training and Validation Area").pack(anchor="center", expand=True)


    def _trigger_event(self, event_name, *args):
        """Trigger an event handler if it exists"""
        if event_name in self.event_handlers:
            self.event_handlers[event_name](*args)


    # Public Methods
    # =================================================
    def bind_events(self, handlers):
        """Bind event handlers from controller"""
        self.event_handlers = handlers

    def clear_display_frame(self):
        """Clear all widgets from display frame and return the figure to be closed."""
        fig = None
        if hasattr(self, 'canvas') and self.canvas:
            fig = self.canvas.figure
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        return fig

    def clear_psd_frame(self):
        """Clear all widgets from the PSD frame and return the figure to be closed."""
        fig = None
        if hasattr(self, 'psd_canvas') and self.psd_canvas:
            fig = self.psd_canvas.figure
            self.psd_canvas.get_tk_widget().destroy()
            self.psd_canvas = None
        for widget in self.psd_frame.winfo_children():
            widget.destroy()
        return fig

    def add_info_label(self, text):
        """Add information label to display frame"""
        # Clear previous details
        for widget in self.detail_panel.winfo_children():
            if isinstance(widget, ttk.Label) and widget != self.detail_panel.winfo_children()[0]:
                widget.destroy()

        info_label = ttk.Label(self.detail_panel, text=text)
        info_label.pack(side='top', padx=5, pady=5, anchor='w')

    def update_file_label(self, filename):
        """Update the file label with new filename"""
        self.file_label.config(text=filename)

    def embed_figure_in_display_frame(self, fig):
        """Embed a matplotlib figure into the display frame"""
        self.clear_display_frame()
        self.canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def embed_psd_figure(self, fig):
        """Embed a matplotlib figure into the PSD frame."""
        self.clear_psd_frame()
        self.psd_canvas = FigureCanvasTkAgg(fig, master=self.psd_frame)
        self.psd_canvas.draw()
        self.psd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Static Methods
    # =================================================
    @staticmethod
    def show_file_dialog(initial_dir):
        """Open a file dialog and return the selected path."""
        return filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Dataset",
            filetypes=(("EDF files", "*.edf"), ("All files", "*.*"))
        )

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