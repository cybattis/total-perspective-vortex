# models/app_model.py
from settings import Settings

class AppModel:
    """Handles data and business logic"""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.theme = settings.get('theme') or "dark"
        self.current_file = settings.get('last_dataset') or None
        self.dataset = None
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, event_type, data=None):
        for observer in self._observers:
            observer.on_model_change(event_type, data)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.settings.set('theme', self.theme)
        self.notify_observers("theme_changed", self.theme)

    def set_dataset(self, file_path):
        self.current_file = file_path
        # Load actual dataset here
        self.settings.set('last_dataset', self.current_file)
        self.notify_observers("dataset_loaded", file_path)
