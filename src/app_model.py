# models/app_model.py
from settings import Settings

class AppModel:
    """Handles data and business logic"""
    def __init__(self, settings: Settings):
        self.current_file = settings.get('last_dataset') or None
        self.settings = settings
        self.dataset = None
        self.theme = settings.get('theme') or "dark"
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, event_type, data=None):
        for observer in self._observers:
            observer.on_model_change(event_type, data)

    def load_dataset(self, file_path):
        self.current_file = file_path
        # Load actual dataset here
        self.settings.set('last_dataset', self.current_file)
        self.settings.save()
        self.notify_observers("dataset_loaded", file_path)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.notify_observers("theme_changed", self.theme)
        self.settings.set('theme', self.theme)
        self.settings.save()
