# models/app_model.py
class AppModel:
    """Handles data and business logic"""
    def __init__(self):
        self.current_file = None
        self.dataset = None
        self.theme = "dark"
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, event_type, data=None):
        for observer in self._observers:
            observer.on_model_change(event_type, data)

    def load_dataset(self, file_path):
        self.current_file = file_path
        # Load actual dataset here
        self.notify_observers("dataset_loaded", file_path)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.notify_observers("theme_changed", self.theme)
