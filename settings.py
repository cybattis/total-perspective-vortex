import json

class Settings:
    def __init__(self, config: dict):
        self.config = config
        self.filepath = "config.json"

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value

    def save(self):
        print("Saving settings to", self.filepath)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)

    def load(self):
        print("Loading settings from", self.filepath)
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def __repr__(self):
        return f"Settings({self.config})"