import json
import os

from src.main_controller import log
from utils import Utils


class Settings:
    def __init__(self):
        self.filepath = "config.json"
        self.data = self.load_config()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self.save()

    def save(self):
        print("Saving settings to", self.filepath)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
            f.close()

    @staticmethod
    def load_config():
        try:
            cfg_path = os.path.join(Utils.get_root_folder_path(), "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            log.info("Loaded config: %s", cfg_path)
            return config
        except Exception as e:
            log.error("Could not load config.json: %s", e)
            exit(1)

    def __repr__(self):
        return f"Settings({self.data})"