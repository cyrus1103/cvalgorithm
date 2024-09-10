import yaml
from pathlib import Path


def yaml_model_load(path):
    import re

    def yaml_load(file, append_file_name: bool = True):
        assert Path(file).suffix in (".yaml", ".yml")
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()
            if not s.isprintable():
                s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = yaml.safe_load(s) or {}

            if append_file_name:
                data['yaml_file'] = str(file)

        return data

    d = yaml_load(path)
    d["yaml_file"] = str(path)
    return d


class DictToAttributes:
    """ A simple class for dict2attr """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)