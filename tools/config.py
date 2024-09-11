import logging

import yaml
from pathlib import Path
import logging


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

import os.path as osp
from typing import Dict, Any, Union, IO
from yacs.config import CfgNode as _CfgNode
from iopath.common.file_io import g_pathmgr


class CfgNode(_CfgNode):
    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        return g_pathmgr.open(filename, "r")

    @classmethod
    def load_yaml_with_base(cls,
                            filename: str, allow_unsafe: bool = False
                            ) -> Dict[str, Any]:
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "warning config"
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)

        def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        return cfg

    def merge_from_file(self, cfg_filename, allow_unsafe: bool = True) -> None:
        assert osp.isfile(cfg_filename), f"Config file'{cfg_filename}'does not exist !"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)


def get_cfg() -> CfgNode:
    from config.default.default import _C

    return _C.clone()
