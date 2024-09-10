from cvalgorithm.utils import get_cfg, DEFAULT_CFG


class Trainer:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)

    #
    def add_callback(self, event: str, callback):
        pass

    def train(self):
        if isinstance(self.args.device, str) and len(self.args.device):
            pass
