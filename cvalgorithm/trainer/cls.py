import torch
from torch.utils.data import DataLoader


class ClsTrainer:

    def __init__(self, cfg):
        super(ClsTrainer, self).__init__()
        self.epoch = cfg.epochs

    @staticmethod
    def build_train_dataloader(train_dataset):
        return DataLoader(train_dataset)




    def train(self):

        schedule_idx = 0


        pass

