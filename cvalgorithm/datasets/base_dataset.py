from nvidia.dali.pipeline import pipeline_def
from torch.utils.data import dataset
import nvidia.dali.fn as fn
from abc import ABC

# TODO
# use albumentations
# use nvidia dali


class BaseDataset(dataset):
    def __init__(self, indices):
        self.indices = indices

    @property
    def _get_indices(self):
        return list(self[i] for i in self.indices)


class NVIDIADataset(BaseDataset):
    def __init__(self):
        super(NVIDIADataset, self).__init__()

    @pipeline_def(num_threads=4, device_id=0)
    def get_dali_pipeline(self):
        images, labels = fn.readers.file(file_root=self.images_dir, random_shuffle=True, name="Reader")


class ClsDataset(BaseDataset):
    pass



class _CloudStorage(ABC):
    def download_obj(self, key):
        pass


    def download_file(self, key, path):
        file_obj = self.download_obj()