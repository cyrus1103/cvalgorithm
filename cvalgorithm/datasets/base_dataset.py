from nvidia.dali.pipeline import pipeline_def
from torch.utils.data import dataset
import nvidia.dali.fn as fn
# TODO
# use albumentations
# use nvidia dali
class BaseDataset(dataset):
    def __init__(self):
        pass


class NVIDIADataset(BaseDataset):
    def __init__(self):
        super(NVIDIADataset, self).__init__()

    @pipeline_def(num_threads=4, device_id=0)
    def get_dali_pipeline(self):
        images, labels = fn.readers.file(file_root=self.images_dir, random_shuffle=True, name="Reader")






