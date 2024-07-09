from pathlib import Path
from typing import Union
import os
import cv2
import numpy as np
from PIL import Image
import logging
import sys

fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()

img_fp = '/workspace/datasets/FPCD2022Dataset/train/images/004813b.png'
img_dir = '/workspace/datasets/FPCD2022Dataset/train/images'


def time_compute(func):
    pass


def crop_defect(images):
    pass


def get_default_ort_providers():
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError('Please install onnxruntime first:\n'
                          '  - For CPU only, use `pip install onnxruntime`;\n'
                          '  - For GPU, use `pip install onnxruntime-gpu`.')

    providers = []
    if 'CPUExecutionProvider' in ort.get_available_providers():
        providers.append('CPUExecutionProvider')
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')  # GPU优先
    if len(providers) == 0:
        providers = ort.get_available_providers()
    return providers


class OCRInferencePipeLine:
    def __init__(self, rec_model="/workspace/weights/en_PP-OCRv3_rec_infer.onnx",
                 det_model='/workspace/weights/en_PP-OCRv3_det_infer.onnx', **kwargs):
        super(OCRInferencePipeLine, self).__init__()
        self.rec_model, self.det_model = rec_model, det_model
        (self.rec_predictor, self.rec_input_tensor, self.rec_output_tensor, self.rec_config) = self.create_predictor(
            self.rec_model, 'rec', ort_providers=kwargs.get('ort_providers', 'CPUExecutionProvider')
        )
        (self.det_predictor, self.det_input_tensor, self.det_output_tensor, self.det_config) = self.create_predictor(
            self.det_model, 'rec', ort_providers=kwargs.get('ort_providers', 'CPUExecutionProvider')
        )
        self.preprocess_op = None

    @staticmethod
    def create_predictor(model_path, mode, ort_providers=None):
        import onnxruntime as ort

        if model_path is None:
            logger.info("not find {} model file path {}".format(mode, model_path))
            sys.exit(0)
        model_file_path = model_path
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path")

        if ort_providers is None:
            ort_providers = get_default_ort_providers()
        logger.debug(f'ort providers: {ort_providers}')
        sess = ort.InferenceSession(model_file_path, providers=ort_providers)
        return sess, sess.get_inputs()[0], None, None

    @staticmethod
    def create_operators(op_param_list, global_config=None):
        assert isinstance(op_param_list, list), 'operator config should be a list'
        ops = []
        for operator in op_param_list:
            assert isinstance(operator, dict) and len(operator) == 1, 'format error'
            param = operator[list(operator)[0]]
            if global_config is not None:
                param.update(global_config)
            op = eval(list(operator)[0])(**param)
            ops.append(op)
        return ops

    def detect(self, img: Union[str, Path, Image.Image, np.ndarray], **det_kwargs):
        img = self._preprocess_image(img)
        ori_im = img.copy()
        data = {'image': img}

        if self.preprocess_op is not None:
            # do transform
            pass
        img, shape_list = data
        if img is None:
            logger.warn("image data is empty, please load correct image data")
            return None, 0
        pass

    @classmethod
    def _preprocess_image(cls, img: Union[str, Path, Image.Image, np.ndarray]
                          ) -> np.ndarray:
        if isinstance(img, (str, Path)):
            if not os.path.isfile(img):
                raise FileNotFoundError(img)
            return cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, Image.Image):
            img = np.asarray(img.convert('RGB'), dtype='float32')
        if isinstance(img, np.ndarray):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError('type %s is not supported now !')


def cycle_load_file(path):
    image_paths = []
    image_extension = ['.png']
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extension:
                image_paths.append(os.path.join(root, file))

    return image_paths


def ocr_infer(img_fp, rec_model= None, det_model=None):
    ocr = OCRInferencePipeLine()
    out = ocr.detect(img_fp, rec_batch_size=1, return_cropped_image=False)
    return out


def predict(out_dict: list, save_root="/workspace/datasets/results"):
    for out in out_dict:
        file, predict = out['file'], out['predict']
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        for pre in predict:
            pt1 = (int(pre['position'][0][0]), int(pre['position'][0][1]))
            pt2 = (int(pre['position'][2][0]), int(pre['position'][2][1]))
            cv2.rectangle(image, pt1, pt2, color=(255, 192, 203), thickness=2)
            cv2.putText(image, text=pre['text'], org=pt1, color=(255, 192, 203),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2)
        cv2.imwrite(os.path.join(save_root, os.path.basename(file)), image)
        pass


file_list = cycle_load_file(img_dir)
filter_images = [file for file in file_list if file.endswith('b.png')]
out_dict = list()
for idx, file in enumerate(filter_images):

    out = ocr_infer(file)
    out_dict.append(dict(file=file, predict=out))

predict(out_dict)
pass