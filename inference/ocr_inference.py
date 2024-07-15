import copy
from pathlib import Path
from typing import Tuple, Union, List, Dict, Any, Optional, Collection
import os

from functools import cmp_to_key
import cv2
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass
import logging
import sys
import torch
from copy import deepcopy

from cvalgorithm.datasets.PIL_transformer import *
from cvalgorithm.ops.post.db_postprocess import DBPostProcess, DistillationDBPostProcess

fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()

img_fp = '/workspace/datasets/FPCD2022Dataset/train/images/004813b.png'
img_dir = '/workspace/datasets/FPCD2022Dataset/train/images'

PRE_LIST = [
    {
        'DetResizeForTest': {
            'limit_side_len': 960,
            'limit_type': "max",
        }
    },
    {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225],
            'mean': [0.485, 0.456, 0.406],
            'scale': '1./255.',
            'order': 'hwc',
        }
    },
    {'ToCHWImage': None},
    {'KeepKeys': {'keep_keys': ['image', 'shape']}},
]

POST_LIST = {
    'name': 'DBPostProcess',
    "max_candidates": 1000,
    "unclip_ratio": 1.5,
    "use_dilation": False,
    "score_mode": 'fast',
    "bin_thresh": 0.3,
    "box_thresh": 0.6,
}

REC_POST_LIST = {
    'name': 'CTCLabelDecode',
    'character_dict_path': "/workspace/CnOCR/cnocr/ppocr/utils/en_dict.txt",
    'use_space_char': True,
    'cand_alphabet': None,
}


def time_compute(func):
    pass


def crop_defect(images):
    pass


@dataclass
class OcrResult(object):
    text: str
    score: float
    position: Optional[np.ndarray] = None
    cropped_img: np.ndarray = None

    def to_dict(self):
        res = deepcopy(self.__dict__)
        if self.position is None:
            res.pop('position')
        if self.cropped_img is None:
            res.pop('cropped_img')
        return res


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
                 det_model='/workspace/weights/en_PP-OCRv3_det_infer.onnx', rec_image_shape="3, 32, 320",  **kwargs):
        super(OCRInferencePipeLine, self).__init__()
        self.rec_model, self.det_model = rec_model, det_model
        (self.rec_predictor, self.rec_input_tensor, self.rec_output_tensor, self.rec_config) = self.create_predictor(
            self.rec_model, 'rec', ort_providers=kwargs.get('ort_providers', 'CPUExecutionProvider')
        )
        (self.det_predictor, self.det_input_tensor, self.det_output_tensor, self.det_config) = self.create_predictor(
            self.det_model, 'det', ort_providers=kwargs.get('ort_providers', 'CPUExecutionProvider')
        )
        self.operator_list = PRE_LIST
        self.preprocess_op = self.create_operators(self.operator_list)
        self.postprocess_params = POST_LIST
        self.postprocess_op = self.create_post_process(self.postprocess_params)
        self.rec_postprocess_params = REC_POST_LIST
        self.rec_postprocess_op = self.create_post_process(self.rec_postprocess_params)
        self.rec_algorithm = 'CRNN'
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]

    @staticmethod
    def create_post_process(config, global_config=None):
        support_dict = [
            'DBPostProcess',
            'ClsPostProcess',
            'DistillationDBPostProcess',
            'CTCLabelDecode'
        ]

        config = copy.deepcopy(config)
        module_name = config.pop('name')
        if module_name == "None":
            return
        if global_config is not None:
            config.update(global_config)
        assert module_name in support_dict, Exception(
            f'post process not support {module_name}'
        )
        module_class = eval(module_name)(**config)
        return module_class

    @staticmethod
    def create_predictor(model_path, mode, ort_providers=None):
        import onnxruntime as ort

        if model_path is None:
            logger.info("not find {} model file path {}".format(mode, model_path))
            sys.exit(0)
        model_file_path = model_path
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path")

        # if ort_providers is None:
        #     ort_providers = get_default_ort_providers()
        logger.debug(f'ort providers: {ort_providers}')
        sess = ort.InferenceSession(model_file_path, providers=get_default_ort_providers())
        return sess, sess.get_inputs()[0], None, None

    @staticmethod
    def create_operators(op_param_list, global_config=None):
        assert isinstance(op_param_list, list), 'operator config should be a list'
        ops = []
        for operator in op_param_list:
            assert isinstance(operator, dict) and len(operator) == 1, 'format error'
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            if global_config is not None:
                param.update(global_config)
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    def __call__(self, img: Union[str, Path, Image.Image, np.ndarray], return_cropped_image=False, **kwargs):
        # one step
        box_infos = self.detect(img, box_score_thresh=kwargs.get('box_score_thresh', 0.5),
                                min_box_size=kwargs.get('min_box_size', 4))
        cropped_img_list = [
            box_info['cropped_img'] for box_info in box_infos['detected_texts']
        ]
        ocr_outs = self.reg(
            cropped_img_list, batch_size=kwargs.get('rec_batch_size', 1)
        )
        results = []
        for box_info, ocr_out in zip(box_infos['detected_texts'], ocr_outs):
            _out = OcrResult(**ocr_out)
            _out.position = box_info['box']
            if return_cropped_image:
                _out.cropped_img = box_info['cropped_img']
            results.append(_out.to_dict())

        return results

    def reg(
        self,
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Batch recognize characters from a list of one-line-characters images.

        Args:
            img_list (List[Union[str, Path, torch.Tensor, np.ndarray]]):
                list of images, in which each element should be a line image array,
                with type torch.Tensor or np.ndarray.
                Each element should be a tensor with values ranging from 0 to 255,
                and with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            list of detected texts, which element is a dict, with keys:
                - 'text' (str): 识别出的文本
                - 'score' (float): 识别结果的得分（置信度），取值范围为 `[0, 1]`；得分越高表示越可信

            示例：
            ```
             [{'score': 0.8812797665596008,
               'text': '第一行'},
              {'score': 0.859879732131958,
               'text': '第二行'},
              {'score': 0.7850906848907471,
               'text': '第三行'}
             ]
            ```
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]

        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]

        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        for beg_img_no in range(0, img_num, batch_size):
            end_img_no = min(img_num, beg_img_no + batch_size)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm != "SRN" and self.rec_algorithm != "SAR":
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = dict()
            input_dict[self.rec_input_tensor.name] = norm_img_batch
            outputs = self.rec_predictor.run(self.rec_output_tensor, input_dict)
            preds = outputs[0]

            rec_result = self.rec_postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        # TODO add recognize
        results = []
        for text, score in rec_res:
            _out = OcrResult(text=text, score=score)
            results.append(_out.to_dict())

        return results

    def detect(self, img: Union[str, Path, Image.Image, np.ndarray], **detect_kwargs):
        img = self._preprocess_image(img)
        ori_im = img.copy()
        data = {'image': img}

        if self.preprocess_op is not None:
            # do transform
            for op in self.preprocess_op:
                data = op(data)
                if data is None:
                    logger.warn("image data warning !")
        img, shape_list = data
        if img is None:
            logger.warn("image data is empty, please load correct image data")
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        # first step
        input_dict = {}
        input_dict[self.det_input_tensor.name] = img
        results = self.det_predictor.run(self.det_output_tensor, input_dict)

        preds = {'maps': results[0]}

        post_result = self.postprocess_op(
            preds, shape_list, box_thresh=detect_kwargs.get('box_score_thresh', 0.6)
        )
        dt_boxes = list(zip(post_result[0]['points'], post_result[0]['scores']))
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape, detect_kwargs.get("min_box_size", 0))
        dt_boxes = sort_boxes(dt_boxes, key=0)

        detected_results = []
        for bno in range(len(dt_boxes)):
            box, score = dt_boxes[bno]
            img_crop = get_rotate_crop_image(ori_im, deepcopy(box))
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            detected_results.append(
                {'box': box, 'score': score, 'cropped_img': img_crop.astype('uint8')}
            )

        return dict(rotated_angle=0.0, detected_texts=detected_results)

    def _prepare_img(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image array with type torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                channel should be 1 (gray image) or 3 (color image).

        Returns:
            np.ndarray: with shape (height, width, 1), dtype uint8, scale [0, 255]
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp, gray=False)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            assert img.shape[2] in (1, 3)

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def filter_tag_det_res(self, dt_boxes, image_shape, min_box_size):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box, score in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if min(rect_width, rect_height) < min_box_size:
                continue
            dt_boxes_new.append((box, score))
        # dt_boxes = np.array(dt_boxes_new)
        return dt_boxes_new

    @staticmethod
    def clip_det_res(points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    @staticmethod
    def order_points_clockwise(pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

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

    def resize_norm_img(self, img, max_wh_ratio):
        """

        Args:
            img (): with shape of (height, width, channel)
            max_wh_ratio ():

        Returns:

        """
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        # resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resize_img(
            img.transpose((2, 0, 1)), target_h_w=(imgH, resized_w), return_torch=False
        ).transpose((1, 2, 0))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(
            self,
            character_dict_path=None,
            use_space_char=False,
            cand_alphabet: Optional[Union[Collection, str]] = None,
    ):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)

    def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
        """
        设置待识别字符的候选集合。

        Args:
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围

        Returns:
            None

        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [
                word if word != ' ' else '<space>' for word in cand_alphabet
            ]
            excluded = set([word for word in cand_alphabet if word not in self.dict])
            if excluded:
                logger.warning(
                    'chars in candidates are not in the vocab, ignoring them: %s'
                    % excluded
                )
            candidates = [word for word in cand_alphabet if word in self.dict]
            self._candidates = None if len(candidates) == 0 else candidates
            logger.debug('candidate chars: %s' % self._candidates)

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                            idx > 0
                            and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


# rec post process
class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        **kwargs
    ):
        super(CTCLabelDecode, self).__init__(
            character_dict_path, use_space_char, cand_alphabet
        )

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        preds = mask_by_candidates(
            preds,
            self._candidates,
            self.character,
            self.dict,
            self.get_ignored_tokens(),
        )

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


def mask_by_candidates(
    logits: np.ndarray,
    candidates: Optional[Union[str, List[str]]],
    vocab: List[str],
    letter2id: Dict[str, int],
    ignored_tokens: List[int],
):
    if candidates is None:
        return logits

    _candidates = [letter2id[word] for word in candidates]
    _candidates.sort()
    _candidates = np.array(_candidates, dtype=int)

    candidates = np.zeros((len(vocab),), dtype=bool)
    candidates[_candidates] = True
    # candidates[-1] = True  # for cnocr, 间隔符号/填充符号，必须为真
    candidates[ignored_tokens] = True
    candidates = np.expand_dims(candidates, axis=(0, 1))  # 1 x 1 x (vocab_size+1)
    candidates = candidates.repeat(logits.shape[1], axis=1)

    masked = np.ma.masked_array(data=logits, mask=~candidates, fill_value=-100.0)
    logits = masked.filled()
    return logits


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img = np.clip(dst_img, 0.0, 255.0).astype('float32')
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def read_img(path: Union[str, Path], gray=True) -> np.ndarray:
    """
    :param path: image file path
    :param gray: whether to return a gray image array
    :return:
        * when `gray==True`, return a gray image, with dim [height, width, 1], with values range from 0 to 255
        * when `gray==False`, return a color image, with dim [height, width, 3], with values range from 0 to 255
    """
    if gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Error loading image: {path}')
        return np.expand_dims(img, -1)
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Error loading image: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compare_box(box1, box2, key):
    # 从上到下，从左到右
    # box1, box2 to: [xmin, ymin, xmax, ymax]
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]

    def y_iou():
        # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
        # 判断是否有交集
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return 0
        # 计算交集的高度
        y_min = max(box1[1], box2[1])
        y_max = min(box1[3], box2[3])
        return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))

    if y_iou() > 0.5:
        return box1[0] - box2[0]
    else:
        return box1[1] - box2[1]


def resize_img(
    img: np.ndarray,
    target_h_w: Optional[Tuple[int, int]] = None,
    min_width: int = 8,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    rescale an image tensor with [Channel, Height, Width] to the given height value, and keep the ratio
    :param img: np.ndarray; should be [c, height, width]
    :param target_h_w: (height, width) of the target image or None
    :param min_width: int; minimum width after resized. Only used when `target_h_w` is None. Default 8
    :param return_torch: bool; whether to return a `torch.Tensor` or `np.ndarray`
    :return: image tensor with the given height. The resulting dim is [C, height, width]
    """
    ori_height, ori_width = img.shape[1:]
    if target_h_w is None:
        ratio = ori_height / 32
        target_w = max(int(ori_width / ratio), min_width)
        target_h_w = (32, target_w)

    if (ori_height, ori_width) != target_h_w:
        # img = F.resize(torch.from_numpy(img), target_h_w, antialias=True)
        new_img = cv2.resize(img.transpose((1, 2, 0)), (target_h_w[1], target_h_w[0]))  # -> (H, W, C)
        if img.ndim > new_img.ndim:
            new_img = np.expand_dims(new_img, axis=-1)
        img = new_img.transpose((2, 0, 1))  # -> (C, H, W)
    if return_torch:
        img = torch.from_numpy(img)
    return img


def sort_boxes(
    dt_boxes: List[Union[Dict[str, Any], Tuple[np.ndarray, float]]],
    key: Union[str, int] = 'box',
) -> List[Union[Dict[str, Any], Tuple[np.ndarray, float]]]:
    """
    Sort resulting boxes in order from top to bottom, left to right
    args:
        dt_boxes(array): list of dict or tuple, box with shape [4, 2]
    return:
        sorted boxes(array): list of dict or tuple, box with shape [4, 2]
    """
    _boxes = sorted(dt_boxes, key=cmp_to_key(lambda x, y: _compare_box(x, y, key)))
    return _boxes


def cycle_load_file(path):
    image_paths = []
    image_extension = ['.png']
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extension:
                image_paths.append(os.path.join(root, file))

    return image_paths


def ocr_infer(img_fp, rec_model=None, det_model=None):
    ocr = OCRInferencePipeLine()
    out = ocr(img_fp, box_score_thresh=0.6, min_box_size=4,  rec_batch_size=1, return_cropped_image=False)
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
