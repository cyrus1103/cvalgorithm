import torch
import numpy as np
from cvalgorithm.datasets.transformer import Transformer


class FourierFilterLower(Transformer):
    def __init__(self, band_width, radius_ratio=0.5):
        super(FourierFilterLower, self).__init__()
        self.band_width = band_width
        self.radius_ratio = radius_ratio

    def get_low_high(self, img):
        f = np.fft.fftn(img)
        fshift = np.fft

    def core(self):
        pass

    def apply(self, img):
        img_size = img.shape[-1]
        band_w = self.band_width // 2

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # 构建掩模，只保留低频成分
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        # 应用掩模
        fshift *= mask

        # 反傅里叶变换
        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift)
        img_back = np.abs(img_back)
        return img_back


def freq_crop(input_t, band_width=None):

    img_size = input_t.size(-1)
    band_w = band_width // 2

    img_f = torch.fft.fftn(input_t)

    img_crop = torch.empty(
        [input_t.size(0), input_t.size(1), band_w * 2, band_w * 2],
        dtype=img_f.dtype, device=img_f.device
        )

    img_crop[:, :, :band_w, :band_w] = img_f[:, :, :band_w, :band_w]
    img_crop[:, :, -band_w:, :band_w] = img_f[:, :, -band_w:, :band_w]
    img_crop[:, :, :band_w, -band_w:] = img_f[:, :, :band_w, -band_w:]
    img_crop[:, :, -band_w:, -band_w:] = img_f[:, :, -band_w:, -band_w:]

    img_crop = img_crop * ((band_w * 2 / img_size) ** 2)

    img_crop = torch.fft.ifftn(img_crop)
    img_crop = torch.real(img_crop)

    return img_crop


if __name__ == "__main__":
    fourier = FourierFilterLower(band_width=512)
    import cv2
    import torch
    img = cv2.imread("/workspace/datasets/text_broke_crop/004682b.png", cv2.IMREAD_UNCHANGED)
    img_torch = torch.tensor(img, dtype=torch.float32).transpose(0, -1).unsqueeze(0).unsqueeze(0)
    img_torch = freq_crop(img_torch, band_width=224)
    low_img_torch = np.array(img_torch.squeeze(0).transpose(0, -1), dtype=np.float32)
    img_lower = fourier(img)
    pass
