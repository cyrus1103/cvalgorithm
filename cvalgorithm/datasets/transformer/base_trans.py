
__all__ = ["Transformer"]


class Transformer(object):
    def __init__(self):
        pass

    def apply(self, img):
        pass

    def __call__(self, img, *args, **kwargs):
        img = self.apply(img)
        return img