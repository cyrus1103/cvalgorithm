from collections import OrderedDict


class EasyDict(OrderedDict):

    def __call__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass