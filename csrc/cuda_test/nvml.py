from contextlib import contextmanager
import pynvml


@contextmanager
def _nvml():
    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()


@_nvml()
def func():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(handle)
    pass


if __name__ == "__main__":
    func()