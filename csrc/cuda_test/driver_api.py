import ctypes
import pynvml

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
def report_used_memory(index=0, msg=""):
    # Get the handle for the same device using NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    # Get memory information
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"{msg}: Used memory: {info.used / 1024 ** 2} MB")

# Load the CUDA driver library
cuda = ctypes.CDLL('libcuda.so')

report_used_memory(msg="before cuinit")

# Initialize the CUDA Driver API
result = cuda.cuInit(0)
if result != 0:
    raise Exception("Failed to initialize the CUDA driver API")

report_used_memory(msg="after cuinit")

device = 0
# Create contexts on device 0
contexts = []
for i in range(3):
    context = ctypes.c_void_p()
    result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
    if result != 0:
        raise Exception("Failed to create one or both CUDA contexts")
    report_used_memory(msg=f"after creating {i + 1} cuda context")
    contexts.append(context)