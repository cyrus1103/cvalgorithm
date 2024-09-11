from tools.config import CfgNode


_C = CfgNode()


_C.VERSION = 1.0

_C.TYPE = 'SEG'

_C.MODEL = CfgNode()
_C.MODEL.BACKBONE = 'ConvNexT'