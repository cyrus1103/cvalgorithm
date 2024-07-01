from typing import TypeVar, List, Union

# 通过TypeVar限定为整数型的列表和浮点数的列表
T = TypeVar("T", bound=Union[List[int], List[float]])
# 也可以写成如下形式
T = TypeVar("T", List[int], List[float])

def printList(l: T):
    for e in l:
        print(e)

printList([1, 2, 3])           # 打印整数型列表
printList([1.1, 2.2, 3.3])     # 打印浮点数列表
printList(["a", "b", "c"])