
class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        # 执行一些初始化操作，比如打开资源
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")
        # 执行一些清理操作，比如关闭资源
        if exc_type is not None:
            print(f"Exception: {exc_type}, {exc_value}")
        # 返回 True 表示异常已经被处理，返回 False 则异常会被传播
        return False

# 使用上下文管理器
with MyContextManager() as manager:
    raise Exception('Raised')