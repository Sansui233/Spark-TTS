import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[Info] {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result

    return wrapper


class Timer:
    def __init__(self):
        self.start_time = time.perf_counter()

    def print(self, prefix: str):
        current_time = time.perf_counter()
        print(f"[Info] {prefix} 执行耗时: {current_time - self.start_time:.4f} 秒")

    def clear(self):
        self.start_time = time.perf_counter()

    def end(self, prefix: str):
        self.print(prefix)
        self.clear()
