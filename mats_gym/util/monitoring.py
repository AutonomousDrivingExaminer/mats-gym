import time


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__}: {duration:.5f}s, {1/duration*0.05:.2f}fps")
        return result

    return wrapper