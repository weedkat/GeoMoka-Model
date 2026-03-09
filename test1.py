DEFAULT = {}

def decorator(func):
    DEFAULT[func.__name__] = func
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def build(key):
    return DEFAULT.get(key, lambda: f"No {key} defined")

